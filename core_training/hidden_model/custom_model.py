import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import random
import os
import json
from typing import Optional, Any, List, Dict, Tuple
from torch.nn.utils.rnn import pad_sequence

IGNORE_TOKEN_ID = -100
IGNORE = -100
EPS = 1e-8

class AdaptiveProjection(nn.Module):
    """Adaptive numerical range projection layer"""

    def __init__(self, hidden_size):
        super().__init__()
        # Learnable scaling factors
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        # Dynamic range adaptation layer
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self._init_weights()

    def _init_weights(self):
        # First layer: small range initialization
        nn.init.normal_(self.proj[0].weight, mean=0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)

        # Second layer: standard initialization
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x):
        # Three-step processing pipeline
        residual = x * self.scale  # Preserve original scaled signal
        x = self.proj(residual)  # Feature transformation
        return (residual + x) * self.output_scale  # Residual connection + calibration


class ModelWithInsertedHiddenState(nn.Module):
    def __init__(self, base_model, prepended_length, hidden_size, prepended_learnable=False, num_heads=8,
                 plan_similarity_weight=0.5, random_contrast_weight=1.5, prepended_input_dim=None):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.step_count = 0
        self.prepended_length = prepended_length
        self.prepended_learnable = prepended_learnable
        self.config = base_model.config
        self.tokenizer = None
        self.ratio_list = []

        self.contrastive_weight = 0.1
        self.plan_similarity_weight = plan_similarity_weight
        self.random_contrast_weight = random_contrast_weight

        self.prepended_input_dim = prepended_input_dim
        if prepended_input_dim is not None and prepended_input_dim != hidden_size:
            self.input_projector = nn.Linear(prepended_input_dim, hidden_size, bias=True)
        else:
            self.input_projector = None

        # MHA layer configuration
        self.hidden_mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self._init_mha_weights()

        # Normalization layers
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6)

        # Adaptive projection layer
        self.adaptive_proj = AdaptiveProjection(hidden_size)

        # Learnable scaling factors (obtained from adaptive_proj)
        self.scale = self.adaptive_proj.scale
        self.output_scale = self.adaptive_proj.output_scale

        # Learnable default hidden state if needed (as fallback)
        if prepended_learnable:
            self.default_prepended_hidden_state = nn.Parameter(
                torch.randn(prepended_length, hidden_size) * 0.02
            )
        else:
            self.register_buffer(
                'default_prepended_hidden_state',
                torch.zeros(prepended_length, hidden_size)
            )

    def _init_mha_weights(self):
        """Initialize MHA layer weights"""
        registered_params = self.hidden_mha._parameters.keys()

        for param_name in ['q_proj_weight', 'k_proj_weight', 'v_proj_weight']:
            if param_name in registered_params:
                param = getattr(self.hidden_mha, param_name)
                if param is not None:
                    nn.init.xavier_uniform_(param, gain=1.0 / math.sqrt(3))

        if 'in_proj_weight' in registered_params:
            param = self.hidden_mha._parameters['in_proj_weight']
            if param is not None:
                nn.init.xavier_uniform_(param, gain=1.0 / math.sqrt(3))

        for bias_name in ['q_proj_bias', 'k_proj_bias', 'v_proj_bias', 'in_proj_bias']:
            if bias_name in registered_params:
                param = getattr(self.hidden_mha, bias_name)
                if param is not None:
                    nn.init.constant_(param, 0.)

        if hasattr(self.hidden_mha, 'out_proj'):
            nn.init.xavier_uniform_(self.hidden_mha.out_proj.weight, gain=1.0)
            if self.hidden_mha.out_proj.bias is not None:
                nn.init.constant_(self.hidden_mha.out_proj.bias, 0.)

        # Ensure MHA parameters are trainable
        for name, param in self.hidden_mha.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True

    def process_hidden_states(self, x):
        # Ensure consistent device/dtype with layer parameters and contiguous memory
        dev = self.pre_ln.weight.device
        dtyp = self.pre_ln.weight.dtype
        x = x.to(device=dev, dtype=dtyp, non_blocking=True).contiguous()

        if self.input_projector is not None:
            x = self.input_projector(x)

        normed = self.pre_ln(x).contiguous()

        w_dtype = None
        ipw = getattr(self.hidden_mha, "in_proj_weight", None)
        if isinstance(ipw, torch.Tensor):
            w_dtype = ipw.dtype
        else:
            opw = getattr(getattr(self.hidden_mha, "out_proj", None), "weight", None)
            if isinstance(opw, torch.Tensor):
                w_dtype = opw.dtype
        if w_dtype is None:
            w_dtype = dtyp

        with torch.cuda.amp.autocast(enabled=False):
            q = normed.to(dtype=w_dtype).contiguous()
            k = normed.to(dtype=w_dtype).contiguous()
            v = normed.to(dtype=w_dtype).contiguous()

            attn_out, _ = self.hidden_mha(q, k, v, need_weights=False)

        attn_out = attn_out.to(dtyp)
        out = self.post_ln(normed + attn_out)
        projected = self.adaptive_proj(out)

        self.step_count += 1
        return projected

    def forward(self, input_tensors):
        """Complete forward pass (hidden processing only)"""
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            processed = self.process_hidden_states(input_tensors)
            out = torch.clamp(processed, -10.0, 10.0)
            return out

    def normalize_hidden_to_embedding_distribution(self, hidden_state, reference_embeds):
        ref_mean = reference_embeds.mean()
        ref_std = reference_embeds.std()
        hidden_mean = hidden_state.mean()
        hidden_std = hidden_state.std()
        normalized_hidden = (hidden_state - hidden_mean) / (hidden_std + 1e-8)
        adjusted_hidden = normalized_hidden * ref_std + ref_mean
        return adjusted_hidden

    def adaptive_mix_with_balanced_distribution(self, hidden_state, plan_embeds, mix_ratio):
        hidden_std = hidden_state.std() + 1e-8
        plan_std = plan_embeds.std() + 1e-8
        ratio = hidden_std / plan_std

        if mix_ratio == 0.0:
            result = plan_embeds
        elif mix_ratio == 1.0:
            result = hidden_state
        else:
            plan_len = plan_embeds.size(0)
            hidden_len = hidden_state.size(0)

            hidden_part_len = int(round(hidden_len * mix_ratio))
            plan_part_len = int(round(plan_len * mix_ratio))

            hidden_part = hidden_state[:hidden_part_len]
            plan_part = plan_embeds[plan_part_len:]

            result = torch.cat([hidden_part, plan_part], dim=0)

        return result

    def process_hidden_states_list(self, hidden_states_list):
        processed_list = []
        for hidden_state in hidden_states_list:
            if hidden_state is not None:
                processed = self.process_hidden_states(hidden_state.unsqueeze(0)).squeeze(0)
                processed_list.append(processed)
            else:
                processed_list.append(None)
        return processed_list

    def adjust_weights_dynamically(self, random_contrast_loss, plan_similarity_loss):
        plan_normalized = max(0.0, min(1.0, plan_similarity_loss / 4))
        self.plan_similarity_weight = 0.01 + plan_normalized * 0.04

        contrast_normalized = max(0.0, min(1.0, random_contrast_loss / 0.69))
        self.random_contrast_weight = 0.01 + contrast_normalized * 0.49

    def insert_plan_tokens(self, input_ids, attention_mask, labels, human_end_positions, plans):
        batch_size = input_ids.shape[0]
        new_input_ids = []
        new_attention_mask = []
        new_labels = []

        if isinstance(human_end_positions, torch.Tensor):
            hep_list = human_end_positions.detach().to('cpu', non_blocking=True).tolist()
        else:
            hep_list = human_end_positions

        bop_token_id = self.tokenizer.convert_tokens_to_ids('<bop>')
        eop_token_id = self.tokenizer.convert_tokens_to_ids('<eop>')

        for batch_idx in range(batch_size):
            insert_pos = int(hep_list[batch_idx])
            if insert_pos >= 0 and plans[batch_idx]:
                plan_token_ids = torch.tensor(plans[batch_idx], device=input_ids.device)

                before_ids = input_ids[batch_idx, :insert_pos]
                after_ids = input_ids[batch_idx, insert_pos:]

                bop_tensor = torch.tensor([bop_token_id], device=input_ids.device)
                eop_tensor = torch.tensor([eop_token_id], device=input_ids.device)
                marked_plan_ids = torch.cat([bop_tensor, plan_token_ids, eop_tensor], dim=0)

                new_seq = torch.cat([before_ids, marked_plan_ids, after_ids], dim=0)
                new_input_ids.append(new_seq)

                if attention_mask is not None:
                    before_mask = attention_mask[batch_idx, :insert_pos]
                    after_mask = attention_mask[batch_idx, insert_pos:]
                    plan_mask = torch.ones(len(marked_plan_ids), device=attention_mask.device,
                                           dtype=attention_mask.dtype)
                    new_attention_mask.append(torch.cat([before_mask, plan_mask, after_mask], dim=0))

                if labels is not None:
                    before_labels = labels[batch_idx, :insert_pos]
                    after_labels = labels[batch_idx, insert_pos:]
                    plan_labels = torch.full((len(marked_plan_ids),), IGNORE_TOKEN_ID, device=labels.device,
                                             dtype=labels.dtype)
                    new_labels.append(torch.cat([before_labels, plan_labels, after_labels], dim=0))
            else:
                new_input_ids.append(input_ids[batch_idx])
                if attention_mask is not None:
                    new_attention_mask.append(attention_mask[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])

        result = {
            'input_ids': pad_sequence(new_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            'attention_mask': pad_sequence(new_attention_mask, batch_first=True,
                                           padding_value=0) if new_attention_mask else None,
            'labels': pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID) if new_labels else None,
        }
        return result

    def cross_gpu_negatives(self, prepended_hidden_states):
        local_pairs = [
            (idx, h.detach().cpu())
            for idx, h in enumerate(prepended_hidden_states)
            if h is not None
        ]
        if not local_pairs:
            return [None] * len(prepended_hidden_states)

        if dist.is_available() and dist.is_initialized():
            local = [h for _, h in local_pairs]
            all_lists = [None] * dist.get_world_size()
            dist.all_gather_object(all_lists, local)
            pool = []
            for lst in all_lists:
                pool.extend(lst)
        else:
            pool = [h for _, h in local_pairs]

        negs = []
        for sample_idx, hidden_state in enumerate(prepended_hidden_states):
            if hidden_state is None:
                negs.append(None)
                continue

            candidates = pool
            if not (dist.is_available() and dist.is_initialized()) and len(local_pairs) > 1:
                candidates = [h for idx, h in local_pairs if idx != sample_idx]

            choice = torch.randint(0, len(candidates), (1,)).item()
            negs.append(candidates[choice].to(hidden_state.device))
        return negs

    def _forward_with_hidden_states_curriculum(
            self, input_ids, plan_ids, attention_mask, inputs_embeds, labels,
            human_end_positions, prepended_hidden_states,
            past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict,
            mode: str = "normal",
            **kwargs
    ):
        emb_weight = self.base_model.get_input_embeddings().weight
        device = emb_weight.device
        model_dtype = emb_weight.dtype

        disable_mix = bool(kwargs.get("disable_mix", False))

        if isinstance(human_end_positions, torch.Tensor):
            human_end_positions_list = human_end_positions.detach().to('cpu', non_blocking=True).tolist()
        else:
            human_end_positions_list = human_end_positions

        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids).to(model_dtype)

        plan_embeds_list = []
        for plan_item in plan_ids:
            plan_tensor = torch.tensor(plan_item, device=device)
            plan_embeds = self.base_model.get_input_embeddings()(plan_tensor).to(model_dtype)
            plan_embeds = plan_embeds.requires_grad_(True)
            plan_embeds_list.append(plan_embeds)

        inputs_embeds = inputs_embeds.to(dtype=model_dtype)

        task_embeds_list = []
        for i, task_item in enumerate(inputs_embeds):
            insert_pos = int(human_end_positions_list[i])
            before = inputs_embeds[i, :insert_pos]
            task_embeds_list.append(before)

        if prepended_hidden_states is not None:
            _aligned = []
            for h in prepended_hidden_states:
                if h is None:
                    _aligned.append(None)
                else:
                    _aligned.append(h.to(device=device, dtype=model_dtype, non_blocking=True))
            prepended_hidden_states = _aligned

            prepended_hidden_states = self.process_hidden_states_list(prepended_hidden_states)
            self._last_processed_hidden_states = prepended_hidden_states

        bop_token_id = self.tokenizer.convert_tokens_to_ids('<bop>')
        eop_token_id = self.tokenizer.convert_tokens_to_ids('<eop>')

        emb_layer = self.base_model.get_input_embeddings()
        emb_weight = emb_layer.weight

        if bop_token_id is None or eop_token_id is None:
            raise RuntimeError(
                "Special tokens <bop>/<eop> not found in tokenizer. "
                "Make sure you added them and resized embeddings before building datasets."
            )

        vocab_size = emb_weight.size(0)
        if not (0 <= bop_token_id < vocab_size) or not (0 <= eop_token_id < vocab_size):
            raise RuntimeError(
                f"Special token id out of range: bop={bop_token_id}, eop={eop_token_id}, vocab={vocab_size}. "
                "Did you call model.resize_token_embeddings(len(tokenizer)) after adding special tokens?"
            )

        bop_vec = emb_weight[bop_token_id].to(dtype=model_dtype)
        eop_vec = emb_weight[eop_token_id].to(dtype=model_dtype)

        batch_size = inputs_embeds.shape[0]
        new_inputs_embeds = []
        new_attention_mask = [] if attention_mask is not None else None
        new_labels = [] if labels is not None else None

        for batch_idx in range(batch_size):
            insert_pos = int(human_end_positions_list[batch_idx])
            if insert_pos < 0:
                new_inputs_embeds.append(inputs_embeds[batch_idx])
                if attention_mask is not None:
                    new_attention_mask.append(attention_mask[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            before = inputs_embeds[batch_idx, :insert_pos]
            after = inputs_embeds[batch_idx, insert_pos:]

            if prepended_hidden_states is not None and batch_idx < len(prepended_hidden_states):
                hidden_state_to_use = prepended_hidden_states[batch_idx]
            else:
                hidden_state_to_use = None

            plan_embeds = plan_embeds_list[batch_idx]

            if disable_mix or hidden_state_to_use is None:
                mixed_hidden_state = hidden_state_to_use
            else:
                random_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                if len(self.ratio_list) <= batch_idx:
                    mix_ratio = random.choice(random_list)
                    self.ratio_list.append(mix_ratio)
                else:
                    mix_ratio = self.ratio_list[batch_idx]

                mixed_hidden_state = self.adaptive_mix_with_balanced_distribution(
                    hidden_state_to_use, plan_embeds, mix_ratio
                )

            if mixed_hidden_state is None:
                marked_hidden_state = torch.cat([
                    bop_vec.unsqueeze(0),
                    eop_vec.unsqueeze(0)
                ], dim=0).to(dtype=model_dtype)
            else:
                marked_hidden_state = torch.cat([
                    bop_vec.unsqueeze(0),
                    mixed_hidden_state,
                    eop_vec.unsqueeze(0)
                ], dim=0).to(dtype=model_dtype)

            batch_embeds = torch.cat([before, marked_hidden_state, after], dim=0)
            new_inputs_embeds.append(batch_embeds)

            if attention_mask is not None:
                before_mask = attention_mask[batch_idx, :insert_pos]
                after_mask = attention_mask[batch_idx, insert_pos:]
                prepended_mask = torch.ones(
                    marked_hidden_state.size(0),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                batch_mask = torch.cat([before_mask, prepended_mask, after_mask], dim=0)
                new_attention_mask.append(batch_mask)

            if labels is not None:
                before_labels = labels[batch_idx, :insert_pos]
                after_labels = labels[batch_idx, insert_pos:]
                prepended_labels = torch.full(
                    (marked_hidden_state.size(0),),
                    IGNORE_TOKEN_ID,
                    dtype=labels.dtype,
                    device=labels.device
                )
                batch_labels = torch.cat([before_labels, prepended_labels, after_labels], dim=0)
                new_labels.append(batch_labels)

        inputs_embeds = torch.nn.utils.rnn.pad_sequence(new_inputs_embeds, batch_first=True, padding_value=0)
        if new_attention_mask:
            attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
        if new_labels:
            labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.float32,
                                               non_blocking=True).contiguous()

        outputs = self.base_model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        return outputs, attention_mask, labels

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            plans=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            human_end_positions=None,
            prepended_hidden_states=None,
            fuse_delta=True,
            **kwargs
    ):
        emb_weight = self.base_model.get_input_embeddings().weight
        device = emb_weight.device
        model_dtype = emb_weight.dtype

        self._current_batch_hidden_states = prepended_hidden_states

        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        if labels is not None:
            labels = labels.to(device)

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len, hidden_size = inputs_embeds.shape
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        hidden_state_seq_len = []
        plan_seq_len = []
        for item in prepended_hidden_states:
            hidden_state_seq_len.append(item.shape[0])

        prepende_hidden_states_store = prepended_hidden_states

        for item in plans:
            plan_seq_len.append(len(item))

        # Plan text insertion
        plan_outputs = None
        attention_mask_p = None
        labels_p = None
        if plans is not None:
            with torch.no_grad():
                plan_data = self.insert_plan_tokens(
                    input_ids, attention_mask, labels, human_end_positions, plans
                )
                plan_outputs = self.base_model(
                    input_ids=plan_data['input_ids'],
                    attention_mask=plan_data['attention_mask'],
                    labels=plan_data['labels'],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs
                )

                attention_mask_p = plan_data['attention_mask']
                labels_p = plan_data['labels']

        random_outputs = None
        negs = self.cross_gpu_negatives(prepended_hidden_states)
        random_hidden_states = negs

        for i in range(len(random_hidden_states)):
            if random_hidden_states[i] is None or prepended_hidden_states[i] is None:
                continue
            hidden_state_len = prepended_hidden_states[i].size(0)
            if random_hidden_states[i].size(0) >= hidden_state_len:
                random_hidden_states[i] = random_hidden_states[i][:hidden_state_len]
            else:
                repeat_times = (hidden_state_len // random_hidden_states[i].shape[0]) + 1
                repeated = random_hidden_states[i].repeat(repeat_times, 1)[:hidden_state_len]
                noise = torch.randn_like(repeated) * 0.01
                random_hidden_states[i] = repeated + noise

        with torch.no_grad():
            random_outputs, attention_mask_r, labels_r = self._forward_with_hidden_states_curriculum(
                input_ids, plans, attention_mask, inputs_embeds, labels,
                human_end_positions, random_hidden_states,
                past_key_values, use_cache, output_attentions,
                output_hidden_states, return_dict,
                mode="random",
                **kwargs
            )

        self.ratio_list = []
        normal_outputs, attention_mask_n, labels_n = self._forward_with_hidden_states_curriculum(
            input_ids, plans, attention_mask, inputs_embeds, labels,
            human_end_positions, prepended_hidden_states,
            past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict,
            mode="normal",
            **kwargs
        )

        # Calculate KL/JSD loss
        if plan_outputs is not None and human_end_positions is not None:
            ce_loss_only = normal_outputs.loss.detach()

            plan_similarity_loss = self.calculate_plan_similarity_loss(
                normal_outputs.logits, plan_outputs.logits,
                attention_mask_n, attention_mask_p, labels_n, labels_p
            )

            random_contrast_loss = self.calculate_random_contrast_loss(
                normal_outputs.logits, random_outputs.logits,
                attention_mask_n, attention_mask_r, labels_n, labels_r
            )

            self.adjust_weights_dynamically(random_contrast_loss, plan_similarity_loss)

            total_loss = normal_outputs.loss \
                         + self.plan_similarity_weight * plan_similarity_loss \
                         + self.random_contrast_weight * random_contrast_loss

            self._current_loss = total_loss

            if not self.training:
                self.last_loss_components = {
                    "eval_ce_loss": float(ce_loss_only.item()),
                    "eval_plan_similarity": float(plan_similarity_loss.detach().item()),
                    "eval_random_contrast": float(random_contrast_loss.detach().item()),
                    "eval_total_loss": float(total_loss.detach().item()),
                    "plan_w": float(self.plan_similarity_weight),
                    "random_w": float(self.random_contrast_weight),
                }

            normal_outputs.loss = total_loss

        return normal_outputs

    def _first_supervised_pos(self, labels, attn):
        mask = (
                (labels != IGNORE) &
                (labels != 151644) &
                (labels != 151645) &
                (labels != 198) &
                attn.bool()
        )
        pos = mask.nonzero(as_tuple=False)
        return pos[0, 0].item() if len(pos) else None

    def _avg_ce_masked(self, logits, labels, attn_mask):
        if logits is None:
            return None
        mask = (labels.ne(IGNORE) & attn_mask.bool())
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        logit_sel = logits[mask]
        y_sel = labels[mask].long()
        return F.cross_entropy(logit_sel, y_sel, reduction="mean")

    def _js_bits_masked(self, logits_p, logits_q, labels, attn_mask):
        mask = (labels.ne(IGNORE) & attn_mask.bool())
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits_p.device)
        p = F.softmax(logits_p[mask], dim=-1).clamp_min(EPS)
        q = F.softmax(logits_q[mask], dim=-1).clamp_min(EPS)
        m = 0.5 * (p + q)
        js_nats = 0.5 * F.kl_div(p.log(), m, reduction="batchmean") + \
                  0.5 * F.kl_div(q.log(), m, reduction="batchmean")
        return js_nats / math.log(2)

    def calculate_plan_similarity_loss(
            self,
            normal_logits, plan_logits,
            attention_mask_n, attention_mask_p,
            labels_n, labels_p,
            margin_kl=0.7, margin_cos=0.3,
    ):
        losses = []
        B = normal_logits.size(0)

        for i in range(B):
            s_n = self._first_supervised_pos(labels_n[i], attention_mask_n[i])
            s_p = self._first_supervised_pos(labels_p[i], attention_mask_p[i])
            if s_n is None or s_p is None:
                continue

            max_len = min(normal_logits.size(1) - s_n,
                          plan_logits.size(1) - s_p)
            n_slice = normal_logits[i, s_n: s_n + max_len]
            p_slice = plan_logits[i, s_p: s_p + max_len].detach()

            joint = (attention_mask_n[i, s_n: s_n + max_len].bool() &
                     attention_mask_p[i, s_p: s_p + max_len].bool() &
                     labels_n[i, s_n: s_n + max_len].ne(IGNORE) &
                     labels_p[i, s_p: s_p + max_len].ne(IGNORE))
            if not joint.any():
                continue

            n_logits = n_slice[joint]
            p_logits = p_slice[joint]

            kl = F.kl_div(F.log_softmax(n_logits, dim=-1),
                          F.softmax(p_logits, dim=-1).clamp_min(EPS),
                          reduction="batchmean")
            n_prob = F.softmax(n_logits, dim=-1).clamp_min(EPS).view(-1)
            p_prob = F.softmax(p_logits, dim=-1).clamp_min(EPS).view(-1)
            cos_loss = 1.0 - F.cosine_similarity(n_prob, p_prob, dim=0)

            cur_loss = margin_kl * kl + margin_cos * cos_loss
            losses.append(cur_loss)

        final = torch.stack(losses).mean() if losses else \
            torch.tensor(0.0, device=normal_logits.device, requires_grad=True)

        return final

    def calculate_random_contrast_loss(
            self,
            normal_logits, random_logits,
            attention_mask_n, attention_mask_r,
            labels_n, labels_r,
            margin=0.69,
    ):
        def js(p, q):
            m = 0.5 * (p + q)
            return 0.5 * F.kl_div(p.log(), m, reduction='batchmean') + \
                0.5 * F.kl_div(q.log(), m, reduction='batchmean')

        losses = []
        B = normal_logits.size(0)

        for i in range(B):
            s = self._first_supervised_pos(labels_n[i], attention_mask_n[i])
            if s is None:
                continue

            max_len = min(normal_logits.size(1) - s,
                          random_logits.size(1) - s)
            n_slice = normal_logits[i, s: s + max_len]
            r_slice = random_logits[i, s: s + max_len].detach()

            joint = (attention_mask_n[i, s: s + max_len].bool() &
                     attention_mask_r[i, s: s + max_len].bool() &
                     labels_n[i, s: s + max_len].ne(IGNORE) &
                     labels_r[i, s: s + max_len].ne(IGNORE))
            if not joint.any():
                continue

            n_probs = F.softmax(n_slice[joint], dim=-1).clamp_min(EPS)
            r_probs = F.softmax(r_slice[joint], dim=-1).clamp_min(EPS)

            loss = torch.clamp(margin - js(n_probs, r_probs), min=0.0)
            losses.append(loss)

        final = torch.stack(losses).mean() if losses else \
            torch.tensor(0.0, device=normal_logits.device, requires_grad=True)

        return final

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.base_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens, **kwargs):
        kwargs.setdefault("mean_resizing", False)
        return self.base_model.resize_token_embeddings(new_num_tokens, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        import os, json
        import numpy as np
        os.makedirs(save_directory, exist_ok=True)

        self.base_model.save_pretrained(save_directory, **kwargs)

        if getattr(self, "prepended_learnable", False):
            prepended_state_path = os.path.join(save_directory, "default_prepended_hidden_state.pt")
            torch.save(self.default_prepended_hidden_state, prepended_state_path)

        mha_state_path = os.path.join(save_directory, "hidden_mha_state.pt")
        mha_state = {
            "hidden_mha": self.hidden_mha.state_dict(),
            "pre_ln": self.pre_ln.state_dict(),
            "post_ln": self.post_ln.state_dict(),
            "adaptive_proj": self.adaptive_proj.state_dict(),
            "scale": float(self.scale.detach().cpu().item()) if hasattr(self.scale, "item") else float(self.scale),
            "output_scale": float(self.output_scale.detach().cpu().item()) if hasattr(self.output_scale, "item") else float(self.output_scale),
        }
        torch.save(mha_state, mha_state_path)

        def _to_py(v):
            try:
                if isinstance(v, (bool, int, float, str)) or v is None:
                    return v
                if hasattr(v, "item"):
                    return v.item()
                if isinstance(v, (np.integer,)):
                    return int(v)
                if isinstance(v, (np.floating,)):
                    return float(v)
                if isinstance(v, (np.bool_,)):
                    return bool(v)
            except Exception:
                pass
            return str(v)

        if hasattr(self, "default_prepended_hidden_state"):
            hidden_size_val = int(self.default_prepended_hidden_state.shape[-1])
        else:
            hidden_size_val = int(getattr(self.base_model.config, "hidden_size", getattr(self, "hidden_size", 0)))

        prepended_config = {
            "prepended_length": _to_py(getattr(self, "prepended_length", 0)),
            "prepended_learnable": _to_py(getattr(self, "prepended_learnable", False)),
            "hidden_size": _to_py(hidden_size_val),
            "mha_num_heads": _to_py(getattr(self.hidden_mha, "num_heads", 0)),
            "plan_similarity_weight": _to_py(getattr(self, "plan_similarity_weight", 0.0)),
            "random_contrast_weight": _to_py(getattr(self, "random_contrast_weight", 0.0)),
        }

        config_path = os.path.join(save_directory, "prepended_config.json")
        with open(config_path, "w") as f:
            json.dump(prepended_config, f, indent=2)
