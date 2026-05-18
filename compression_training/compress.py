#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 Interlat Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Interlat Compression Training - Open Source Version

This module implements teacher-student distillation for compressed latent communication.
Key features:
- Fully open source (no proprietary dependencies)
- Uses HuggingFace models and datasets
- Teacher-student distillation framework
- Uncertainty-weighted KL divergence loss
- Supports heterogeneous model architectures

Usage:
    python compression_training/compress.py --student_model_path meta-llama/Llama-3.1-8B-Instruct \
                                          --teacher_model_path ./trained_models/teacher_model \
                                          --data_path ./data/alfworld_sft.json \
                                          --hf_hidden_repo your_hidden_states_dataset
"""

from __future__ import annotations

import os
import re
import gc
import json
import time
import random
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    EarlyStoppingCallback,
)
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.model.model_adapter import get_model_adapter
from fastchat.conversation import SeparatorStyle

# Import ModelWithInsertedHiddenState - adjust path based on your project structure
try:
    from core_training.hidden_model.custom_model import ModelWithInsertedHiddenState
except ImportError:
    try:
        from ..core_training.hidden_model.custom_model import ModelWithInsertedHiddenState
    except ImportError:
        # Fallback for different project structures
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_training', 'hidden_model'))
        from custom_model import ModelWithInsertedHiddenState

from callbacks import (
    PreCreateCkptDirCallback,
    ParamChangeTrackerCallback,
    EarlyStoppingStatusCallback,
    PrintMetricsCallback,
)


# =========================
# Constants
# =========================
IGNORE = -100
EPS = 1e-8
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# =========================
# Small utilities
# =========================
def is_rank0() -> bool:
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def rank0_print(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)


# =========================
# Modules
# =========================
class AdaptiveProjection(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.proj[0].weight, mean=0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        residual = x * self.scale
        x = self.proj(residual)
        out = (residual + x) * self.output_scale
        return out.squeeze(0) if out.size(0) == 1 else out


class HiddenStateHead(nn.Module):
    """
    Hidden state processing head (LayerNorm + MHA + projection).
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.hidden_mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaptive_proj = AdaptiveProjection(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single = True

        dev = self.pre_ln.weight.device
        dtyp = self.pre_ln.weight.dtype
        x = x.to(dev, dtyp).contiguous()

        normed = self.pre_ln(x).contiguous()
        attn_out, _ = self.hidden_mha(normed, normed, normed, need_weights=False)
        out = self.post_ln(normed + attn_out)
        out = self.adaptive_proj(out)
        return out.squeeze(0) if single else out


def load_hidden_head_from_ckpt(head: HiddenStateHead, ckpt_path: str, to_dtype: torch.dtype):
    state = torch.load(ckpt_path, map_location="cpu")

    def cast_sd(sd):
        return {k: (v.to(to_dtype) if hasattr(v, "dtype") else v) for k, v in sd.items()}

    head.load_state_dict(
        {
            **head.state_dict(),
            **{f"pre_ln.{k}": v for k, v in cast_sd(state["pre_ln"]).items()},
            **{f"post_ln.{k}": v for k, v in cast_sd(state["post_ln"]).items()},
        },
        strict=False,
    )

    head.hidden_mha.load_state_dict(cast_sd(state["hidden_mha"]), strict=True)

    if "adaptive_proj" in state:
        head.adaptive_proj.load_state_dict(cast_sd(state["adaptive_proj"]), strict=False)
    else:
        with torch.no_grad():
            if "scale" in state:
                head.adaptive_proj.scale.fill_(float(state["scale"]))
            if "output_scale" in state:
                head.adaptive_proj.output_scale.fill_(float(state["output_scale"]))


# =========================
# Core model: student training + fully frozen teacher
# =========================
class StudentOverTeacher(nn.Module):
    def __init__(self, teacher, tokenizer, student_lm, K=128, beta=2.0, lambda_pref=2.0):
        super().__init__()
        self.teacher = teacher
        self.student_lm = student_lm
        self.tokenizer = tokenizer
        self.K = K

        # Loss weights (kept consistent with your current code usage)
        self.lambda_ce = 1.0
        self.lambda_kl = 0.1
        self.lambda_align = 0.1

        # Distillation temperature & uncertainty settings
        self.T = 4.0
        self.window_len = 64
        self.unc_quantile = 0.95

        # Freeze teacher parameters
        for _, p in self.teacher.named_parameters():
            p.requires_grad = False
        self.teacher.eval()
        rank0_print("[teacher] all parameters frozen")

        n_teacher_train = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        rank0_print(f"[check] teacher trainable params: {n_teacher_train}")
        assert n_teacher_train == 0, "Teacher should be fully frozen."

        Hs = student_lm.config.hidden_size

        # Latent autoregressive bridge
        self.h2e = nn.Sequential(
            nn.LayerNorm(Hs),
            nn.Linear(Hs, Hs, bias=False),
        )
        self.lat_bos = nn.Parameter(torch.zeros(Hs))
        nn.init.normal_(self.lat_bos, mean=0.0, std=0.02)

        # Special tokens
        self.bop_id = tokenizer.convert_tokens_to_ids("<bop>")
        self.eop_id = tokenizer.convert_tokens_to_ids("<eop>")
        assert self.bop_id is not None and self.eop_id is not None, \
            "Missing <bop>/<eop>. Please add_special_tokens then resize embeddings."

        # For callbacks
        self.last_metrics: Dict[str, float] = {}

    # HF Trainer compatibility helpers
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.student_lm, "gradient_checkpointing_enable"):
            self.student_lm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.student_lm, "gradient_checkpointing_disable"):
            self.student_lm.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if hasattr(self.student_lm, "enable_input_require_grads"):
            self.student_lm.enable_input_require_grads()

    # Teacher-side head processing (keeps gradient flow from student path)
    def process_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        normed = self.teacher.pre_ln(hidden_state)
        attn_output, _ = self.teacher.hidden_mha(normed, normed, normed)
        attn_output = self.teacher.post_ln(normed + attn_output)
        hidden_state = self.teacher.adaptive_proj(attn_output)
        return hidden_state

    # Insert latent into context
    def _assemble_with_latent(self, inputs_embeds, attention_mask, labels, human_end_positions, latent_list):
        emb_weight = self.teacher.get_input_embeddings().weight
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        bop_vec = emb_weight[self.bop_id].to(device=device, dtype=dtype)
        eop_vec = emb_weight[self.eop_id].to(device=device, dtype=dtype)

        B, T, H = inputs_embeds.shape
        new_embeds, new_masks, new_labels = [], [], []
        for b in range(B):
            pos = int(human_end_positions[b].item())
            pos = max(1, min(pos, T))
            before = inputs_embeds[b, :pos]
            after = inputs_embeds[b, pos:]
            z = latent_list[b].to(device=device, dtype=dtype)

            marked = torch.cat([bop_vec.unsqueeze(0), z, eop_vec.unsqueeze(0)], dim=0)
            emb_b = torch.cat([before, marked, after], dim=0)

            m_bef = attention_mask[b, :pos]
            m_aft = attention_mask[b, pos:]
            m_mid = torch.ones(marked.size(0), device=device, dtype=attention_mask.dtype)
            mask_b = torch.cat([m_bef, m_mid, m_aft], dim=0)

            y_bef = labels[b, :pos]
            y_aft = labels[b, pos:]
            y_mid = torch.full((marked.size(0),), IGNORE, device=device, dtype=labels.dtype)
            y_b = torch.cat([y_bef, y_mid, y_aft], dim=0)

            new_embeds.append(emb_b)
            new_masks.append(mask_b)
            new_labels.append(y_b)

        inputs_embeds = pad_sequence(new_embeds, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(new_masks, batch_first=True, padding_value=0)
        labels = pad_sequence(new_labels, batch_first=True, padding_value=IGNORE)
        return inputs_embeds, attention_mask, labels

    # Window masks
    def _build_window_masks(self, labels_A: torch.Tensor, labels_B: torch.Tensor, human_end_positions: torch.Tensor, window_len: int):
        sup_mask_AD = (labels_A != IGNORE)
        B, LA = sup_mask_AD.shape
        LB = labels_B.shape[1]

        orig_idx_AD = torch.cumsum(sup_mask_AD.int(), dim=1) - 1
        win_AD = torch.zeros_like(sup_mask_AD, dtype=torch.bool)

        win_B = torch.zeros((B, LB), dtype=torch.bool, device=labels_B.device)

        for b in range(B):
            s = int(human_end_positions[b].item()) + 1
            if s < 0:
                s = 0
            eB = min(s + window_len, LB)
            if s < eB:
                win_B[b, s:eB] = True

            cond = (orig_idx_AD[b] >= s) & (orig_idx_AD[b] < s + window_len) & sup_mask_AD[b]
            win_AD[b, cond] = True

        return win_AD, win_B

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        human_end_positions: torch.Tensor,
        prepended_hidden_states: List[Optional[torch.Tensor]] = None,
        **kwargs,
    ):
        device = input_ids.device
        B, T = input_ids.shape
        z_theta_list: List[torch.Tensor] = []
        emb_s = self.student_lm.get_input_embeddings()

        # Student path: generate latent representations
        for b in range(B):
            pos = int(human_end_positions[b].item())
            pos = max(1, min(pos, T))

            require_text = "Please generate a plan to solve this task: "
            require_ids = self.tokenizer(
                require_text,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids.to(input_ids.device)

            ctx_ids = torch.cat([require_ids, input_ids[b:b + 1, :pos]], dim=1)
            ctx_attn = torch.cat([torch.ones_like(require_ids), attention_mask[b:b + 1, :pos]], dim=1)

            ctx_out = self.student_lm(
                input_ids=ctx_ids,
                attention_mask=ctx_attn,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_kv = ctx_out.past_key_values
            past_len = ctx_ids.size(1)

            step_in = self.lat_bos.to(ctx_ids.device, dtype=emb_s.weight.dtype).view(1, 1, -1)
            zs = []
            for _ in range(self.K):
                position_ids = torch.tensor([[past_len]], device=ctx_ids.device, dtype=torch.long)
                out_k = self.student_lm(
                    inputs_embeds=step_in,
                    attention_mask=torch.ones_like(position_ids),
                    past_key_values=past_kv,
                    position_ids=position_ids,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                z_k = out_k.hidden_states[-1][:, -1, :]
                zs.append(z_k.squeeze(0))
                past_kv = out_k.past_key_values
                past_len += 1
                step_in = self.h2e(z_k).view(1, 1, -1)

            z_b = torch.stack(zs, dim=0)          # [K, Hs]
            z_b = self.process_hidden_state(z_b)  # [K, Ht] (keeps grad)
            z_theta_list.append(z_b)

        # Teacher three paths: A=student latent, D=data latent (no grad), B=baseline
        emb_layer = self.teacher.get_input_embeddings()
        inputs_embeds_full = emb_layer(input_ids)

        # A: student latent (grad)
        emb_A, mask_A, label_A = self._assemble_with_latent(
            inputs_embeds_full, attention_mask, labels, human_end_positions, z_theta_list
        )
        out_A = self.teacher(
            input_ids=None,
            inputs_embeds=emb_A,
            attention_mask=mask_A,
            labels=label_A,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        ce_theta = out_A.loss

        # D: data latent (no grad)
        dtype = inputs_embeds_full.dtype
        z_data_list: List[torch.Tensor] = []
        for b in range(B):
            z = prepended_hidden_states[b]
            assert z is not None, "Sample missing hidden_state"
            if isinstance(z, torch.Tensor) and z.dim() == 3:
                z = z[0]
            z = z.to(device=device, dtype=dtype)
            if z.size(0) >= self.K:
                z = z[: self.K]
            else:
                z = z.repeat((self.K + z.size(0) - 1) // z.size(0), 1)[: self.K]
            with torch.no_grad():
                z = self.process_hidden_state(z)
            z_data_list.append(z)

        with torch.no_grad():
            emb_D, mask_D, label_D = self._assemble_with_latent(
                inputs_embeds_full, attention_mask, labels, human_end_positions, z_data_list
            )
            out_D = self.teacher(
                input_ids=None,
                inputs_embeds=emb_D,
                attention_mask=mask_D,
                labels=label_D,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )

        # B: baseline (no latent)
        with torch.no_grad():
            out_B = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )

        # ===== Uncertainty-weighted KL =====
        win_AD, win_B = self._build_window_masks(label_A, labels, human_end_positions, self.window_len)

        logits_B = out_B.logits.detach()
        logits_D = out_D.logits.detach()

        pB = F.softmax(logits_B, dim=-1)
        pD = F.softmax(logits_D, dim=-1)

        ent_B = -(pB * (pB.clamp_min(1e-6)).log()).sum(-1)
        ent_D = -(pD * (pD.clamp_min(1e-6)).log()).sum(-1)

        sup_AD = (label_A != IGNORE) & win_AD
        sup_B = (labels != IGNORE) & win_B

        ent_B_flat = ent_B[sup_B]
        ent_D_flat = ent_D[sup_AD]

        if ent_B_flat.numel() == 0 or ent_D_flat.numel() == 0 or ent_B_flat.numel() != ent_D_flat.numel():
            sup_AD = (label_A != IGNORE)
            sup_B = (labels != IGNORE)
            ent_B_flat = ent_B[sup_B]
            ent_D_flat = ent_D[sup_AD]

        dU = ent_B_flat - ent_D_flat
        w_raw = torch.relu(dU).float()
        if (w_raw > 0).any():
            q_hi = torch.quantile(w_raw[w_raw > 0], self.unc_quantile)
            w_raw = torch.clamp(w_raw, 0, q_hi)
        den = w_raw.mean().clamp_min(1e-6)
        W_unc_flat = (w_raw / den).detach()

        Ttemp = self.T
        logits_A = out_A.logits
        log_p_t = F.log_softmax(logits_D / Ttemp, dim=-1)
        log_p_s = F.log_softmax(logits_A / Ttemp, dim=-1)
        p_t = log_p_t.exp()

        log_p_t_tok = log_p_t[sup_AD]
        log_p_s_tok = log_p_s[sup_AD]
        p_t_tok = p_t[sup_AD]

        if p_t_tok.numel() == 0:
            L_kl_unc = logits_A.new_tensor(0.0)
        else:
            kl_tok = (Ttemp * Ttemp) * (p_t_tok * (log_p_t_tok - log_p_s_tok)).sum(dim=-1)
            L_kl_unc = (W_unc_flat * kl_tok).sum() / (W_unc_flat.sum() + 1e-6)

        # ===== Latent direction alignment (cosine) =====
        def _pool_mean(z_list: List[torch.Tensor]) -> torch.Tensor:
            return torch.stack([z.mean(dim=0) for z in z_list], dim=0)

        Z_theta = _pool_mean(z_theta_list)
        with torch.no_grad():
            Z_data = _pool_mean(z_data_list)

        L_align = 1.0 - F.cosine_similarity(Z_theta, Z_data, dim=-1).mean()

        loss = (
            self.lambda_ce * ce_theta
            + self.lambda_kl * L_kl_unc
            + self.lambda_align * L_align
        )

        # Write metrics for PrintMetricsCallback (no per-step print spam)
        self.last_metrics = {
            "ce_theta": float(ce_theta.detach().item()),
            "kl": float(L_kl_unc.detach().item()),
            "align": float(L_align.detach().item()),
            "total": float(loss.detach().item()),
        }

        out = out_A
        out.loss = loss
        return out


# =========================
# Data loading
# =========================
class HiddenStateLoader:
    """
    Load hidden_state/plan data from HF datasets.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self._load_data()

    def _load_data(self):
        rank0_print(f"Loading tensor data from {self.dataset_name}")
        ds = datasets.load_dataset(self.dataset_name, split=datasets.Split.TRAIN)
        rank0_print(f"Loaded {len(ds)} records.")

        with tqdm(total=1, desc="Converting Dataset to Pandas") as pbar:
            df = ds.to_pandas()
            pbar.update(1)

        id_to_data: Dict[str, Dict[str, Any]] = {}
        t0 = time.time()
        for _, row in df.iterrows():
            task_id = row.get("task_id") or row.get("id")
            hs = row.get("hidden_state", None)
            plan = row.get("plan", "")

            if task_id is None or hs is None:
                continue

            if isinstance(hs, np.ndarray) and hs.dtype == object:
                hs = np.array(hs.tolist(), dtype=np.float32)
            else:
                hs = np.array(hs, dtype=np.float32)

            tensor = torch.from_numpy(hs)
            id_to_data[str(task_id)] = {
                "hidden_state": tensor,
                "plan": plan if isinstance(plan, str) else str(plan),
            }

        rank0_print(f"Built id_to_data for {len(id_to_data)} items in {time.time() - t0:.2f}s")
        self.id_to_data = id_to_data

    def get_hidden_state_and_plan(self, task_id: str):
        task_id = str(task_id)
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        d = self.id_to_data[task_id]
        return d["hidden_state"], d["plan"]


def preprocess_with_position_tracking(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    conv_template_model_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess conversations and track the position of the first human message end marker.
    """

    conv = get_model_adapter(conv_template_model_path).get_default_conv_template(conv_template_model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    human_end_markers = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        first_human_content = None

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"

            if sentence["from"] == "human" and first_human_content is None:
                first_human_content = sentence["value"]
                marked_content = sentence["value"] + "<FIRST_HUMAN_END>"
                conv.append_message(role, marked_content)
            else:
                conv.append_message(role, sentence["value"])

        conversations.append(conv.get_prompt())
        human_end_markers.append("<FIRST_HUMAN_END>" if first_human_content else None)

    tokenized = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids = tokenized.input_ids
    targets = input_ids.clone()

    # Locate <FIRST_HUMAN_END>
    human_end_positions = []
    marker_token_ids = tokenizer("<FIRST_HUMAN_END>", add_special_tokens=False).input_ids

    for batch_idx in range(input_ids.shape[0]):
        if human_end_markers[batch_idx]:
            position = -1
            for i in range(len(input_ids[batch_idx]) - len(marker_token_ids) + 1):
                if torch.all(
                    input_ids[batch_idx, i : i + len(marker_token_ids)]
                    == torch.tensor(marker_token_ids)
                ):
                    position = i
                    input_ids[batch_idx, i : i + len(marker_token_ids)] = tokenizer.pad_token_id
                    break
            human_end_positions.append(position)
        else:
            human_end_positions.append(-1)

    human_end_positions = torch.tensor(human_end_positions)

    # Conversation style handling
    if conv.sep_style == SeparatorStyle.CHATML:
        sep2 = "<|im_end|>\n"
        sep = conv.roles[1] + "\n"
        sep_len = len(tokenizer(sep, add_special_tokens=False).input_ids)
        sep2_len = len(tokenizer(sep2, add_special_tokens=False).input_ids)

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(sep2)
            cur_len = 1
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                if "<|im_start|>system\nYou are a helpful assistant." == turn:
                    sys_len = len(tokenizer("system\nYou are a helpful assistant.", add_special_tokens=False).input_ids)
                    target[cur_len : cur_len + sys_len] = IGNORE_TOKEN_ID
                    cur_len += sys_len
                elif i % 2 == 1:
                    instruction_len = len(tokenizer(turn, add_special_tokens=False).input_ids)
                    target[cur_len + 1 : cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += instruction_len
                else:
                    turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)
                    target[cur_len + 1 : cur_len + sep_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len
                cur_len += sep2_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    elif conv.sep_style == SeparatorStyle.LLAMA3:
        sep2 = "<|eot_id|>"
        sep2_len = len(tokenizer(sep2, add_special_tokens=False).input_ids)
        role_pattern = re.compile(r"<\|start_header_id\|>(.*?)<\|end_header_id\|>", re.DOTALL)

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            turns = conversation.split(sep2)
            cur_len = 0

            for turn in turns:
                if turn == "":
                    break

                turn_ids = tokenizer(turn, add_special_tokens=False).input_ids
                turn_len = len(turn_ids)

                m = role_pattern.search(turn)
                role = m.group(1).strip() if m else None

                if role is None:
                    target[cur_len : cur_len + turn_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len
                elif role in ("system", "user"):
                    target[cur_len : cur_len + turn_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len
                elif role == "assistant":
                    header_text = m.group(0)
                    header_len = len(tokenizer(header_text, add_special_tokens=False).input_ids)
                    target[cur_len : cur_len + header_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len
                else:
                    target[cur_len : cur_len + turn_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                if cur_len + sep2_len > tokenizer.model_max_length:
                    break

                if role in ("system", "user", None):
                    target[cur_len : cur_len + sep2_len] = IGNORE_TOKEN_ID
                else:
                    # assistant: keep eot token as label
                    pass
                cur_len += sep2_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length and cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch (llama3): {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        human_end_positions=human_end_positions,
    )


class SupervisedDataset(Dataset):
    """
    Preprocess conversations and attach plan/hidden_state.
    """

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizerBase, conv_template_model_path: str):
        super().__init__()
        sources = [ex["conversations"] for ex in raw_data]
        data_dict = preprocess_with_position_tracking(sources, tokenizer, conv_template_model_path)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.human_end_positions = data_dict["human_end_positions"]

        self.plans: List[List[int]] = []
        self.hidden_states: List[Optional[torch.Tensor]] = []
        self.tokenizer = tokenizer

        for ex in raw_data:
            plan = ex.get("plan", "")
            plan_ids = tokenizer(plan, add_special_tokens=False).input_ids
            self.plans.append(plan_ids)

            if "hidden_state" in ex and ex["hidden_state"] is not None:
                hs = ex["hidden_state"]
                if isinstance(hs, torch.Tensor):
                    self.hidden_states.append(hs)
                else:
                    self.hidden_states.append(torch.tensor(hs, dtype=torch.float32))
            else:
                self.hidden_states.append(None)

        assert len(self.input_ids) == len(self.plans) == len(self.hidden_states)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Any]:
        ret = dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            human_end_positions=self.human_end_positions[i],
            plan=self.plans[i],
        )
        if self.hidden_states[i] is not None:
            ret["prepended_hidden_states"] = self.hidden_states[i]
        return ret


class DataCollatorForSupervisedDataset:
    """
    Collator for supervised dataset.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids, labels = tuple([inst[k] for inst in instances] for k in ("input_ids", "labels"))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        human_end_positions = torch.tensor(
            [inst.get("human_end_positions", -1) for inst in instances],
            dtype=torch.long,
        )

        prepended_hidden_states = [inst.get("prepended_hidden_states", None) for inst in instances]
        plans = [inst.get("plan", []) for inst in instances]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "human_end_positions": human_end_positions,
            "plans": plans,
            "prepended_hidden_states": prepended_hidden_states,
        }


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizerBase,
    data_path: str,
    hf_hidden_repo: str,
    conv_template_model_path: str,
    eval_ratio: float = 0.05,
    prepended_length: int = 128,
) -> Dict[str, Any]:
    """
    Build data module.
    """
    loader_train = HiddenStateLoader(hf_hidden_repo)

    train_json = json.load(open(data_path, "r"))

    for item in train_json:
        merged_value = item["conversations"][0]["value"] + "\n" + item["conversations"][2]["value"]
        new_first = {"from": "human", "value": merged_value}
        new_first["value"] += "\nNow, you are given a step-by-step plan to complete this task as follow: "
        item["conversations"][0] = new_first

        hidden_state, plan = loader_train.get_hidden_state_and_plan(item["id"])
        L, H = hidden_state.shape
        if L >= prepended_length:
            hidden_state = hidden_state[:prepended_length, :]
        item["hidden_state"] = hidden_state
        item["plan"] = plan

        del item["conversations"][1:3]

    rng = random.Random(42)
    rng.shuffle(train_json)
    n_total = len(train_json)
    n_eval = max(1, int(n_total * eval_ratio))
    eval_json = train_json[:n_eval]
    train_json = train_json[n_eval:]

    train_dataset = SupervisedDataset(train_json, tokenizer=tokenizer, conv_template_model_path=conv_template_model_path)
    eval_dataset = SupervisedDataset(eval_json, tokenizer=tokenizer, conv_template_model_path=conv_template_model_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


# =========================
# Args
# =========================
@dataclass
class Args:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-7B")

    data_path: str = field(
        default="Interlat_preview/datasets/alfworld_sft.json"
    )
    hf_hidden_repo: str = field(default="your_dataset") # collected latent communications parquet file, path to the parquet file

    # Open source friendly model paths
    student_model_path: Optional[str] = field(default=None, metadata={"help": "Path to student model (local path or HuggingFace model name)"})
    teacher_model_path: Optional[str] = field(default=None, metadata={"help": "Path to teacher model (local path or HuggingFace model name)"})

    output_dir: str = field(default="./latent_out")
    model_max_length: int = field(default=4096)

    # Conv template model for fastchat adapter (avoid hard-coded local paths)
    conv_template_model_path: str = field(default="Qwen/Qwen2.5-7B-Instruct")

    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    learning_rate: float = field(default=5e-5)
    num_train_epochs: int = field(default=3)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    warmup_ratio: float = field(default=0.03)

    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    seed: int = field(default=42)

    K: int = field(default=128)
    beta: float = field(default=2.0)          # kept for compatibility
    lambda_pref: float = field(default=1.0)   # kept for compatibility

    deepspeed: Optional[str] = field(default=None)
    gradient_checkpointing: bool = field(default=False)

    early_stopping_patience: int = field(default=5)
    early_stopping_threshold: float = field(default=0.0)


def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    for k, v in Args().__dict__.items():
        t = type(v)
        if v is None:
            ap.add_argument(f"--{k}", type=str, default=None)
        elif t is bool:
            ap.add_argument(f"--{k}", action="store_true" if not v else "store_false")
        else:
            ap.add_argument(f"--{k}", type=t, default=v)
    ns = ap.parse_args()
    return Args(**vars(ns))


# =========================
# Main
# =========================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model path handling - Open source friendly approach
    # Use command line arguments or HuggingFace model names directly
    if args.student_model_path:
        student_model_path = args.student_model_path
        rank0_print(f"Using provided student model path: {student_model_path}")
    else:
        # Default to HuggingFace model that will be downloaded automatically
        student_model_path = "meta-llama/Llama-3.1-8B-Instruct"
        rank0_print(f"Using default student model: {student_model_path}")

    if args.teacher_model_path:
        teacher_model_path = args.teacher_model_path
        rank0_print(f"Using provided teacher model path: {teacher_model_path}")
    else:
        # Default teacher model - should be the output from core training step
        teacher_model_path = "./trained_models/teacher_model"
        rank0_print(f"Using default teacher model path: {teacher_model_path}")
        if not os.path.exists(teacher_model_path):
            rank0_print(f"[ERROR] Teacher model not found at {teacher_model_path}")
            rank0_print("Please provide --teacher_model_path or train a teacher model first using core_training/")
            raise FileNotFoundError(f"Teacher model not found: {teacher_model_path}")

    rank0_print("Models will be loaded from HuggingFace Hub or local paths automatically")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens_dict = {"additional_special_tokens": ["<FIRST_HUMAN_END>", "<bop>", "<eop>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.model_max_length = args.model_max_length

    # Teacher model
    config = AutoConfig.from_pretrained(teacher_model_path, trust_remote_code=True)
    config.use_cache = False

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32),
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    teacher.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_size = teacher.config.hidden_size
    prepended_length = 800

    teacher_model = ModelWithInsertedHiddenState(
        base_model=teacher,
        prepended_length=prepended_length,
        hidden_size=hidden_size,
        prepended_learnable=False,
    )
    teacher_model.tokenizer = tokenizer

    def count_parameters(module):
        return sum(p.numel() for p in module.parameters())

    def get_memory_size(module, dtype=torch.float32):
        param_count = count_parameters(module)
        bytes_per_param = 4 if dtype == torch.float32 else 2
        return param_count * bytes_per_param / (1024 * 1024)

    # Load MHA states if available
    mha_state_path = os.path.join(teacher_model_path, "hidden_mha_state.pt")
    if os.path.exists(mha_state_path):
        mha_state = torch.load(mha_state_path, map_location="cpu")

        teacher_model.hidden_mha.load_state_dict(mha_state["hidden_mha"])
        teacher_model.hidden_mha.to(device)

        teacher_model.pre_ln.load_state_dict(mha_state["pre_ln"])
        teacher_model.post_ln.load_state_dict(mha_state["post_ln"])
        teacher_model.pre_ln.to(device)
        teacher_model.post_ln.to(device)

        teacher_model.adaptive_proj.load_state_dict(mha_state["adaptive_proj"])
        teacher_model.adaptive_proj.to(device)

        teacher_model.scale.data.copy_(mha_state["scale"])
        teacher_model.output_scale.data.copy_(mha_state["output_scale"])
        teacher_model.scale.to(device)
        teacher_model.output_scale.to(device)

        mha_params = count_parameters(teacher_model.hidden_mha)
        mha_size = get_memory_size(teacher_model.hidden_mha, torch.bfloat16)
        rank0_print(f"MHA params: {mha_params:,} ({mha_size:.1f}MB)")

        proj_params = count_parameters(teacher_model.adaptive_proj)
        proj_size = get_memory_size(teacher_model.adaptive_proj, torch.bfloat16)
        rank0_print(f"Projection params: {proj_params:,} ({proj_size:.1f}MB)")

        del mha_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Student model
    config_s = AutoConfig.from_pretrained(student_model_path, trust_remote_code=True)
    config_s.use_cache = False

    student_lm = AutoModelForCausalLM.from_pretrained(
        student_model_path,
        config=config_s,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    student_lm.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Data module
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_path=args.data_path,
        hf_hidden_repo=args.hf_hidden_repo,
        conv_template_model_path=args.conv_template_model_path,
        eval_ratio=0.05,
        prepended_length=args.K,
    )

    # Show trainables
    def show_trainables(model: nn.Module, top_k: int = 20):
        total, trainable = 0, 0
        buckets: Dict[str, int] = {}
        for name, p in model.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
                prefix = name.split(".")[0]
                buckets[prefix] = buckets.get(prefix, 0) + n

        rank0_print(f"[trainables] {trainable:,} / {total:,} ({trainable / max(total,1):.2%}) params are trainable.")
        items = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
        rank0_print("[trainables] by top-level module:")
        for k, v in items[:top_k]:
            rank0_print(f"  - {k:20s}: {v:,}")

    # Build wrapper model
    model = StudentOverTeacher(
        teacher=teacher_model,
        tokenizer=tokenizer,
        student_lm=student_lm,
        K=args.K,
        beta=args.beta,
        lambda_pref=args.lambda_pref,
    )

    show_trainables(model)

    # Callbacks
    tracker_cb = ParamChangeTrackerCallback(model, track_patterns=["h2e"])
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )
    status_cb = EarlyStoppingStatusCallback(
        metric_for_best="eval_loss",
        greater_is_better=False,
        patience=args.early_stopping_patience,
        threshold=args.early_stopping_threshold,
        show_last=5,
    )

    # Optimizer sanity-check
    tmp_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-8)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    in_opt = sum(p.numel() for g in tmp_opt.param_groups for p in g["params"])
    rank0_print(f"[opt-check] trainable params = {trainable:,} | params handed to optimizer = {in_opt:,}")
    del tmp_opt

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to=[],
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        **data_module,
        callbacks=[
            PrintMetricsCallback(),
            early_stopping_callback,
            status_cb,
            tracker_cb,
            PreCreateCkptDirCallback(),
        ],
    )

    rank0_print("=== Training started ===")
    trainer.train()
    trainer.save_model(args.output_dir)
    rank0_print("=== Training finished ===")


if __name__ == "__main__":
    main()
