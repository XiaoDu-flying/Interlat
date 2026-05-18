import json
import copy
import re
import random
from typing import Dict, List, Sequence, Union, Any
import torch
from torch.utils.data import Dataset
import transformers
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from hidden_state_loader import HiddenStateLoader

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def debug_mask_for_conversation(
    conversation: str,
    target: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    ignore_token_id: int = -100,
    max_segments: int = 100,
    max_chars_per_segment: int = 8000,
):
    """
    Visualize which tokens in a sample are ignored (not contributing to loss)
    and which tokens are supervised.
    Tokens are grouped into continuous segments by IGNORE / LABEL status and printed segment by segment.

    Args:
      - conversation: original prompt string (output of conv.get_prompt())
      - target: corresponding label tensor (1D or 2D, length = tokenizer.model_max_length)
      - tokenizer: the same tokenizer
      - ignore_token_id: usually -100
    """

    # ---- 1. Flatten target to 1D ----
    if target.dim() == 2:
        target = target[0]
    target = target.detach().cpu()

    pad_id = tokenizer.pad_token_id

    # ---- 2. Re-tokenize conversation in the same way as training ----
    tokenized = tokenizer(
        conversation,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = tokenized.input_ids[0]  # [seq_len]
    input_ids = input_ids.detach().cpu()

    # ---- 3. Only inspect non-pad tokens ----
    total_len = int(input_ids.ne(pad_id).sum())
    print("=" * 120)
    print(f"[DEBUG] total non-pad length = {total_len}")
    print(f"[DEBUG] first 32 token_ids: {input_ids[:32].tolist()}")
    print(f"[DEBUG] first 32 target   : {target[:32].tolist()}")
    print("=" * 120)

    # ---- 4. Segment by IGNORE / LABEL ----
    segments = []  # list of (status, start_idx, end_idx)
    cur_status = None
    cur_start = 0

    for idx in range(total_len):
        tid = input_ids[idx].item()
        lab = target[idx].item()

        # Pad tokens are treated as IGNORE
        if tid == pad_id:
            status = "PAD"
        else:
            status = "IGN" if lab == ignore_token_id else "LAB"

        if cur_status is None:
            cur_status = status
            cur_start = idx
        elif status != cur_status:
            segments.append((cur_status, cur_start, idx))
            cur_status = status
            cur_start = idx

    # Append the last segment
    if cur_status is not None:
        segments.append((cur_status, cur_start, total_len))

    # ---- 5. Print text for each segment ----
    print("[DEBUG] Segments (grouped by IGN/LAB):")
    for seg_idx, (status, s, e) in enumerate(segments):
        if seg_idx >= max_segments:
            print(f"... (segments truncated at {max_segments})")
            break

        seg_ids = input_ids[s:e].tolist()
        # Do not skip special tokens so that <|eot_id|>, <|im_end|>, etc. are visible
        text = tokenizer.decode(seg_ids, skip_special_tokens=False)

        # Preview only, avoid overly long segments
        preview = text.replace("\n", "\\n")
        if len(preview) > max_chars_per_segment:
            preview = preview[:max_chars_per_segment] + "..."

        print(f"[{seg_idx:02d}] [{status}] tokens[{s}:{e}] (len={e-s}): {repr(preview)}")

    print("=" * 120)


def preprocess_with_position_tracking(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        model_path: str,
) -> Dict:
    """Preprocess data by tokenizing and tracking first human positions."""

    def decode_targets(
        targets: torch.Tensor,
        tokenizer,
        ignore_token_id: int = -100,
        pad_token_id: int = None,
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Decode a single sample's target (label tensor) into a string.
        ignore_token_id (default -100) and pad_token_id are skipped.
        """
        if targets.dim() == 2:
            targets = targets[0]
        targets = targets.detach().cpu().clone()

        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id

        valid_ids = [
            tid.item()
            for tid in targets
            if tid != ignore_token_id and tid != pad_token_id
        ]

        if not valid_ids:
            return ""

        decoded = tokenizer.decode(valid_ids, skip_special_tokens=skip_special_tokens)
        return decoded

    # ------- 1. Get base conversation template and inject system prompt -------

    base_conv = get_model_adapter(model_path).get_default_conv_template(model_path)

    # If the template has no system message, add one
    if getattr(base_conv, "system_message", None) in (None, ""):
        base_conv.set_system_message("You are a helpful assistant.")

    roles = {"human": base_conv.roles[0], "gpt": base_conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    human_end_markers: List[str] = []  # Whether each sample has <FIRST_HUMAN_END>

    # ------- 2. Build each conversation string and mark the first human message -------

    for i, source in enumerate(sources):
        # Use a fresh conv per sample to avoid message residue
        conv = copy.deepcopy(base_conv)

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        first_human_content = None

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"

            if sentence["from"] == "human" and first_human_content is None:
                first_human_content = sentence["value"]
                # Append marker at the end of the first human message
                marked_content = sentence["value"] + "<FIRST_HUMAN_END>"
                conv.append_message(role, marked_content)
            else:
                conv.append_message(role, sentence["value"])

        conversations.append(conv.get_prompt())
        human_end_markers.append("<FIRST_HUMAN_END>" if first_human_content else None)

    # ------- 3. Tokenize conversations -------

    tokenized = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids = tokenized.input_ids
    targets = input_ids.clone()

    # ------- 4. Locate <FIRST_HUMAN_END> positions in each sample -------

    human_end_positions = []
    marker_token_ids = tokenizer("<FIRST_HUMAN_END>", add_special_tokens=False).input_ids
    marker_token_ids_t = torch.tensor(marker_token_ids, dtype=input_ids.dtype)

    for batch_idx in range(input_ids.shape[0]):
        if human_end_markers[batch_idx]:
            position = -1
            row = input_ids[batch_idx]
            for i_pos in range(len(row) - len(marker_token_ids) + 1):
                if torch.all(row[i_pos:i_pos + len(marker_token_ids)] == marker_token_ids_t):
                    position = i_pos  # record marker start position
                    # Replace marker with pad tokens (model should not see it)
                    row[i_pos:i_pos + len(marker_token_ids)] = tokenizer.pad_token_id
                    break
            human_end_positions.append(position)
        else:
            human_end_positions.append(-1)

    human_end_positions = torch.tensor(human_end_positions, dtype=torch.long)

    # ------- 5. Compute label masks according to sep_style -------

    sep_style = base_conv.sep_style

    # --------- 5.1 Qwen2.5 / ChatML style ---------
    if sep_style == SeparatorStyle.CHATML:
        # Handle Qwen2.5 style
        sep2 = "<|im_end|>\n"
        sep = base_conv.roles[1] + "\n"
        sep_len = len(tokenizer(sep, add_special_tokens=False).input_ids)
        sep2_len = len(tokenizer(sep2, add_special_tokens=False).input_ids)

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(sep2)
            cur_len = 1  # skip BOS
            for i_turn, turn in enumerate(turns):
                if turn == "":
                    break
                if "<|im_start|>system\nYou are a helpful assistant." == turn:
                    sys_len = len(
                        tokenizer("system\nYou are a helpful assistant.", add_special_tokens=False).input_ids
                    )
                    target[cur_len: cur_len + sys_len] = IGNORE_TOKEN_ID
                    cur_len += sys_len
                elif i_turn % 2 == 1:
                    # human / instruction
                    instruction_len = len(tokenizer(turn, add_special_tokens=False).input_ids)
                    target[cur_len + 1: cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += instruction_len
                else:
                    # assistant
                    turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)
                    target[cur_len + 1: cur_len + sep_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                cur_len += sep2_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

    # --------- 5.2 LLaMA3 style ---------
    elif sep_style == SeparatorStyle.LLAMA3:  # for llama3
        sep2 = "<|eot_id|>"
        sep2_len = len(tokenizer(sep2, add_special_tokens=False).input_ids)

        # Match header: <|start_header_id|>role<|end_header_id|>
        role_pattern = re.compile(
            r"<\|start_header_id\|>(.*?)<\|end_header_id\|>",
            re.DOTALL
        )

        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            # Each turn = header + content, excluding <|eot_id|>
            turns = conversation.split(sep2)

            # The first token (<|begin_of_text|>) is usually ignored
            cur_len = 0

            for turn in turns:
                if turn == "":
                    break

                turn_ids = tokenizer(turn, add_special_tokens=False).input_ids
                turn_len = len(turn_ids)

                # Parse role
                m = role_pattern.search(turn)
                role = m.group(1).strip() if m else None  # "system"/"user"/"assistant"/None

                if role is None:
                    # Unknown segment, ignore entire turn
                    target[cur_len: cur_len + turn_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                elif role in ("system", "user"):
                    # system & user: ignore entire turn
                    target[cur_len: cur_len + turn_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                elif role == "assistant":
                    # assistant: ignore header, keep content as labels
                    header_text = m.group(0)
                    header_len = len(
                        tokenizer(header_text, add_special_tokens=False).input_ids
                    )
                    target[cur_len: cur_len + header_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                else:
                    # Other unexpected roles: ignore entire turn
                    target[cur_len: cur_len + turn_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                # Skip and handle <|eot_id|>
                if cur_len + sep2_len > tokenizer.model_max_length:
                    break
                if role in ("system", "user", None):
                    target[cur_len: cur_len + sep2_len] = IGNORE_TOKEN_ID
                elif role == "assistant":
                    pass
                cur_len += sep2_len

            # Ignore the rest
            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch (llama3): {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

    # Other sep_style branches can be added if needed

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        human_end_positions=human_end_positions,
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with position tracking."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path: str = None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs with position tracking...")
        sources = [example["conversations"] for example in raw_data]

        # Only pass conversations for preprocessing, without plan or extra data
        data_dict = preprocess_with_position_tracking(sources, tokenizer, model_path)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.human_end_positions = data_dict["human_end_positions"]
        self.tokenizer = tokenizer

        # Directly extract from raw_data to keep index alignment
        self.plans = []
        self.hidden_states = []

        for i, example in enumerate(raw_data):
            # Plan data
            plan = example.get('plan', '')
            plan_ids = self.tokenizer(plan, add_special_tokens=False).input_ids
            self.plans.append(plan_ids)

            # Hidden state data
            if 'hidden_state' in example:
                hidden_state = example['hidden_state']
                if isinstance(hidden_state, torch.Tensor):
                    self.hidden_states.append(hidden_state)
                else:
                    self.hidden_states.append(torch.tensor(hidden_state, dtype=torch.float32))
            else:
                self.hidden_states.append(None)

        # Verify data consistency
        assert len(self.input_ids) == len(self.plans) == len(self.hidden_states), \
            f"Data length mismatch: input_ids={len(self.input_ids)}, plans={len(self.plans)}, hidden_states={len(self.hidden_states)}"

        rank0_print(f"Dataset loaded: {len(self.input_ids)} examples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        result = dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            human_end_positions=self.human_end_positions[i],
            plan=self.plans[i],  # ensure index alignment
        )

        if self.hidden_states[i] is not None:
            result['prepended_hidden_states'] = self.hidden_states[i]

        return result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with lazy preprocessing and position tracking."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path: str = None):
        super(LazySupervisedDataset, self).__init__()

        rank0_print("Formatting inputs... Skipped in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.model_path = model_path

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        # Only preprocess conversation data
        ret = preprocess_with_position_tracking(
            [self.raw_data[i]["conversations"]],
            self.tokenizer,
            self.model_path
        )

        # Build return dict, directly reading plan and hidden_state from raw data
        result = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            human_end_positions=ret["human_end_positions"][0],
            plan=self.tokenizer(self.raw_data[i].get('plan', ''), add_special_tokens=False).input_ids,
        )

        # Add hidden state (directly from raw data)
        if 'hidden_state' in self.raw_data[i]:
            hidden_state = self.raw_data[i]['hidden_state']
            if isinstance(hidden_state, torch.Tensor):
                result['prepended_hidden_states'] = hidden_state
            else:
                result['prepended_hidden_states'] = torch.tensor(hidden_state, dtype=torch.float32)

        self.cached_data_dict[i] = result
        print(f"DEBUG: Dataset __getitem__ for index {i} returned keys: {result.keys()}")
        return result


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning with plan data."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Process human_end_positions
        human_end_positions = [instance.get("human_end_positions", -1) for instance in instances]
        human_end_positions = torch.tensor(human_end_positions)

        # Process prepended_hidden_states
        prepended_hidden_states = []
        for instance in instances:
            if "prepended_hidden_states" in instance:
                prepended_hidden_states.append(instance["prepended_hidden_states"])
            else:
                prepended_hidden_states.append(None)

        # Process plan data
        plans = [instance.get("plan", "") for instance in instances]

        result = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            human_end_positions=human_end_positions,
            plans=plans,
        )

        if prepended_hidden_states is not None:
            result["prepended_hidden_states"] = prepended_hidden_states

        return result


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        model_path: str = None,
        use_position_tracking: bool = True,
        prepended_length=800
) -> Dict:
    """Create dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )

    hidden_data = data_args.hidden_data

    loader_train = HiddenStateLoader(hidden_data)

    rank0_print("Loading data...")

    # Handle training data
    if data_args.data_path:
        train_json = json.load(open(data_args.data_path, "r"))
        # train_json = train_json[:50]
        print(f"data_args.data_path: {data_args.data_path}")
        for item in train_json:
            merged_value = item['conversations'][0]['value'] + '\n' + item['conversations'][2]['value']
            new_first_entry = {'from': 'human', 'value': merged_value}
            new_first_entry['value'] += '\n' + 'Now, you are given a step-by-step plan to complete this task as follow: '
            item['conversations'][0] = new_first_entry

            hidden_state, plan = loader_train.get_hidden_state_and_plan(item['id'])
            hidden_length = hidden_state.shape[0]
            if hidden_length >= prepended_length:
                hidden_state = hidden_state[:prepended_length, :]

            item['hidden_state'] = hidden_state
            item['plan'] = plan

            # Remove the second entry (original first gpt response)
            del item['conversations'][1:3]

    # === New: split validation set from training set if eval_data_path not provided ===
    if data_args.eval_data_path is None:
        rng = random.Random(42)
        rng.shuffle(train_json)
        n_total = len(train_json)
        n_eval = max(1, int(n_total * data_args.eval_ratio))
        eval_json = train_json[:n_eval]
        train_json = train_json[n_eval:]
    else:
        eval_json = json.load(open(data_args.eval_data_path, "r"))

    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, model_path=model_path)
    eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, model_path=model_path)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
