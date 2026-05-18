#!/usr/bin/env python3
"""
ALFWorld Data Collection with Command Line Arguments

This script collects ALFWorld data using a language model, with all parameters
configurable via command line arguments instead of environment variables.

Usage:
    python alfworld_collection.py --dataset_path ./alfworld_dataset.json --output_dir ./output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import glob
import json
import pickle
import time
import argparse
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
import sys
import datasets
from datasets import Features, Value, Sequence, load_from_disk
from datasets import Dataset as HFDataset
import pandas as pd
import re
import socket
import datetime
import math
from pathlib import Path

# Global variable to collect all data
all_data = []

def save_train_data(task, task_id, plan, hidden_state):
    """Collect task, task_id, and hidden_state for each sample into a global list"""
    # Convert to numpy and squeeze shape (assume hidden_state shape is [1, 64, 3584])
    hidden_np = hidden_state.cpu().numpy().squeeze(0)  # shape: [64, 3584]

    entry = {
        "task": task,
        "task_id": task_id,
        "plan": plan,
        "hidden_state": hidden_np
    }

    all_data.append(entry)
    return None

def convert_to_hf_dataset(data, output_dir="final_output"):
    """Convert collected data to a HuggingFace Dataset and save as Dataset and Parquet files"""
    os.makedirs(output_dir, exist_ok=True)

    # Convert data format
    tasks = [d['task'] for d in data]
    task_ids = [d['task_id'] for d in data]
    plan = [d['plan'] for d in data]
    hidden_states = [d['hidden_state'].astype(np.float32) for d in data]  # list of ndarrays [T, D]

    # Define feature schema
    features = Features({
        'task': Value('string'),
        'task_id': Value('string'),
        'plan': Value('string'),
        'hidden_state': Sequence(Sequence(Value('float32')))  # support variable length [T, D]
    })

    # Create dataset
    hf_dataset = HFDataset.from_dict({
        "task": tasks,
        "task_id": task_ids,
        "plan": plan,
        "hidden_state": [hs.tolist() for hs in hidden_states]  # ndarray -> list
    }, features=features)

    # Save to disk (HuggingFace Dataset format)
    hf_dataset.save_to_disk(os.path.join(output_dir, "hf_dataset"))
    print(f"✅ HuggingFace Dataset saved to {os.path.join(output_dir, 'hf_dataset')}")

    # Also save as Parquet
    parquet_path = os.path.join(output_dir, "data.parquet")
    hf_dataset.to_parquet(parquet_path)
    print(f"✅ Parquet file saved to {parquet_path}")

def save_rank_data(rank, all_data, temp_dir="temp_rank_data"):
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.join(temp_dir, f"rank_{rank}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(all_data, f)
    print(f"Rank {rank} saved data to {filename}")

def finalize_data_save_and_merge(rank, world_size, output_dir="final_output", temp_dir="temp_rank_data"):
    """Each process saves its own data; the main process merges them"""
    if world_size <= 1:
        # Single-process mode: directly save
        convert_to_hf_dataset(all_data, output_dir)
        return

    # Step 1: Each rank saves its data locally
    save_rank_data(rank, all_data, temp_dir=temp_dir)

    # Step 2: Main process waits for others to finish writing
    dist.barrier()

    if rank == 0:
        print("Start merging data from all ranks...")
        full_data = []

        # Load all rank pkl files
        for i in range(world_size):
            filename = os.path.join(temp_dir, f"rank_{i}.pkl")
            if not os.path.exists(filename):
                print(f"Warning: missing file {filename}, skipping rank {i}")
                continue
            with open(filename, "rb") as f:
                rank_data = pickle.load(f)
                full_data.extend(rank_data)

        # Convert and save as HF Dataset and Parquet
        convert_to_hf_dataset(full_data, output_dir)

        # Optional: clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✅ Temporary directory removed: {temp_dir}")

    else:
        print(f"Rank {rank} finished saving data.")


def setup_distributed(args):
    """Initialize distributed training environment"""
    # Print current environment for debugging
    print("Environment variables before setup:")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"Hostname: {socket.gethostname()}")

    # Get system-wide GPU count
    gpu_count = torch.cuda.device_count()
    print(f"Available GPU count: {gpu_count}")

    # Respect existing environment variables
    if 'LOCAL_RANK' not in os.environ and args.local_rank != -1:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # Get rank and world_size
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    rank = int(os.environ.get('RANK', local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size or gpu_count))

    print(f"Using rank: {rank}, local_rank: {local_rank}, world_size: {world_size}")

    try:
        # Set device before initializing process group
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=datetime.timedelta(minutes=args.distributed_timeout),
            init_method=args.init_method,
            world_size=world_size,
            rank=rank
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Process group initialized. Rank: {rank}, World Size: {world_size}")
    except Exception as e:
        print(f"Error initializing process group: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to single-process mode
        rank = 0
        local_rank = 0
        world_size = 1
        print("Falling back to single-process mode")

    return rank, world_size


class MMDataset(Dataset):
    """Dataset class for MMLU data"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_model(model_path, rank, torch_dtype="float32"):
    """Load model and prepare for DDP"""
    device = torch.device(f"cuda:{rank}")

    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype_obj = dtype_mapping.get(torch_dtype, torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype_obj,
        device_map={"": device}
    )

    print(f"Model initialized on GPU {rank}.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def agent_generate(model, tokenizer, text, device, args):
    text = '<|im_start|>user\n' + text + '<|im_end|>\n<|im_start|>assistant\n'
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    input_length = input_ids.shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id  # fallback

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_id,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        return_dict_in_generate=True,
        output_hidden_states=True
    )

    generated_text = tokenizer.decode(
        outputs.sequences[0][input_length:], skip_special_tokens=True
    )

    # Collect hidden states
    hidden_state_list = []
    start_index = max(1, len(outputs.hidden_states) - args.max_hidden_states)
    for i in range(start_index, len(outputs.hidden_states)):
        hidden_state_list.append(outputs.hidden_states[i][-1])

    hidden_state = torch.cat(hidden_state_list, dim=1)

    # Correct extraction: last layer, last position at each step
    step_hiddens = []
    steps = outputs.hidden_states
    start_index = max(0, len(steps) - args.max_hidden_states)
    for i in range(start_index, len(steps)):
        last_layer = steps[i][-1]          # [B, S_i, H]
        h_last = last_layer[:, -1, :]      # [B, H]
        step_hiddens.append(h_last)

    hidden_seq = torch.stack(step_hiddens, dim=1)

    # Single-sample case: remove batch dimension
    if hidden_seq.size(0) == 1:
        hidden_seq = hidden_seq.squeeze(0)

    return generated_text, hidden_state, hidden_seq


def infer_chain(model, tokenizer, task, task_id, device, args):
    """Execute a three-layer reasoning chain with different roles and prompts"""

    # First reasoning layer
    ALFWORLD_TEMPLATE = '''
    Please provide a general plan to solve this task.\n

    The task is: {task}
    '''
    prompt = ALFWORLD_TEMPLATE.format(task=task)
    generated_text1, hidden_state, hidden_seq = agent_generate(
        model, tokenizer, prompt, device, args
    )

    plan = generated_text1
    print(f"Agent 1 output: {generated_text1}")

    save_train_data(task=task, task_id=task_id, plan=plan, hidden_state=hidden_state)

    return hidden_state


def evaluate(model, tokenizer, dataloader, device, rank, world_size, args):
    """Evaluate model performance"""
    task_num = 0
    correct_count = 0

    pbar = tqdm(total=len(dataloader), desc=f"GPU {rank} processing")

    try:
        for batch_idx, task_item in enumerate(dataloader):
            task_num += 1
            task = task_item['conversations'][2]['value'][0]
            task_id = task_item['id'][0]

            # Extra logging to confirm data distribution
            if args.verbose:
                print(
                    f"GPU {rank}, batch {batch_idx}/{len(dataloader)}, "
                    f"processing task starting with: {task[:50]}..."
                )

            _ = infer_chain(model, tokenizer, task, task_id, device, args)
            pbar.update(1)
    except Exception as e:
        print(f"GPU {rank} encountered error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()

    return correct_count, task_num


def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="ALFWorld Data Collection with Configurable Parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_path", type=str,
        default="Interlat_preview/models/Qwen2.5-7B",
        help="Path to the model (HuggingFace model name or local path)"
    )
    model_group.add_argument(
        "--torch_dtype", type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="PyTorch dtype for the model"
    )

    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument(
        "--dataset_path", type=str,
        default="Interlat_preview/datasets/alfworld_sft.json",
        help="Path to the ALFWorld dataset JSON file"
    )
    data_group.add_argument(
        "--output_dir", type=str,
        help="Output directory for collected data (default: auto-generated from temperature)"
    )

    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for text generation"
    )
    gen_group.add_argument(
        "--max_new_tokens", type=int, default=5000,
        help="Maximum number of new tokens to generate"
    )
    gen_group.add_argument(
        "--do_sample", action="store_true", default=True,
        help="Whether to use sampling for generation"
    )
    gen_group.add_argument(
        "--no_sample", dest="do_sample", action="store_false",
        help="Disable sampling (use greedy decoding)"
    )
    gen_group.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    gen_group.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k sampling parameter"
    )
    gen_group.add_argument(
        "--num_beams", type=int, default=1,
        help="Number of beams for beam search"
    )
    gen_group.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="Repetition penalty for generation"
    )
    gen_group.add_argument(
        "--max_hidden_states", type=int, default=10000,
        help="Maximum number of hidden states to collect"
    )

    # Distributed training
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training"
    )
    dist_group.add_argument(
        "--world_size", type=int, default=None,
        help="World size for distributed training (auto-detected if not specified)"
    )
    dist_group.add_argument(
        "--distributed_backend", type=str, default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend"
    )
    dist_group.add_argument(
        "--distributed_timeout", type=int, default=300,
        help="Distributed training timeout in minutes"
    )
    dist_group.add_argument(
        "--init_method", type=str, default="env://",
        help="Initialization method for distributed training"
    )

    # Storage configuration
    storage_group = parser.add_argument_group('Storage Configuration')
    storage_group.add_argument(
        "--temp_dir", type=str, default="temp_rank_data",
        help="Temporary directory for rank data during distributed processing"
    )

    # Miscellaneous
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging"
    )
    misc_group.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for data processing"
    )
    misc_group.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of workers for data loading"
    )

    return parser


def validate_arguments(args):
    """Validate and process arguments"""
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"./alfworld_data_temp_{args.temperature}"

    # Validate paths
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate model path (if local)
    if not args.model_path.startswith(('http://', 'https://')) and not '/' in args.model_path:
        # Assume it's a HuggingFace model name, no validation needed
        pass
    elif os.path.exists(args.model_path):
        # Local model path exists
        pass
    elif '/' in args.model_path and not os.path.exists(args.model_path):
        # Might be HuggingFace format like "Qwen/Qwen2.5-7B-Instruct"
        print(f"Model path appears to be HuggingFace format: {args.model_path}")

    return args


def main():
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    args = validate_arguments(args)

    # Print configuration
    print("=" * 50)
    print("ALFWorld Data Collection Configuration")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Torch dtype: {args.torch_dtype}")
    print("=" * 50)

    # Initialize distributed environment
    rank, world_size = setup_distributed(args)
    print(f"After setup: Rank = {rank}, World Size = {world_size}")

    # Device setup
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f"cuda:{local_rank}")

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, local_rank, args.torch_dtype)

    if rank == 0:
        print(f'Dataset path: {args.dataset_path}')
        print(f'Output dir: {args.output_dir}')

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        full_train_dataset = json.load(f)

    print(f"Training set size: {len(full_train_dataset)}")

    full_train_dataset = HFDataset.from_list(full_train_dataset)

    # Data sharding
    if world_size > 1:
        total_samples = len(full_train_dataset)
        samples_per_worker = math.ceil(total_samples / world_size)
        start_idx = rank * samples_per_worker
        end_idx = min(start_idx + samples_per_worker, total_samples)
        my_indices = list(range(start_idx, end_idx))
        my_dataset = full_train_dataset.select(my_indices)
        print(
            f"GPU {rank} handling samples {start_idx} to {end_idx-1}, "
            f"total: {len(my_dataset)}"
        )
    else:
        my_dataset = full_train_dataset
        print(f"Single process mode - handling all {len(my_dataset)} samples")

    mm_dataset = MMDataset(my_dataset)
    dataloader = DataLoader(
        mm_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"GPU {rank} number of samples to process: {len(dataloader)}")

    # Start evaluation
    correct_count, task_num = evaluate(
        model, tokenizer, dataloader, device, rank, world_size, args
    )

    # Synchronize processes
    if world_size > 1:
        try:
            dist.barrier()
        except Exception as e:
            print(f"Barrier error on GPU {rank}: {e}")

    # Final data save
    finalize_data_save_and_merge(
        rank, world_size, output_dir=args.output_dir, temp_dir=args.temp_dir
    )

    # Cleanup
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"Error destroying process group on GPU {rank}: {e}")

    print(f"GPU {rank} completed processing {task_num} tasks.")


if __name__ == "__main__":
    main()
