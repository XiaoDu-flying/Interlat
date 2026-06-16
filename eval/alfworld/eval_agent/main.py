import os
import json
import logging
import pathlib
import argparse
import sys
from typing import Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
import torch
import datasets
import time
import numpy as np
import math
import re
import csv

_EVAL_AGENT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_AGENT_DIR.parents[2]
_CORE_TRAINING_DIR = _REPO_ROOT / "core_training"
if _CORE_TRAINING_DIR.is_dir() and str(_CORE_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_TRAINING_DIR))

try:
    from core_training.fastchat.model.model_adapter import get_conversation_template
except ImportError:
    from fastchat.model.model_adapter import get_conversation_template

import tasks as tasks
import agents as agents
import envs as envs
from utils.datatypes import State
from typing import Optional, Tuple
from pathlib import PurePath

logger = logging.getLogger("agent_frame")

# ====================== Performance Testing Tools ======================
CSV_FIELDS = [
    "dataset_split", "task_id", "seed",
    "keep_ratio_pct", "n_percent_removed",
    "success", "steps",
    "wall_time_ms_total", "model_time_ms", "env_time_ms",
    "gen_tokens", "tokens_per_s", "max_mem_gb",
    "ttfa_decision_ms", "ttfa_step1_ms",
    "L_hidden_full", "L_hidden_kept", "L_before", "L_after", "L_total",
    "gpu_name", "dtype", "env_class", "agent_class", "model_path", "notes"
]

def now_ms():
    return time.perf_counter() * 1000.0

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def append_csv_row(path: str, row: Dict[str, Any]):
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            wr.writeheader()
        wr.writerow({k: row.get(k, "") for k in CSV_FIELDS})

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3  # GB
    return 0.0

# ==================== Configuration Management ====================
def load_model_configs(config_path: str = None) -> Dict[str, str]:
    """Load model configurations"""
    default_configs = {
        # Example model configurations
        # Replace these paths with your actual model paths
        "qwen7b_base_v1": "/path/to/your/models/Qwen7B_base/checkpoint-1",
        "qwen7b_base_v2": "/path/to/your/models/Qwen7B_base/checkpoint-2",
        "qwen05b_base": "/path/to/your/models/Qwen0.5B_base/checkpoint-3",
        "llama8b_base": "/path/to/your/models/Llama8B_base/checkpoint-4",
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return default_configs

def load_dataset_configs(config_path: str = None) -> Dict[str, str]:
    """Load dataset configurations"""
    default_configs = {
        # "qwen7b": "your_data",
        "qwen7b": "pailitao_v100/alfworld_llama8B_seen_unseen",
        "qwen05b": "your_data",
        "llama8b": "your_data",
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return default_configs

class HiddenStateLoader:
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.split = split
        self._load_data()

    def _load_data(self):
        print(f"Loading tensor data from {self.dataset_name}")
        if self.split == 'valid_seen' or self.split == 'dev':
            self.dataset = datasets.load_dataset(self.dataset_name, split=datasets.Split.TEST)
        else:
            self.dataset = datasets.load_dataset(self.dataset_name, split=datasets.Split.VALIDATION)
        print(f"Loaded {len(self.dataset)} records.")

        def optimized_convert_nested_arrays_with_plan(df):
            """Optimized nested array conversion (following PyTorch recommendations) + plan text"""

            print(f"Optimized converting {len(df)} nested arrays with plan text...")
            start_time = time.time()

            def optimized_nested_convert(nested_array):
                """Optimized nested array conversion"""
                try:
                    if isinstance(nested_array, np.ndarray) and nested_array.dtype == object:
                        list_data = nested_array.tolist()
                        numpy_array = np.array(list_data, dtype=np.float32)
                        return torch.from_numpy(numpy_array)
                    else:
                        return torch.from_numpy(nested_array.astype(np.float32))
                        
                except Exception as e:
                    print(f"Conversion failed: {e}")
                    return None
            
            df['tensor_hidden_state'] = df['hidden_state'].apply(optimized_nested_convert)
            
            success_mask = df['tensor_hidden_state'].notna()
            success_count = success_mask.sum()
            
            print(f"Successfully converted: {success_count}/{len(df)} arrays")
            
            valid_df = df[success_mask]
            
            id_to_data = {}
            for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Building id_to_data"):
                row['task'] = row['task'].replace('\n\n', '\n')
                id_to_data[row['task']] = {
                    'hidden_state': row['tensor_hidden_state'],
                    'plan': row['plan']
                }
          
            conversion_time = time.time() - start_time
            print(f"Optimized conversion completed in {conversion_time:.2f} seconds")
            
            if id_to_data:
                sample_key = next(iter(id_to_data))
                sample_data = id_to_data[sample_key]
                print(f"Sample tensor shape: {sample_data['hidden_state'].shape}")
                print(f"Sample tensor dtype: {sample_data['hidden_state'].dtype}")
                print(f"Sample plan: {sample_data['plan'][:100]}...")
            
            return id_to_data

        with tqdm(total=1, desc="Converting Dataset to Pandas") as pbar:
            df = self.dataset.to_pandas()
            pbar.update(1)

        self.id_to_data = optimized_convert_nested_arrays_with_plan(df)

    def get_hidden_state_and_plan(self, task_id):
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        return self.id_to_data[task_id]['hidden_state'], self.id_to_data[task_id]['plan']

# ==================== Utility Functions ====================
def sample_cov_gauss_from_hidden(
    hidden_state: torch.Tensor,
    use_mean: bool = False,
) -> torch.Tensor:
    """
    Resample hidden states according to CovGauss definition in paper:
    - use_mean = False  -> CovGauss-0μ:  H_new ~ N(0, Σ̂)
    - use_mean = True   -> CovGauss-μ :  H_new ~ N(μ̂, Σ̂)

    Assumes hidden_state shape is [T, D]
    """
    if hidden_state.dim() != 2:
        raise ValueError(f"hidden_state must be [T, D], got {hidden_state.shape}")

    T, D = hidden_state.shape
    # Too short to estimate covariance, return original H
    if T < 2:
        return hidden_state

    # 1) Calculate sample mean μ̂
    mu = hidden_state.mean(dim=0, keepdim=True)     # [1, D]

    # 2) Calculate centered X = H - μ̂
    X = hidden_state - mu                           # [T, D]

    # 3) Use "row space sampling" trick to sample N(0, Σ̂)
    #    Σ̂ = (X^T X) / (T-1)
    #    If Z ~ N(0, I_T), then Y = Z X / sqrt(T-1) satisfies Cov(Y) = Σ̂
    Z = torch.randn(T, T, device=hidden_state.device, dtype=hidden_state.dtype)
    H_new = (Z @ X) / math.sqrt(max(T - 1, 1))      # [T, D], ~ N(0, Σ̂)

    # 4) Whether to add back mean μ̂
    if use_mean:
        H_new = H_new + mu

    return H_new

def random_rot_hidden(hidden_state: torch.Tensor) -> torch.Tensor:
    """
    RandomRot: H' = μ̂ + (H - μ̂) Σ̂^{-1/2} Q Σ̂^{1/2}
    Where Q is a Haar random orthogonal matrix, ensuring:
    - Exactly preserve mean / covariance
    - Shuffle higher-order structure and specific representations

    Implemented using SVD to avoid direct eigendecomposition of D×D covariance matrix.
    """
    if hidden_state.dim() != 2:
        raise ValueError(f"hidden_state must be [T, D], got {hidden_state.shape}")

    T, D = hidden_state.shape
    if T < 2:
        return hidden_state

    orig_dtype = hidden_state.dtype
    H = hidden_state.to(torch.float32)
    device = H.device

    # 1) Mean & centering
    mu = H.mean(dim=0, keepdim=True)   # [1, D]
    X = H - mu                         # [T, D]

    # 2) Economy SVD on X / sqrt(T-1):
    #    X_norm = U S V^T  => Σ̂ = V diag(S^2) V^T
    X_norm = X / math.sqrt(max(T - 1, 1))
    U, S, Vh = torch.linalg.svd(X_norm, full_matrices=False)  # U: [T, r], S: [r], Vh: [r, D]
    V = Vh.transpose(0, 1)                                   # [D, r]
    r = S.shape[0]

    # 3) Generate r×r Haar random orthogonal matrix Q_r
    A = torch.randn(r, r, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)  # QR -> Q is orthogonal matrix
    # Adjust sign to ensure det(Q) > 0 (orthogonal group standardization, optional)
    try:
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
    except RuntimeError:
        # det may be unstable for large r, can safely ignore
        pass

    # 4) Construct M = Σ^{-1/2} Q Σ^{1/2} in eigenspace equivalent form:
    #    Σ^{1/2} = diag(S), Σ^{-1/2} = diag(1/S)
    eps = 1e-6
    S_safe = S.clamp_min(eps)                       # [r]
    # Right multiply diag(S): Q * diag(S) -> multiply each column by S_j
    B = Q * S_safe                                  # [r, r]
    # Left multiply diag(1/S): diag(1/S) @ B -> multiply each row by 1/S_i
    M = (1.0 / S_safe).unsqueeze(1) * B             # [r, r]

    # 5) Transform in eigenspace then project back to original space:
    #    Z_r = X V_r      (r-dimensional eigenspace)
    #    Z_r' = Z_r M
    #    H' = μ̂ + Z_r' V_r^T
    V_r = V                                         # [D, r]
    Z_r = X @ V_r                                   # [T, r]
    Z_r_prime = Z_r @ M                             # [T, r]
    H_prime = mu + Z_r_prime @ V_r.transpose(0, 1)  # [T, D]

    return H_prime.to(orig_dtype)

# ==================== Main Logic ====================
def interactive_loop(
    task: tasks.Task,
    loader,
    agent: agents.LMAgent,
    pruning_config: Dict[str, Any],
    env_config: Dict[str, Any],
    args: argparse.Namespace,
    enable_timing: bool = False,
    csv_path: str = None,
) -> State:
    # Performance monitoring initialization
    if enable_timing:
        wall_start = now_ms()
        model_time_total = 0.0
        env_time_total = 0.0
        gen_tokens_total = 0
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)

    game_file = getattr(task, 'game_file', None)
    if game_file:
        observation, state = env.reset([game_file])

    hidden_state = loader.id_to_data[state.history[2]['content']]['hidden_state']
    original_hidden_len = hidden_state.shape[0] if hidden_state is not None else 0

    # Process hidden_state based on configuration
    method = pruning_config['method']

    # Process method variants directly

    if method == 'text':
        # Text variant: Replace latent messages with corresponding CoT plan
        hidden_state = None
    elif method == 'no_comm':
        # No-Comm: Remove communication entirely
        hidden_state = None
    elif method == 'no_cot':
        # No-CoT: Directly predict final answers without any plan
        hidden_state = None
    elif method == 'cross_task':
        # CrossTask: Replace current task's latents with one sampled from different task
        current_task_id = state.history[2]['content']
        all_task_ids = list(loader.id_to_data.keys())
        other_task_ids = [tid for tid in all_task_ids if tid != current_task_id]
        if other_task_ids:
            random_task_id = np.random.choice(other_task_ids)
            hidden_state = loader.id_to_data[random_task_id]['hidden_state']
    elif method == 'noised':
        # Noised: Add structured or unstructured perturbations to H
        noise_std = pruning_config.get('noise_std', 1.0)
        hidden_state = hidden_state + torch.randn_like(hidden_state) * noise_std
    elif method == 'covgauss0':
        # CovGauss: N(0, Σ̂) - preserve covariance statistics while destroying higher-order structure
        hidden_state = sample_cov_gauss_from_hidden(hidden_state, use_mean=False)
    elif method == 'covgauss1':
        # CovGauss: N(μ̂, Σ̂) - preserve mean and covariance statistics
        hidden_state = sample_cov_gauss_from_hidden(hidden_state, use_mean=True)
    elif method == 'randomrot':
        # RandomRot: Preserve mean/covariance while destroying higher-order structure
        hidden_state = random_rot_hidden(hidden_state)
    elif method == 'qwen2llama':
        # Qwen2LLaMA: Use latents from different model family
        cross_family = True
    elif method == 'cot_full':
        # CoT (full) baseline: Use complete CoT plans for full-parameter supervised fine-tuning
        pass  # Keep original hidden_state
    elif method == 'none':
        # Complete hidden states (our method)
        pass  # Keep original hidden_state
    # Any other case: keep original hidden states

    print(f"Processed hidden_state shape: {hidden_state.shape if hidden_state is not None else 'None'}")

    # Merge and process history
    if method == 'cot_full':
        # For CoT (full) baseline, get plan and add it to the content
        plan = loader.id_to_data[state.history[2]['content']]['plan']
        merged_content = state.history[0]['content'] + "\n" + state.history[2]['content'] + '\nNow, you are given a step by step plan as follow: '
        plan = '<bop>' + plan + '<eop>'
        state.history[0]['content'] = merged_content + plan
    else:
        merged_content = state.history[0]['content'] + "\n" + state.history[2]['content'] + 'Now, you are given a step by step plan as follow: '
        # merged_content = state.history[0]['content'] + "\n" + state.history[2]['content']
        state.history[0]['content'] = observation
    del state.history[1]
    del state.history[1]

    init_msg = observation
    logger.info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

    cur_step = 1
    while not state.finished:
        logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1
        
        try:
            # Model inference time monitoring
            if enable_timing:
                cuda_sync()
                model_start = now_ms()

            # Select input type based on method
            if args.plan_only_mode or method == "text":
                # Text variant: Use plan text instead of hidden states
                try:
                    _, plan_text = loader.get_hidden_state_and_plan(state.history[2]['content'])
                    llm_output: str = agent(state.history, plan_text)
                except Exception:
                    llm_output: str = agent(state.history, "")
            elif method == "cot_full":
                # CoT (full) baseline: Plan is already included in state.history, use None for hidden states
                llm_output: str = agent(state.history, None)
            elif method in ["no_comm", "no_cot"]:
                # No-Comm or No-CoT: Use no additional information
                llm_output: str = agent(state.history, None)
            else:
                # Standard case: Use processed hidden states
                llm_output: str = agent(state.history, hidden_state)

            if enable_timing:
                cuda_sync()
                model_time_total += now_ms() - model_start
                # Simple token generation count (approximation)
                gen_tokens_total += len(llm_output.split())

            # llm_output = llm_output[:-10]
            print(f"llm_output: {llm_output}")
            logger.info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n")
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            print(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break

        # Environment execution time monitoring
        if enable_timing:
            env_start = now_ms()

        observation, state = env.step(llm_output)

        if enable_timing:
            env_time_total += now_ms() - env_start
        print(f"Observation: {observation}")
        
        if not state.finished:
            logger.info(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n")

        if state.finished:
            break

    if state.reward is not None:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}")
    else:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}")

    # Record performance data to CSV
    if enable_timing and csv_path:
        wall_total = now_ms() - wall_start
        max_mem_gb = get_gpu_memory_usage()

        kept_hidden_len = hidden_state.shape[0] if hidden_state is not None else 0
        # Restore original method name for recording
        original_method = pruning_config['method']
        tokens_per_s = gen_tokens_total / (model_time_total / 1000) if model_time_total > 0 else 0

        # Get GPU name
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        csv_row = {
            "task_id": getattr(task, 'task_id', 'unknown'),
            "success": state.success,
            "steps": state.steps,
            "wall_time_ms_total": wall_total,
            "model_time_ms": model_time_total,
            "env_time_ms": env_time_total,
            "gen_tokens": gen_tokens_total,
            "tokens_per_s": tokens_per_s,
            "max_mem_gb": max_mem_gb,
            "L_hidden_full": original_hidden_len,
            "L_hidden_kept": kept_hidden_len,
            "gpu_name": gpu_name,
            "env_class": env_config.get('env_class', ''),
            "notes": f"method_{original_method}",
        }
        append_csv_row(csv_path, csv_row)

    return state

def run_single_iteration(args: argparse.Namespace, iteration: int):
    # No special handling needed for the new method names
    timestamp = int(time.time())
    
    # Load experiment configuration
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    # Get model path
    model_configs = load_model_configs(args.model_config_file)
    if args.model_key:
        MODEL = model_configs.get(args.model_key)
        if MODEL is None:
            raise ValueError(f"Model key '{args.model_key}' not found in model configs")
    elif args.model_path:
        MODEL = args.model_path
    else:
        raise ValueError("Either --model_key or --model_path must be provided")

    args.model_name = get_conversation_template(MODEL)

    config_path = os.path.join(MODEL, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    args.model_name = os.path.basename(os.path.normpath(data["_name_or_path"]))
        
    if args.model_name is not None:
        agent_config['config']['model_name'] = f"hidden/{args.model_name}-{timestamp}"
    print(agent_config)

    # Build method variant configuration
    pruning_config = {
        'method': args.variants,
        'n_percent': args.dropout_percent,
        'keep_ratio': args.keep_ratio,
        'threshold': args.pruning_threshold,
        'min_keep': args.min_keep,
        'leverage_rank': args.leverage_rank,
        'weights': [float(w) for w in args.pruning_weights.split(',')] if args.pruning_weights else [0.6, 0.3, 0.1],
        'noise_std': args.noise_std,
    }
    
    p = PurePath(MODEL)
    last_two = p.parts[-2] + '/' + p.parts[-1]
    print(last_two)

    # Build output path
    if args.output_path == "":
        output_path = os.path.join(
            args.experiment_type,
            args.split,
            args.variants,
            last_two,
            agent_config['config']['model_name'].replace('/', '_'),
            args.exp_config + args.exp_name
        )
    else:
        output_path = args.output_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Setup logging
    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    # Environment initialization
    if env_config['env_class'] == 'WebShopEnv':
        from webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)
    elif env_config['env_class'] == 'TextCraftEnv':
        from eval_agent.envs.textcraft_env import TextCraft
        env_config['env'] = TextCraft(minecraft_dir="eval_agent/envs")

    # Initialize tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    # Get dataset path
    dataset_configs = load_dataset_configs(args.dataset_config_file)
    if args.dataset_key:
        data_path = dataset_configs.get(args.dataset_key)
        if data_path is None:
            raise ValueError(f"Dataset key '{args.dataset_key}' not found in dataset configs")
    elif args.dataset_path:
        data_path = args.dataset_path
    else:
        raise ValueError("Either --dataset_key or --dataset_path must be provided")

    loader = HiddenStateLoader(data_path, args.split)

    # cross_family = True

    if pruning_config['method'] == "cross_family":
        cross_family=True
    else:
        cross_family=False
    
    # Initialize agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"], 
        MODEL,
        cross_family,
    )

    state_list = []
    done_task_id = []
    
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")

    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")
        reward_list = []
        success_list = []
        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
            success_list.append(state.success)

        if len(reward_list) != 0:
            logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")
        return

    # Run tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks. Iteration {iteration}")
    n_todo_tasks = n_tasks - len(done_task_id)

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            if args.debug and i == 5:
                break

            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            # Build CSV path (if performance monitoring enabled)
            csv_path = None
            if args.enable_timing:
                csv_path = os.path.join(output_path, "performance_metrics.csv")

            state = interactive_loop(
                task, loader, agent, pruning_config, env_config, args,
                enable_timing=args.enable_timing,
                csv_path=csv_path
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            pbar.update(1)
        pbar.close()
    
    logger.warning(f"Iteration {iteration} completed.")
    logger.warning(f"Output saved to {output_path}")

    # Calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")

def main(args: argparse.Namespace):
    # Set environment variables
    if args.openlm_token:
        os.environ["OPENLM_TOKEN"] = args.openlm_token
    
    iteration = 1
    while True:
        try:
            print(f"\nStarting iteration {iteration} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            run_single_iteration(args, iteration)
            iteration += 1
            if not args.loop:
                break
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Stopping the loop.")
            break
        except Exception as e:
            print(f"Error in iteration {iteration}: {str(e)}")
            if args.loop:
                time.sleep(10)
            else:
                raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    
    # Basic parameters
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiment.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default=str(_EVAL_AGENT_DIR / "configs" / "task"),
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default=str(_EVAL_AGENT_DIR / "configs" / "model"),
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="hidden",
        help="Config of model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Custom output path."
    )
    parser.add_argument(
        "--loop",
        default=True,
        help="Whether to run in infinite loop mode."
    )
    
    # Model configuration parameters
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help="Key to select model from predefined configs (e.g., 'qwen7b_base_v1')."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Direct path to model directory."
    )
    parser.add_argument(
        "--model_config_file",
        type=str,
        default=None,
        help="Path to custom model config JSON file."
    )
    
    # Dataset configuration parameters
    parser.add_argument(
        "--dataset_key",
        type=str,
        help="Key to select dataset from predefined configs (e.g., 'qwen7b')."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='your_latent_communication_data',
        help="Direct path to dataset."
    )
    parser.add_argument(
        "--dataset_config_file",
        type=str,
        default=None,
        help="Path to custom dataset config JSON file."
    )
    
    # Experiment type parameters
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="Experiment",
        help="Type of experiment for organizing output directories."
    )
    
    # Environment variable parameters
    parser.add_argument(
        "--openlm_token",
        type=str,
        help="OPENLM_TOKEN environment variable value."
    )
    
    # Method variant parameters
    parser.add_argument(
        "--variants",
        type=str,
        default="none",
        choices=[
            # === Main Method ===
            "none",             # Complete hidden states (our method)

            # === Baselines ===
            "cot_full",         # CoT (full): Complete CoT plans for full-parameter supervised fine-tuning
            "no_cot",           # No-CoT: Directly predict final answers without any plan

            # === Variants ===
            "text",             # Text: Replace latent messages with corresponding CoT plan
            "no_comm",          # No-Comm: Remove communication entirely
            "cross_task",       # CrossTask: Replace current task's latents with one sampled from different task
            "noised",           # Noised: Add structured or unstructured perturbations to H
            "covgauss0",        # CovGauss: Preserve mean or covariance statistics (0μ variant)
            "covgauss1",        # CovGauss: Preserve mean or covariance statistics (μ variant)
            "randomrot",        # RandomRot: Preserve mean/covariance while destroying higher-order structure
            "qwen2llama"        # Qwen2LLaMA: Use latents from Qwen2.5-7B to train LLaMA3.1-8B model
        ],
        help="Variants and baselines for hidden state processing."
    )
    parser.add_argument(
        "--dropout_percent",
        type=float,
        default=0.0,
        help="Percentage to drop from end for dropout method (0-100)."
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.7,
        help="Ratio of tokens to keep for importance method (0-1)."
    )
    parser.add_argument(
        "--pruning_threshold",
        type=float,
        default=None,
        help="Absolute threshold for importance method."
    )
    parser.add_argument(
        "--min_keep",
        type=int,
        default=8,
        help="Minimum number of tokens to keep."
    )
    parser.add_argument(
        "--leverage_rank",
        type=int,
        default=16,
        help="Rank for SVD leverage calculation."
    )
    parser.add_argument(
        "--pruning_weights",
        type=str,
        default="0.6,0.3,0.1",
        help="Weights for norm, leverage, and plan mask (comma-separated)."
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.5,
        help="Standard deviation for noise addition."
    )

    # Performance monitoring parameters
    parser.add_argument(
        "--enable_timing",
        action="store_true",
        help="Enable performance timing and CSV logging."
    )
    parser.add_argument(
        "--plan_only_mode",
        action="store_true",
        help="Use plan-only mode (no hidden states)."
    )
    parser.add_argument(
        "--first_k_tokens",
        type=int,
        default=None,
        help="Keep only first k tokens from hidden state."
    )

    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)
