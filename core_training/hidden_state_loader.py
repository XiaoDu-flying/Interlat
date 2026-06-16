import time
import random
import string
import numpy as np
import pandas as pd
import torch
import datasets
from datasets import load_dataset, load_from_disk, Split
from tqdm import tqdm
from pathlib import Path

class HiddenStateLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        # self.id_to_hidden_state = {}

        # Automatically load data during initialization
        self._load_data()

    def _load_data(self):
        print(f"Loading tensor data from {self.dataset_name}")
        dataset_path = Path(self.dataset_name)
        if dataset_path.exists():
            if (dataset_path / "dataset_info.json").exists() or (dataset_path / "state.json").exists():
                self.dataset = load_from_disk(str(dataset_path))
            elif (dataset_path / "hf_dataset").is_dir():
                self.dataset = load_from_disk(str(dataset_path / "hf_dataset"))
            elif (dataset_path / "data.parquet").is_file():
                self.dataset = load_dataset("parquet", data_files=str(dataset_path / "data.parquet"), split="train")
            elif (dataset_path / "data_full.parquet").is_file():
                self.dataset = load_dataset("parquet", data_files=str(dataset_path / "data_full.parquet"), split="train")
            else:
                self.dataset = datasets.load_dataset(str(dataset_path), split=datasets.Split.TRAIN)
        else:
            self.dataset = datasets.load_dataset(self.dataset_name, split=datasets.Split.TRAIN)

        print(f"Loaded {len(self.dataset)} records.")

        def optimized_convert_nested_arrays_with_plan(df):
            """Optimized nested array conversion (following PyTorch recommendations) + including plan text"""

            print(f"Optimized converting {len(df)} nested arrays with plan text...")
            start_time = time.time()

            def optimized_nested_convert(nested_array):
                """Optimized nested array conversion"""
                try:
                    # Follow PyTorch recommendation: convert to a single NumPy array first, then to tensor
                    if isinstance(nested_array, np.ndarray) and nested_array.dtype == object:
                        # Use numpy.array() to convert list into a single NumPy array
                        list_data = nested_array.tolist()
                        numpy_array = np.array(list_data, dtype=np.float32)
                        return torch.from_numpy(numpy_array).to(torch.bfloat16)
                    else:
                        # If not an object array, convert directly
                        return torch.from_numpy(nested_array.astype(np.float32))

                except Exception as e:
                    print(f"Conversion failed: {e}")
                    return None

            # Vectorized pandas conversion for hidden_state
            df['tensor_hidden_state'] = df['hidden_state'].apply(optimized_nested_convert)

            # Check conversion results
            success_mask = df['tensor_hidden_state'].notna()
            success_count = success_mask.sum()

            print(f"Successfully converted: {success_count}/{len(df)} arrays")

            # Build dictionary containing hidden_state and plan
            valid_df = df[success_mask]

            # Option 1: create a nested dictionary structure
            id_to_data = {}

            # Use tqdm to add a progress bar
            for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Building id_to_data"):
                id_to_data[row['task_id']] = {
                    'hidden_state': row['tensor_hidden_state'],
                    'plan': row['plan']
                }

            conversion_time = time.time() - start_time
            print(f"Optimized conversion completed in {conversion_time:.2f} seconds")

            # Verify results
            if id_to_data:
                sample_key = next(iter(id_to_data))
                sample_data = id_to_data[sample_key]
                print(f"Sample tensor shape: {sample_data['hidden_state'].shape}")
                print(f"Sample tensor dtype: {sample_data['hidden_state'].dtype}")
                print(f"Sample plan: {sample_data['plan'][:100]}...")  # Show only first 100 characters

            return id_to_data

        # Show a progress bar (as a single overall step)
        with tqdm(total=1, desc="Converting Dataset to Pandas") as pbar:
            df = self.dataset.to_pandas()
            pbar.update(1)

        self.id_to_data = optimized_convert_nested_arrays_with_plan(df)

    # Query function is very efficient
    def get_hidden_state_and_plan(self, task_id):
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        return self.id_to_data[task_id]['hidden_state'], self.id_to_data[task_id]['plan']
