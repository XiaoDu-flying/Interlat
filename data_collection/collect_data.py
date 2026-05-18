#!/usr/bin/env python3
"""Unified entrypoint for Interlat data collection.

This wrapper preserves the public CLI documented in the README and used by
scripts/quick_start.sh while forwarding task-specific arguments to the real
collection implementations.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


TASK_TO_SCRIPT = {
    "alfworld": "alfworld_collection.py",
    "math": "math_collection.py",
}


def print_help() -> None:
    task_list = "|".join(TASK_TO_SCRIPT)
    print("Interlat unified data collection CLI")
    print(f"Usage: {Path(sys.argv[0]).name} <{task_list}> [task options]")
    print("")
    print("Tasks:")
    print("  alfworld    Collect hidden states from ALFWorld tasks")
    print("  math        Collect hidden states from MATH reasoning tasks")
    print("")
    print("Examples:")
    print("  python data_collection/collect_data.py alfworld --dataset_path ./datasets/alfworld_dataset.json")
    print("  python data_collection/collect_data.py math --mode train --subjects algebra geometry")
    print("")
    print("Tip: append '--help' after a task name to see task-specific options.")


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print_help()
        return 0

    task = sys.argv[1]
    task_args = sys.argv[2:]

    if task not in TASK_TO_SCRIPT:
        print(f"Error: unsupported task '{task}'.", file=sys.stderr)
        print_help()
        return 1

    script_path = Path(__file__).resolve().parent / TASK_TO_SCRIPT[task]
    if not script_path.is_file():
        print(f"Error: task implementation not found: {script_path}", file=sys.stderr)
        return 1

    command = [sys.executable, str(script_path), *task_args]
    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
