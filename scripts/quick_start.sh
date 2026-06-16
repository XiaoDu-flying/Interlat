#!/bin/bash
# Quick Start Script for Interlat
# This script provides a complete end-to-end workflow for getting started with Interlat

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Default configurations
TASK="${TASK:-alfworld}"
MODEL_SIZE="${MODEL_SIZE:-small}"
QUICK_MODE="${QUICK_MODE:-true}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --full)
            QUICK_MODE="false"
            shift
            ;;
        --help|-h)
            echo "Interlat Quick Start Script"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --task <alfworld|math>    Task type to run (default: alfworld)"
            echo "  --model-size <small|large> Model size to use:"
            echo "                           small: Qwen2.5-0.5B (fast, for testing)"
            echo "                           large: Qwen2.5-7B (better performance)"
            echo "  --full                   Run full workflow (disable quick mode)"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Quick mode (default):"
            echo "  - Uses smaller model and datasets for fast testing"
            echo "  - 100 samples for data collection"
            echo "  - 2 epochs for training"
            echo ""
            echo "Full mode (--full):"
            echo "  - Uses complete datasets"
            echo "  - Full training epochs"
            echo "  - Better for actual research/production use"
            echo ""
            echo "Examples:"
            echo "  $0                           # Quick ALFWorld demo"
            echo "  $0 --task math --full       # Full math reasoning workflow"
            echo "  $0 --model-size large       # Use larger model"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate task
if [[ "$TASK" != "alfworld" && "$TASK" != "math" ]]; then
    print_error "Task must be 'alfworld' or 'math', got: $TASK"
    exit 1
fi

# Set model based on size preference
if [[ "$MODEL_SIZE" == "small" ]]; then
    if [[ "$TASK" == "alfworld" ]]; then
        MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
    else
        MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
    fi
elif [[ "$MODEL_SIZE" == "large" ]]; then
    if [[ "$TASK" == "alfworld" ]]; then
        MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    else
        MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    fi
else
    print_error "Model size must be 'small' or 'large', got: $MODEL_SIZE"
    exit 1
fi

# Set parameters based on mode
if [[ "$QUICK_MODE" == "true" ]]; then
    MAX_SAMPLES="100"
    EPOCHS="2"
    MODE_DESC="Quick Demo"
else
    MAX_SAMPLES="-1"  # All samples
    EPOCHS="10"
    MODE_DESC="Full Workflow"
fi

echo "========================================================"
echo -e "${BLUE}Interlat Quick Start - $MODE_DESC${NC}"
echo "========================================================"
echo "Task: $TASK"
echo "Model: $MODEL_NAME"
echo "Mode: $MODE_DESC"
if [[ "$QUICK_MODE" == "true" ]]; then
    echo "Samples: $MAX_SAMPLES (limited for quick demo)"
    echo "Epochs: $EPOCHS (reduced for quick demo)"
else
    echo "Samples: All available"
    echo "Epochs: $EPOCHS"
fi
echo "========================================================"

# Create directories
mkdir -p data/datasets data/logs data/models
DATA_DIR="./data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="$DATA_DIR/experiment_${TASK}_${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

print_step "1" "Environment Setup"

# Check if required scripts exist
REQUIRED_SCRIPTS=(
    "data_collection/collect_data.py"
    "core_training/train.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -f "$script" ]]; then
        print_error "Required script not found: $script"
        echo "Please ensure you're running this from the project root directory"
        exit 1
    fi
done

print_success "Environment check passed"

# Step 2: Data Collection
print_step "2" "Hidden State Data Collection"

if [[ "$TASK" == "alfworld" ]]; then
    # Check for ALFWorld dataset
    DATASET_PATH="$DATA_DIR/datasets/alfworld_dataset.json"
    if [[ ! -f "$DATASET_PATH" ]]; then
        print_warning "ALFWorld dataset not found at $DATASET_PATH"
        print_warning "You'll need to provide your own ALFWorld dataset"
        echo "Please place your ALFWorld dataset at: $DATASET_PATH"
        echo "Or modify the DATASET_PATH variable in this script"
        exit 1
    fi

    HIDDEN_DATA_DIR="$EXPERIMENT_DIR/alfworld_hidden_states"

    python data_collection/collect_data.py alfworld \
        --dataset_path "$DATASET_PATH" \
        --model_path "$MODEL_NAME" \
        --output_dir "$HIDDEN_DATA_DIR" \
        --temperature 0.7 \
        --max_new_tokens 1500 \
        --verbose

else
    # Math data collection
    HIDDEN_DATA_DIR="$EXPERIMENT_DIR/math_hidden_states"

    SUBJECTS="algebra geometry"  # Limited subjects for quick demo
    if [[ "$QUICK_MODE" == "false" ]]; then
        SUBJECTS="algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus"
    fi

    python data_collection/collect_data.py math \
        --model_path "$MODEL_NAME" \
        --output_dir "$HIDDEN_DATA_DIR" \
        --mode train \
        --temperature 0.8 \
        --subjects $SUBJECTS \
        --max_new_tokens 1500 \
        --verbose
fi

if [[ $? -ne 0 ]]; then
    print_error "Data collection failed"
    exit 1
fi

print_success "Data collection completed"
echo "📁 Hidden states saved to: $HIDDEN_DATA_DIR"

# Step 3: Training Data Preparation
print_step "3" "Training Data Preparation"

TRAINING_DATA="$EXPERIMENT_DIR/training_data.json"

# Build a small training file from the hidden-state dataset so ids line up with
# core_training.HiddenStateLoader and the expected 4-turn conversation shape.
python - "$HIDDEN_DATA_DIR" "$TRAINING_DATA" "$MAX_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

from datasets import load_dataset, load_from_disk

hidden_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
max_samples = int(sys.argv[3])

if (hidden_dir / "hf_dataset").is_dir():
    ds = load_from_disk(str(hidden_dir / "hf_dataset"))
elif (hidden_dir / "dataset_info.json").exists() or (hidden_dir / "state.json").exists():
    ds = load_from_disk(str(hidden_dir))
elif (hidden_dir / "data.parquet").is_file():
    ds = load_dataset("parquet", data_files=str(hidden_dir / "data.parquet"), split="train")
elif (hidden_dir / "data_full.parquet").is_file():
    ds = load_dataset("parquet", data_files=str(hidden_dir / "data_full.parquet"), split="train")
else:
    raise SystemExit(f"Cannot find a supported hidden-state dataset under {hidden_dir}")

limit = len(ds) if max_samples < 0 else min(len(ds), max_samples)
rows = []
for row in ds.select(range(limit)):
    task_id = str(row.get("task_id") or row.get("id") or len(rows))
    task = str(row.get("task") or "Solve the task.")
    plan = str(row.get("plan") or "Use the latent plan to solve the task.")
    rows.append(
        {
            "id": task_id,
            "conversations": [
                {"from": "human", "value": "You are solving an interactive reasoning task."},
                {"from": "gpt", "value": "I will use the latent communication to help solve it."},
                {"from": "human", "value": task},
                {"from": "gpt", "value": plan},
            ],
        }
    )

if not rows:
    raise SystemExit("Hidden-state dataset is empty; cannot build training data.")

output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"Wrote {len(rows)} training examples to {output_path}")
PY

print_success "Training data prepared"

# Step 4: Model Training
print_step "4" "Model Training with Hidden State Integration"

MODEL_OUTPUT_DIR="$EXPERIMENT_DIR/trained_model"

python core_training/train.py \
    --model_name_or_path "$MODEL_NAME" \
    --data_path "$TRAINING_DATA" \
    --hidden_data "$HIDDEN_DATA_DIR" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-5 \
    --prepended_length 128 \
    --save_strategy "epoch" \
    --logging_steps 10 \
    --save_total_limit 2 \
    --dataloader_num_workers 2 \
    --report_to "none"

if [[ $? -ne 0 ]]; then
    print_error "Training failed"
    exit 1
fi

print_success "Training completed"

# Step 5: Summary
print_step "5" "Summary"

echo "========================================================"
print_success "Interlat Quick Start Completed Successfully!"
echo "========================================================"
echo "📁 Experiment directory: $EXPERIMENT_DIR"
echo "📊 Hidden states: $HIDDEN_DATA_DIR"
echo "🤖 Trained model: $MODEL_OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Evaluate your trained model on test data"
echo "2. Fine-tune hyperparameters for better performance"
echo "3. Scale up to larger datasets and models"
echo "4. Implement latent communication for multi-agent scenarios"
echo ""
echo "For more advanced usage, see:"
echo "- README.md for detailed documentation"
echo "- scripts/ directory for individual component scripts"
echo "- examples/ directory for specific use cases"
echo "========================================================"

print_success "Quick start completed! 🎉"
