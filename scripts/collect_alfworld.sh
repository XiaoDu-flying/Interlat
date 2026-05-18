#!/bin/bash
# ALFWorld Data Collection Script
# This script collects hidden states from ALFWorld tasks for latent communication training

set -e  # Exit on error

# Default configurations
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATASET_PATH="${DATASET_PATH:-./datasets/alfworld_dataset.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/alfworld_hidden_states}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1500}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --torch_dtype)
            TORCH_DTYPE="$2"
            shift 2
            ;;
        --distributed)
            DISTRIBUTED="true"
            shift
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help|-h)
            echo "ALFWorld Data Collection Script"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_path <path>       Model path or HuggingFace model name"
            echo "  --dataset_path <path>     Path to ALFWorld dataset JSON file"
            echo "  --output_dir <path>       Output directory for collected data"
            echo "  --temperature <float>     Sampling temperature (default: 0.7)"
            echo "  --max_new_tokens <int>    Maximum new tokens to generate (default: 1500)"
            echo "  --torch_dtype <type>      PyTorch dtype: float32/float16/bfloat16"
            echo "  --distributed             Enable distributed training"
            echo "  --num_gpus <int>         Number of GPUs for distributed training"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================================"
echo "ALFWorld Hidden State Collection"
echo "========================================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "PyTorch dtype: $TORCH_DTYPE"
echo "========================================================"

# Validate dataset path
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    echo "Please ensure the dataset file exists or provide the correct path with --dataset_path"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")/logs"

# Set up logging
LOG_FILE="$(dirname "$OUTPUT_DIR")/logs/alfworld_collection_$(date +%Y%m%d_%H%M%S).log"

echo "Starting ALFWorld data collection..."
echo "Logs will be saved to: $LOG_FILE"

# Determine whether to use distributed training
if [[ "$DISTRIBUTED" == "true" ]]; then
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Auto-detected $NUM_GPUS GPUs for distributed training"
    fi

    echo "Running distributed data collection on $NUM_GPUS GPUs..."

    # Run with torchrun for distributed training
    torchrun --nproc_per_node="$NUM_GPUS" \
        --master_port=12345 \
        data_collection/alfworld_collection.py \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --torch_dtype "$TORCH_DTYPE" \
        --verbose \
        2>&1 | tee "$LOG_FILE"
else
    echo "Running single-GPU data collection..."

    # Run single process
    python data_collection/alfworld_collection.py \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --torch_dtype "$TORCH_DTYPE" \
        --verbose \
        2>&1 | tee "$LOG_FILE"
fi

# Check if collection was successful
if [[ $? -eq 0 ]]; then
    echo "✅ ALFWorld data collection completed successfully!"
    echo "📁 Output saved to: $OUTPUT_DIR"
    echo "📊 Logs saved to: $LOG_FILE"

    # Show output structure
    echo ""
    echo "Output structure:"
    find "$OUTPUT_DIR" -type f | head -10 | sed 's/^/  /'
    if [[ $(find "$OUTPUT_DIR" -type f | wc -l) -gt 10 ]]; then
        echo "  ... and $(( $(find "$OUTPUT_DIR" -type f | wc -l) - 10 )) more files"
    fi
else
    echo "❌ ALFWorld data collection failed. Check the logs for details."
    echo "📊 Logs: $LOG_FILE"
    exit 1
fi
