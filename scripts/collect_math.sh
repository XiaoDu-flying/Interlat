#!/bin/bash
# Math Data Collection Script
# This script collects hidden states from math reasoning tasks for latent communication training

set -e  # Exit on error

# Default configurations
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/math_hidden_states}"
MODE="${MODE:-train}"
TEMPERATURE="${TEMPERATURE:-0.8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1500}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
SUBJECTS="${SUBJECTS:-algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
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
        --subjects)
            shift
            SUBJECTS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SUBJECTS="$SUBJECTS $1"
                shift
            done
            SUBJECTS=$(echo "$SUBJECTS" | xargs)  # Trim whitespace
            ;;
        --prompt_file)
            PROMPT_FILE="$2"
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
            echo "Math Data Collection Script"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_path <path>       Model path or HuggingFace model name"
            echo "  --output_dir <path>       Output directory for collected data"
            echo "  --mode <train|test>       Dataset split to use (default: train)"
            echo "  --temperature <float>     Sampling temperature (default: 0.8)"
            echo "  --max_new_tokens <int>    Maximum new tokens to generate (default: 1500)"
            echo "  --torch_dtype <type>      PyTorch dtype: float32/float16/bfloat16"
            echo "  --subjects <list>         Math subjects (space-separated)"
            echo "                           Available: algebra counting_and_probability geometry"
            echo "                                     intermediate_algebra number_theory prealgebra precalculus"
            echo "  --prompt_file <path>      Custom prompt template file"
            echo "  --distributed             Enable distributed training"
            echo "  --num_gpus <int>         Number of GPUs for distributed training"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --mode train --temperature 0.8"
            echo "  $0 --subjects algebra geometry --temperature 0.6"
            echo "  $0 --distributed --num_gpus 4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
    echo "Error: Mode must be 'train' or 'test', got: $MODE"
    exit 1
fi

# Print configuration
echo "========================================================"
echo "Math Hidden State Collection"
echo "========================================================"
echo "Model: $MODEL_PATH"
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "Temperature: $TEMPERATURE"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "PyTorch dtype: $TORCH_DTYPE"
echo "Subjects: $SUBJECTS"
if [[ -n "$PROMPT_FILE" ]]; then
    echo "Custom prompt: $PROMPT_FILE"
fi
echo "========================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")/logs"

# Set up logging
LOG_FILE="$(dirname "$OUTPUT_DIR")/logs/math_collection_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Math data collection..."
echo "Logs will be saved to: $LOG_FILE"

# Build command arguments
CMD_ARGS=(
    --model_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --mode "$MODE"
    --temperature "$TEMPERATURE"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --torch_dtype "$TORCH_DTYPE"
    --verbose
)

# Add subjects
if [[ -n "$SUBJECTS" ]]; then
    CMD_ARGS+=(--subjects)
    for subject in $SUBJECTS; do
        CMD_ARGS+=("$subject")
    done
fi

# Add custom prompt if provided
if [[ -n "$PROMPT_FILE" ]]; then
    if [[ ! -f "$PROMPT_FILE" ]]; then
        echo "Error: Prompt file not found: $PROMPT_FILE"
        exit 1
    fi
    CMD_ARGS+=(--prompt_file "$PROMPT_FILE")
fi

# Determine whether to use distributed training
if [[ "$DISTRIBUTED" == "true" ]]; then
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Auto-detected $NUM_GPUS GPUs for distributed training"
    fi

    echo "Running distributed data collection on $NUM_GPUS GPUs..."

    # Run with torchrun for distributed training
    torchrun --nproc_per_node="$NUM_GPUS" \
        --master_port=12346 \
        data_collection/math_collection.py \
        "${CMD_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
else
    echo "Running single-GPU data collection..."

    # Run single process
    python data_collection/math_collection.py \
        "${CMD_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
fi

# Check if collection was successful
if [[ $? -eq 0 ]]; then
    echo "✅ Math data collection completed successfully!"
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
    echo "❌ Math data collection failed. Check the logs for details."
    echo "📊 Logs: $LOG_FILE"
    exit 1
fi
