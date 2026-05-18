#!/bin/bash
# Model Training Script for Interlat
# This script trains language models with hidden state integration for latent communication

set -e  # Exit on error

# Default configurations
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
DATA_PATH="${DATA_PATH:-./data/training_data.json}"
HIDDEN_DATA="${HIDDEN_DATA:-./data/hidden_states}"
OUTPUT_DIR="${OUTPUT_DIR:-./trained_models}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PREPENDED_LENGTH="${PREPENDED_LENGTH:-1000}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --hidden-data)
            HIDDEN_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --prepended-length)
            PREPENDED_LENGTH="$2"
            shift 2
            ;;
        --deepspeed)
            DEEPSPEED_CONFIG="$2"
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
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help|-h)
            echo "Interlat Model Training Script"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model <name>            Model name or path (default: Qwen/Qwen2.5-7B-Instruct)"
            echo "  --data <path>             Training data JSON path"
            echo "  --hidden-data <path>      Hidden states data directory"
            echo "  --output-dir <path>       Output directory for trained model"
            echo "  --epochs <int>            Number of training epochs (default: 10)"
            echo "  --batch-size <int>        Training batch size (default: 4)"
            echo "  --learning-rate <float>   Learning rate (default: 5e-5)"
            echo "  --prepended-length <int>  Hidden state sequence length (default: 1000)"
            echo "  --deepspeed <config>      DeepSpeed config file path"
            echo "  --distributed             Enable distributed training"
            echo "  --num_gpus <int>         Number of GPUs for distributed training"
            echo "  --resume <path>          Resume training from checkpoint"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model Qwen/Qwen2.5-7B --data ./data/train.json --epochs 5"
            echo "  $0 --distributed --num_gpus 4 --deepspeed ds_config.json"
            echo "  $0 --resume ./checkpoints/checkpoint-1000"
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
echo "Interlat Model Training"
echo "========================================================"
echo "Model: $MODEL_NAME"
echo "Training data: $DATA_PATH"
echo "Hidden states: $HIDDEN_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Prepended length: $PREPENDED_LENGTH"
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
    echo "DeepSpeed config: $DEEPSPEED_CONFIG"
fi
if [[ -n "$RESUME_FROM" ]]; then
    echo "Resume from: $RESUME_FROM"
fi
echo "========================================================"

# Validate data paths
if [[ ! -f "$DATA_PATH" ]]; then
    echo "Error: Training data file not found: $DATA_PATH"
    echo "Please ensure the training data file exists"
    exit 1
fi

if [[ ! -d "$HIDDEN_DATA" ]]; then
    echo "Error: Hidden states directory not found: $HIDDEN_DATA"
    echo "Please run data collection first to generate hidden states"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")/logs"

# Set up logging
LOG_FILE="$(dirname "$OUTPUT_DIR")/logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting model training..."
echo "Logs will be saved to: $LOG_FILE"

# Build command arguments
CMD_ARGS=(
    --model_name_or_path "$MODEL_NAME"
    --data_path "$DATA_PATH"
    --hidden_data "$HIDDEN_DATA"
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --prepended_length "$PREPENDED_LENGTH"
    --save_strategy "epoch"
    --evaluation_strategy "epoch"
    --logging_steps 100
    --save_total_limit 3
    --load_best_model_at_end true
    --metric_for_best_model "eval_loss"
    --greater_is_better false
    --report_to "none"  # Disable wandb/tensorboard by default
    --dataloader_num_workers 4
    --dataloader_pin_memory true
    --gradient_checkpointing true
)

# Add DeepSpeed config if provided
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
    if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
        echo "Error: DeepSpeed config file not found: $DEEPSPEED_CONFIG"
        exit 1
    fi
    CMD_ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")
fi

# Add resume checkpoint if provided
if [[ -n "$RESUME_FROM" ]]; then
    if [[ ! -d "$RESUME_FROM" ]]; then
        echo "Error: Resume checkpoint directory not found: $RESUME_FROM"
        exit 1
    fi
    CMD_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
fi

# Determine whether to use distributed training
if [[ "$DISTRIBUTED" == "true" ]]; then
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Auto-detected $NUM_GPUS GPUs for distributed training"
    fi

    echo "Running distributed training on $NUM_GPUS GPUs..."

    # Use torchrun for distributed training
    if [[ -n "$DEEPSPEED_CONFIG" ]]; then
        # DeepSpeed training
        deepspeed --num_gpus="$NUM_GPUS" \
            core_training/train.py \
            "${CMD_ARGS[@]}" \
            2>&1 | tee "$LOG_FILE"
    else
        # Standard distributed training
        torchrun --nproc_per_node="$NUM_GPUS" \
            --master_port=12347 \
            core_training/train.py \
            "${CMD_ARGS[@]}" \
            2>&1 | tee "$LOG_FILE"
    fi
else
    echo "Running single-GPU training..."

    # Single process training
    python core_training/train.py \
        "${CMD_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
fi

# Check if training was successful
if [[ $? -eq 0 ]]; then
    echo "✅ Model training completed successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo "📊 Logs saved to: $LOG_FILE"

    # Show output structure
    echo ""
    echo "Trained model structure:"
    find "$OUTPUT_DIR" -name "pytorch_model*.bin" -o -name "config.json" -o -name "tokenizer*" | head -10 | sed 's/^/  /'

    echo ""
    echo "🚀 You can now use the trained model for inference or further training!"
else
    echo "❌ Model training failed. Check the logs for details."
    echo "📊 Logs: $LOG_FILE"
    exit 1
fi
