#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B}"
DATA_DIR="${DATA_DIR:-data/MT}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-0.5b-multilingual-aligned}"
DIRECTION="${DIRECTION:-both}"
PRECISION="${PRECISION:-bf16}"

ARGS=(
  -m src.train
  --stage alignment
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --data_dir "$DATA_DIR"
  --language_pairs "${LANGUAGE_PAIRS:-all}"
  --output_dir "$OUTPUT_DIR"
  --direction "$DIRECTION"
  --align_layer "${ALIGN_LAYER:--1}"
  --contrastive_weight "${CONTRASTIVE_WEIGHT:-0.0}"
  --temperature "${CONTRASTIVE_TEMPERATURE:-0.07}"
  --ot_weight "${OT_WEIGHT:-0.0}"
  --attention_mass_weight "${ATTENTION_MASS_WEIGHT:-0.5}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --epochs "${EPOCHS:-3}"
  --batch_size "${BATCH_SIZE:-2}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --sinkhorn_iterations "${SINKHORN_ITERATIONS:-20}"
  --sinkhorn_epsilon "${SINKHORN_EPSILON:-0.1}"
  --seed "${SEED:-42}"
  --attn_implementation eager
  --gradient_checkpointing
  --use_lora
  --lora_r "${LORA_R:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
  --logging_steps "${LOGGING_STEPS:-10}"
  --save_steps "${SAVE_STEPS:-500}"
  --eval_steps "${EVAL_STEPS:-500}"
  --save_strategy "${SAVE_STRATEGY:-epoch}"
  --eval_strategy "${EVAL_STRATEGY:-epoch}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --warmup_steps "${WARMUP_STEPS:-0}"
  --report_to "${REPORT_TO:-tensorboard}"
)

if [[ "$PRECISION" == "bf16" ]]; then
  ARGS+=(--bf16)
elif [[ "$PRECISION" == "fp16" ]]; then
  ARGS+=(--fp16)
elif [[ "$PRECISION" != "fp32" ]]; then
  echo "PRECISION must be bf16, fp16, or fp32" >&2
  exit 2
fi

python "${ARGS[@]}"
