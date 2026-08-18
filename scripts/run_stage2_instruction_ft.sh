#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-outputs/stage1-multilingual-alignment}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2-instruction-ft}"
XLSUM_DIR="${XLSUM_DIR:-data/XLSum/XLSum}"
BACTRIAN_DIR="${BACTRIAN_DIR:-data/Bactrian-Multilingual_Instruction}"
LANGUAGES="${LANGUAGES:-en,km,my,th,vi}"
PRECISION="${PRECISION:-bf16}"

# Stage 2 uses response-only next-token prediction. It does not instantiate the
# alignment wrapper and does not pass contrastive or OT arguments.
ARGS=(
  -m src.train
  --stage instruction
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --xlsum_dir "$XLSUM_DIR"
  --bactrian_dir "$BACTRIAN_DIR"
  --languages "$LANGUAGES"
  --output_dir "$OUTPUT_DIR"
  --prompt_format "${PROMPT_FORMAT:-plain}"
  --learning_rate "${LEARNING_RATE:-1e-5}"
  --epochs "${EPOCHS:-2}"
  --batch_size "${BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
  --max_length "${MAX_LENGTH:-1024}"
  --bactrian_validation_ratio "${BACTRIAN_VALIDATION_RATIO:-0.01}"
  --seed "${SEED:-42}"
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

if [[ "${ENABLE_THINKING:-false}" == "true" ]]; then
  ARGS+=(--enable_thinking)
else
  ARGS+=(--no-enable_thinking)
fi

if [[ "$PRECISION" == "bf16" ]]; then
  ARGS+=(--bf16)
elif [[ "$PRECISION" == "fp16" ]]; then
  ARGS+=(--fp16)
elif [[ "$PRECISION" != "fp32" ]]; then
  echo "PRECISION must be bf16, fp16, or fp32" >&2
  exit 2
fi

echo "Stage 2 loss  : response-only next-token prediction"
echo "Stage 2 model : $MODEL_NAME_OR_PATH"
echo "Stage 2 output: $OUTPUT_DIR"
python "${ARGS[@]}"
