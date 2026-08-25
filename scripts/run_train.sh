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
  --prompt_format "${PROMPT_FORMAT:-plain}"
  --training_mode "${TRAINING_MODE:-finetune}"
  --attention_mass_weight "${ATTENTION_MASS_WEIGHT:-0.5}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --epochs "${EPOCHS:-3}"
  --batch_size "${BATCH_SIZE:-2}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --sinkhorn_iterations "${SINKHORN_ITERATIONS:-20}"
  --sinkhorn_epsilon "${SINKHORN_EPSILON:-0.1}"
  --seed "${SEED:-42}"
  --data_seed "${DATA_SEED:-${SEED:-42}}"
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

if [[ -n "${CONFIG_FILE:-}" ]]; then
  ARGS+=(--config "$CONFIG_FILE")
fi

if [[ -n "${ALIGN_LAYER:-}" ]]; then
  ARGS+=(--align_layer "$ALIGN_LAYER")
fi

if [[ -n "${CONTRASTIVE_WEIGHT:-}" ]]; then
  ARGS+=(--contrastive_weight "$CONTRASTIVE_WEIGHT")
fi
if [[ -n "${CONTRASTIVE_TEMPERATURE:-}" ]]; then
  ARGS+=(--temperature "$CONTRASTIVE_TEMPERATURE")
fi
if [[ -n "${OT_WEIGHT:-}" ]]; then
  ARGS+=(--ot_weight "$OT_WEIGHT")
fi

if [[ -n "${CANDIDATE_LAYERS:-}" ]]; then
  ARGS+=(
    --candidate_layers "$CANDIDATE_LAYERS"
    --reward_ema_rho "${REWARD_EMA_RHO:-0.1}"
    --ucb_beta "${UCB_BETA:-0.5}"
    --layer_temperature "${LAYER_TEMPERATURE:-1.0}"
    --layer_warmup_steps "${LAYER_WARMUP_STEPS:-100}"
  )
  if [[ "${FORCE_EACH_LAYER_ONCE:-true}" == "true" ]]; then
    ARGS+=(--force_each_layer_once)
  else
    ARGS+=(--no-force_each_layer_once)
  fi
fi

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

python "${ARGS[@]}"
