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
  --loss_schedule "${LOSS_SCHEDULE:-joint}"
  --stage1_data_mode "${STAGE1_DATA_MODE:-same_data}"
  --stage1_sampling_order "${STAGE1_SAMPLING_ORDER:-sft_first}"
  --instruction_datasets "${INSTRUCTION_DATASETS:-xlsum,bactrian}"
  --xlsum_dir "${XLSUM_DIR:-data/XLSum/XLSum}"
  --bactrian_dir "${BACTRIAN_DIR:-data/Bactrian-Multilingual_Instruction}"
  --globalmgsm_dir "${GLOBALMGSM_DIR:-data/GlobalMGSM}"
  --globalmgsm_validation_ratio "${GLOBALMGSM_VALIDATION_RATIO:-0.01}"
  --bactrian_validation_ratio "${BACTRIAN_VALIDATION_RATIO:-0.01}"
  --languages "${LANGUAGES:-en,km,my,th,vi}"
  --max_length "${MAX_LENGTH:-1024}"
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --data_dir "$DATA_DIR"
  --language_pairs "${LANGUAGE_PAIRS:-all}"
  --output_dir "$OUTPUT_DIR"
  --direction "$DIRECTION"
  --prompt_format "${PROMPT_FORMAT:-plain}"
  --training_mode "${TRAINING_MODE:-finetune}"
  --align_layer "${ALIGN_LAYER:--1}"
  --contrastive_forward_mode "${CONTRASTIVE_FORWARD_MODE:-joint}"
  --ot_forward_mode "${OT_FORWARD_MODE:-joint}"
  --contrastive_weight "${CONTRASTIVE_WEIGHT:-0.0}"
  --temperature "${CONTRASTIVE_TEMPERATURE:-0.07}"
  --ot_weight "${OT_WEIGHT:-0.0}"
  --structure_weight "${STRUCTURE_WEIGHT:-0.0}"
  --structure_temperature "${STRUCTURE_TEMPERATURE:-0.1}"
  --attention_mass_weight "${ATTENTION_MASS_WEIGHT:-0.5}"
  --ot_solver "${OT_SOLVER:-sinkhorn}"
  --ot_distance "${OT_DISTANCE:-cosine}"
  --ot_covariance_regularization "${OT_COVARIANCE_REGULARIZATION:-0.0001}"
  --ot_whitening_method "${OT_WHITENING_METHOD:-truncated}"
  --ot_eigenvalue_threshold "${OT_EIGENVALUE_THRESHOLD:-0.00001}"
  --ot_calibration_samples_per_language "${OT_CALIBRATION_SAMPLES_PER_LANGUAGE:-1024}"
  --ot_calibration_batch_size "${OT_CALIBRATION_BATCH_SIZE:-8}"
  --ot_calibration_max_length "${OT_CALIBRATION_MAX_LENGTH:-256}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --epochs "${EPOCHS:-3}"
  --batch_size "${BATCH_SIZE:-2}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --sinkhorn_iterations "${SINKHORN_ITERATIONS:-20}"
  --sinkhorn_epsilon "${SINKHORN_EPSILON:-0.1}"
  --sinkhorn_epsilon_schedule "${SINKHORN_EPSILON_SCHEDULE:-constant}"
  --sinkhorn_epsilon_end "${SINKHORN_EPSILON_END:-0.01}"
  --ipot_beta "${IPOT_BETA:-0.5}"
  --ipot_iterations "${IPOT_ITERATIONS:-50}"
  --ipot_inner_iterations "${IPOT_INNER_ITERATIONS:-1}"
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

if [[ "${STRUCTURE_WEIGHT:-0.0}" != "0" && "${STRUCTURE_WEIGHT:-0.0}" != "0.0" ]]; then
  if [[ -z "${STRUCTURE_REFERENCE_MODEL_NAME_OR_PATH:-}" ]]; then
    echo "STRUCTURE_REFERENCE_MODEL_NAME_OR_PATH is required when STRUCTURE_WEIGHT is non-zero" >&2
    exit 2
  fi
  ARGS+=(
    --structure_reference_model_name_or_path
    "$STRUCTURE_REFERENCE_MODEL_NAME_OR_PATH"
  )
fi

if [[ "${ENABLE_THINKING:-false}" == "true" ]]; then
  ARGS+=(--enable_thinking)
else
  ARGS+=(--no-enable_thinking)
fi

if [[ "${BATCH_BY_LANGUAGE_PAIR:-false}" == "true" ]]; then
  ARGS+=(--batch_by_language_pair)
fi
if [[ -n "${STAGE1_SAMPLES_PER_STREAM:-}" ]]; then
  ARGS+=(--stage1_samples_per_stream "$STAGE1_SAMPLES_PER_STREAM")
fi
if [[ "${STAGE1_BALANCE_PAIRS:-true}" == "true" ]]; then
  ARGS+=(--stage1_balance_pairs)
else
  ARGS+=(--no-stage1_balance_pairs)
fi

if [[ -n "${OT_COVARIANCE_PATH:-}" ]]; then
  ARGS+=(--ot_covariance_path "$OT_COVARIANCE_PATH")
fi
if [[ "${OT_COVARIANCE_AUTO:-true}" == "true" ]]; then
  ARGS+=(--ot_covariance_auto)
else
  ARGS+=(--no-ot_covariance_auto)
fi
if [[ "${OT_STOP_GRADIENT_ENGLISH:-false}" == "true" ]]; then
  ARGS+=(--ot_stop_gradient_english)
fi
if [[ "${CONTRASTIVE_STOP_GRADIENT_ENGLISH:-false}" == "true" ]]; then
  ARGS+=(--contrastive_stop_gradient_english)
fi
if [[ -n "${OT_COVARIANCE_CACHE:-}" ]]; then
  ARGS+=(--ot_covariance_cache "$OT_COVARIANCE_CACHE")
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
