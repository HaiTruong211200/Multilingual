#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARGS=(
  -m src.test_inference
  --model_name_or_path "${MODEL_NAME_OR_PATH:-outputs/stage2-instruction-ft}"
  --data_dir "${DATA_DIR:-data/MT}"
  --language_pairs "${LANGUAGE_PAIRS:-all}"
  --direction "${DIRECTION:-forward}"
  --output_dir "${OUTPUT_DIR:-outputs/test_predictions}"
  --batch_size "${BATCH_SIZE:-4}"
  --max_input_length "${MAX_INPUT_LENGTH:-512}"
  --max_new_tokens "${MAX_NEW_TOKENS:-256}"
  --num_beams "${NUM_BEAMS:-1}"
  --device "${DEVICE:-auto}"
  --dtype "${DTYPE:-auto}"
  --prompt_format "${PROMPT_FORMAT:-plain}"
)
if [[ "${ENABLE_THINKING:-false}" == "true" ]]; then
  ARGS+=(--enable_thinking)
else
  ARGS+=(--no-enable_thinking)
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi
python "${ARGS[@]}"
