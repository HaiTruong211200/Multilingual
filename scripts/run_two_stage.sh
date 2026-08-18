#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_OUTPUT="${STAGE1_OUTPUT:-outputs/stage1-multilingual-alignment}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-outputs/stage2-instruction-ft}"

echo "========== Stage 1: NTP with optional Contrastive/OT =========="
MODEL_NAME_OR_PATH="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}" \
OUTPUT_DIR="$STAGE1_OUTPUT" \
LANGUAGE_PAIRS="${TRANSLATION_PAIRS:-all}" \
PRECISION="${PRECISION:-bf16}" \
PROMPT_FORMAT="${STAGE1_PROMPT_FORMAT:-plain}" \
ENABLE_THINKING="${STAGE1_ENABLE_THINKING:-false}" \
"$SCRIPT_DIR/run_stage1_alignment.sh"

echo "========== Stage 2: response-only NTP =========="
MODEL_NAME_OR_PATH="$STAGE1_OUTPUT" \
OUTPUT_DIR="$STAGE2_OUTPUT" \
LANGUAGES="${INSTRUCTION_LANGUAGES:-en,km,my,th,vi}" \
PRECISION="${PRECISION:-bf16}" \
PROMPT_FORMAT="${STAGE2_PROMPT_FORMAT:-plain}" \
ENABLE_THINKING="${STAGE2_ENABLE_THINKING:-false}" \
"$SCRIPT_DIR/run_stage2_instruction_ft.sh"
