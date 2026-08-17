# Multilingual alignment fine-tuning

The objective is:

`L = L_NTP + contrastive_weight * L_InfoNCE + ot_weight * L_Sinkhorn`

By default all `data/MT/*/train.*.json` and `data/MT/*/valid.*.json` files are loaded.
The repository's JSONL schema is supported directly:

```json
{"src_lang":"vi","tgt_lang":"en","translation":{"vi":"Xin chào","en":"Hello"}}
```

Install and run from the repository root:

```bash
pip install -r src/requirements.txt
python -m src.train --stage alignment \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --data_dir data/MT \
  --output_dir outputs/qwen-multilingual \
  --direction both \
  --bf16 --gradient_checkpointing
```

`--direction forward` trains e.g. Vietnamese -> English, `reverse` trains English
-> Vietnamese, and `both` adds both examples. A batch size of at least 2 is
recommended because in-batch negatives are used by InfoNCE.

OT is quadratic in `alignment_max_length`; keep it much smaller than `max_length`.
The implementation uses the causal LM's shared hidden space and excludes padding
tokens from mean pooling and transport marginals.

The collator records `source_start_positions`, `source_end_positions`,
`target_start_positions`, and `target_end_positions` in the same causal sequence.
The model performs only one forward pass and slices both spans from its hidden
states. Contrastive loss uses the selected alignment layer; OT uses the final layer.

Contrastive loss follows the supplied within-instruction Llama implementation:
mean pooling over each span, an in-batch cosine-similarity matrix, `LogSoftmax`
over dimension 0, and diagonal positive-pair NLL scaled by 1/2. Use
`--align_layer -1` (default) or another hidden-state index to select its layer.

OT uses token-pair cosine distance as its transport cost. Its source and target
marginals come from the attention at the alignment layer: attention is averaged
over heads, accumulated over target queries, restricted to each span, and
normalized so each marginal sums to one. `--attention_mass_weight` mixes this
distribution with uniform mass: `0` is uniform-only, `1` is attention-only, and
the default `0.5` gives equal weight to both.

## Two-stage training

Stage 1 trains translation alignment. Stage 2 reloads that exported checkpoint and
performs response-only instruction tuning on XLSum (`text -> summary`) and
Bactrian (`instruction + input -> output`):

Both stages target base causal LMs and use plain-text prompts from `prompts.py`;
no tokenizer chat template or user/assistant special tokens are applied.
Both `.sh` stage scripts use LoRA by default (`r=16`, `alpha=32`, dropout `0.05`)
on `q_proj,k_proj,v_proj,o_proj`. The adapter is merged when a stage finishes so
the exported checkpoint is a normal Hugging Face model that the next stage can
load directly.

Training logs the selected stage, data sources/languages, processed dataset sizes
and a truncated processed sample, LoRA/trainable parameter statistics, and every
loss component. Stage 1 reports `trainer_loss` (the authoritative value returned
by Trainer), `model_total_loss`, and raw/weighted NTP/contrastive/OT losses so the
two totals can be compared. Stage 2 reports response-only NTP. TensorBoard events
are written to `<output>/runs`.

Linux/server scripts:

```bash
chmod +x scripts/*.sh

# Stage 1: NTP + Contrastive + OT
./scripts/run_stage1_alignment.sh

# Stage 2: response-only NTP
MODEL_NAME_OR_PATH=outputs/stage1-multilingual-alignment \
  ./scripts/run_stage2_instruction_ft.sh

# Or run both sequentially
./scripts/run_two_stage.sh
```

Language selection is independent across stages:

```bash
# Available translation directories include km-en, lo-en, my-en, th-en,
# vi-en, and zh-en. Use "all" to load every pair.
LANGUAGE_PAIRS=vi-en,th-en ./scripts/run_stage1_alignment.sh

# Load only the selected XLSum/Bactrian languages.
LANGUAGES=vi,th,en \
MODEL_NAME_OR_PATH=outputs/stage1-multilingual-alignment \
./scripts/run_stage2_instruction_ft.sh

# Both stages with separate selections.
TRANSLATION_PAIRS=vi-en,th-en \
INSTRUCTION_LANGUAGES=vi,th,en \
./scripts/run_two_stage.sh
```

To run only stage 2 from another checkpoint, set `MODEL_NAME_OR_PATH` before
calling `run_stage2_instruction_ft.sh`.

Test translation inference. Each translation direction is exported to its own
UTF-8 CSV (`vi-en.csv`, `en-vi.csv`, etc.) with exactly `src,ref,pred`:

```bash
MODEL_NAME_OR_PATH=outputs/stage2-instruction-ft \
LANGUAGE_PAIRS=vi-en,th-en \
DIRECTION=forward \
OUTPUT_DIR=outputs/test_predictions \
./scripts/run_test_inference.sh
```

For a quick subset, set `MAX_SAMPLES=100` to keep up to 100 rows per translation
direction. The notebook contains a separate
three-step LoRA mini-training cell for checking backward and component logging.
