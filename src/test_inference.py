"""Run translation inference on MT test splits and export src/ref/pred CSV."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from tqdm.auto import tqdm

from .collator import TranslationInferenceCollator
from .prepare_data import (
    _discover_mt,
    _parallel_rows,
    prepare_translation_inference_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_dir", default="data/MT")
    parser.add_argument("--language_pairs", default="all")
    parser.add_argument("--direction", choices=["forward", "reverse", "both"], default="forward")
    parser.add_argument("--output_dir", default="outputs/test_predictions")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        "--attn_implementation", choices=["eager", "sdpa"], default="sdpa"
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--prompt_format", choices=["plain", "chat"], default="plain")
    parser.add_argument(
        "--enable_thinking", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def limit_per_direction(iterator, limit: int | None):
    if limit is None:
        yield from iterator
        return
    seen = defaultdict(int)
    for row in iterator:
        direction = (row["source_lang"], row["target_lang"])
        if seen[direction] < limit:
            seen[direction] += 1
            yield row


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


@torch.inference_mode()
def generate_batch(model, input_ids, attention_mask, tokenizer, args):
    """Generate one already-tokenized and already-collated inference batch."""
    generation_kwargs = {}
    if args.enable_thinking:
        generation_kwargs.update(
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20
        )
    else:
        generation_kwargs["do_sample"] = False
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        **generation_kwargs,
    )


def main() -> None:
    total_started = time.perf_counter()
    timings = {}
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    phase_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.prompt_format == "chat" and not tokenizer.chat_template:
        raise ValueError("--prompt_format chat requires tokenizer.chat_template")
    timings["tokenizer_load"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if dtype == torch.float16 and (
        args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available())
    ):
        print("WARNING: float16 on CPU is unsuitable; falling back to float32.")
        dtype = torch.float32
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
    }
    if args.device == "auto":
        load_kwargs["device_map"] = "auto"
    adapter_config_path = Path(args.model_name_or_path) / "adapter_config.json"
    if adapter_config_path.exists():
        peft_config = PeftConfig.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, **load_kwargs
        )
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, **load_kwargs
        )
    if args.device in {"cuda", "cpu"}:
        model.to(args.device)
    # Training enables gradient checkpointing and sets use_cache=False. If that
    # configuration is saved in the checkpoint, autoregressive decoding becomes
    # extremely slow because every new token recomputes the full prefix.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
        if not args.enable_thinking:
            # Avoid warnings from sampling values stored in generation_config
            # while greedy decoding is requested.
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
    model.eval()
    device = model.get_input_embeddings().weight.device
    device_map = getattr(model, "hf_device_map", None)
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
    print(f"Requested mode : {args.device}")
    print(f"Input device   : {device}")
    print(f"Model dtype    : {next(model.parameters()).dtype}")
    print(f"Attention impl : {args.attn_implementation}")
    print(f"KV cache       : {model.config.use_cache}")
    print(f"HF device map  : {device_map if device_map is not None else 'single device'}")
    if device_map and any(str(value) in {"cpu", "disk"} for value in device_map.values()):
        print("WARNING: Some model layers are offloaded to CPU/disk; generation will be slow.")
    timings["model_load"] = time.perf_counter() - phase_started

    paths = _discover_mt(args.data_dir, "test", args.language_pairs)
    inference_collator = TranslationInferenceCollator(tokenizer=tokenizer)
    # Materialize the selected test set first, then tokenize every prompt before
    # generation. This keeps tokenization time separate from generation time.
    print("Loading test samples...", flush=True)
    phase_started = time.perf_counter()
    rows = list(limit_per_direction(
        _parallel_rows(paths, args.direction), args.max_samples
    ))
    timings["data_load"] = time.perf_counter() - phase_started
    total_rows = len(rows)
    print(f"Test samples   : {total_rows}", flush=True)
    if total_rows == 0:
        raise ValueError("No test samples matched the requested language pairs/direction")
    phase_started = time.perf_counter()
    tokenized_rows = prepare_translation_inference_dataset(
        rows=rows,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        prompt_format=args.prompt_format,
        enable_thinking=args.enable_thinking,
    )
    timings["tokenization"] = time.perf_counter() - phase_started
    # DataLoader forms fixed-size batches. The inference collator left-pads each
    # batch independently to that batch's longest prompt. Materialize all
    # collated batches now so no tokenization/collation happens during generate.
    data_loader = DataLoader(
        tokenized_rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=inference_collator,
    )
    print("Collating inference batches...", flush=True)
    phase_started = time.perf_counter()
    collated_batches = list(tqdm(
        data_loader,
        total=len(data_loader),
        desc="Collating batches",
        unit="batch",
        dynamic_ncols=True,
        file=sys.stdout,
    ))
    timings["collation"] = time.perf_counter() - phase_started
    print(f"Collated batches: {len(collated_batches)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    predictions_by_direction = defaultdict(list)
    generated_token_count = 0
    reached_eos_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    phase_started = time.perf_counter()
    with tqdm(
        total=total_rows,
        desc="Generating translations",
        unit="sample",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as progress:
        for batch in collated_batches:
            # Tokenization, batching and padding are already complete. The
            # generation loop only transfers a prepared batch to the model.
            row_batch = batch["rows"]
            prompt_width = batch["prompt_width"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = generate_batch(
                model, input_ids, attention_mask, tokenizer, args
            )
            new_token_ids = generated[:, prompt_width:]
            predictions = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
            for token_row in new_token_ids:
                eos_positions = (
                    (token_row == tokenizer.eos_token_id).nonzero(as_tuple=False)
                    if tokenizer.eos_token_id is not None else []
                )
                if len(eos_positions):
                    generated_token_count += int(eos_positions[0].item()) + 1
                    reached_eos_count += 1
                else:
                    generated_token_count += token_row.numel()
            for row, prediction in zip(row_batch, predictions):
                direction = f"{row['source_lang']}-{row['target_lang']}"
                predictions_by_direction[direction].append({
                    "src": row["source"],
                    "ref": row["target"],
                    "pred": prediction.strip(),
                })
                counts[direction] += 1
            progress.update(len(row_batch))
            progress.set_postfix_str(
                ", ".join(
                    f"{key}={value}" for key, value in sorted(counts.items())
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["generation"] = time.perf_counter() - phase_started

    print("Inference complete. Writing CSV files...", flush=True)
    phase_started = time.perf_counter()
    for direction, result_rows in tqdm(
        sorted(predictions_by_direction.items()),
        desc="Writing CSV files",
        unit="file",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        path = output_dir / f"{direction}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["src", "ref", "pred"])
            writer.writeheader()
            writer.writerows(result_rows)
        print(f"Saved {len(result_rows)} rows to {path.resolve()}")
    timings["csv_write"] = time.perf_counter() - phase_started
    timings["total"] = time.perf_counter() - total_started

    print("\nTiming summary")
    measured_total = timings["total"]
    for name in (
        "tokenizer_load", "model_load", "data_load", "tokenization",
        "collation", "generation", "csv_write", "total",
    ):
        seconds = timings[name]
        percentage = 100.0 * seconds / measured_total
        print(f"  {name:16s} {format_duration(seconds)}  ({percentage:6.2f}%)")
    if total_rows:
        print(
            f"  tokenize_speed   {total_rows / timings['tokenization']:.2f} sample/s"
        )
        print(
            f"  generation_speed {total_rows / timings['generation']:.2f} sample/s"
        )
        print(f"  csv_write_speed  {total_rows / timings['csv_write']:.2f} sample/s")
        print(f"  avg_new_tokens   {generated_token_count / total_rows:.2f} token/sample")
        print(f"  reached_eos      {reached_eos_count}/{total_rows}")


if __name__ == "__main__":
    main()
