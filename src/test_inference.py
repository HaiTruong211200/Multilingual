"""Run translation inference on MT test splits and export src/ref/pred CSV."""
from __future__ import annotations

import argparse
import csv
import sys
from contextlib import ExitStack
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from .collator import TranslationInferenceCollator
from .prepare_data import _discover_mt, _parallel_rows


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
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
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


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.prompt_format == "chat" and not tokenizer.chat_template:
        raise ValueError("--prompt_format chat requires tokenizer.chat_template")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, **load_kwargs
    )
    if args.device in {"cuda", "cpu"}:
        model.to(args.device)
    model.eval()
    device = model.get_input_embeddings().weight.device
    device_map = getattr(model, "hf_device_map", None)
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
    print(f"Requested mode : {args.device}")
    print(f"Input device   : {device}")
    print(f"Model dtype    : {next(model.parameters()).dtype}")
    print(f"HF device map  : {device_map if device_map is not None else 'single device'}")
    if device_map and any(str(value) in {"cpu", "disk"} for value in device_map.values()):
        print("WARNING: Some model layers are offloaded to CPU/disk; generation will be slow.")

    paths = _discover_mt(args.data_dir, "test", args.language_pairs)
    inference_collator = TranslationInferenceCollator(
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        prompt_format=args.prompt_format,
        enable_thinking=args.enable_thinking,
    )
    # Materialize the selected test set first, then tokenize every prompt before
    # generation. This keeps tokenization time separate from generation time.
    print("Loading test samples...", flush=True)
    rows = list(limit_per_direction(
        _parallel_rows(paths, args.direction), args.max_samples
    ))
    total_rows = len(rows)
    print(f"Test samples   : {total_rows}", flush=True)
    if total_rows == 0:
        raise ValueError("No test samples matched the requested language pairs/direction")
    tokenized_rows = [
        inference_collator.encode(row)
        for row in tqdm(
            rows,
            total=total_rows,
            desc="Tokenizing prompts",
            unit="sample",
            dynamic_ncols=True,
            file=sys.stdout,
        )
    ]
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
    collated_batches = list(tqdm(
        data_loader,
        total=len(data_loader),
        desc="Collating batches",
        unit="batch",
        dynamic_ncols=True,
        file=sys.stdout,
    ))
    print(f"Collated batches: {len(collated_batches)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with ExitStack() as stack, tqdm(
        total=total_rows,
        desc="Generating translations",
        unit="sample",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    ) as progress:
        writers = {}

        def writer_for(row: dict):
            direction = f"{row['source_lang']}-{row['target_lang']}"
            if direction not in writers:
                path = output_dir / f"{direction}.csv"
                handle = stack.enter_context(
                    path.open("w", encoding="utf-8-sig", newline="")
                )
                writers[direction] = csv.DictWriter(
                    handle, fieldnames=["src", "ref", "pred"]
                )
                writers[direction].writeheader()
            return direction, writers[direction]

        for batch in collated_batches:
            # Tokenization, batching and padding are already complete. The
            # generation loop only transfers a prepared batch to the model.
            row_batch = batch["rows"]
            prompt_width = batch["prompt_width"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.inference_mode():
                generation_kwargs = {}
                if args.enable_thinking:
                    generation_kwargs.update(
                        do_sample=True, temperature=0.6, top_p=0.95, top_k=20
                    )
                else:
                    generation_kwargs["do_sample"] = False
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                )
            predictions = tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            for row, prediction in zip(row_batch, predictions):
                direction, writer = writer_for(row)
                writer.writerow({
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
    for direction, count in sorted(counts.items()):
        print(f"Saved {count} rows to {(output_dir / f'{direction}.csv').resolve()}")


if __name__ == "__main__":
    main()
