"""Run translation inference on MT test splits and export src/ref/pred CSV."""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from .prepare_data import _discover_mt, _parallel_rows
from .prompts import TRANSLATION_TARGET_MARKER, translation_instruction


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
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def build_prompt_ids(tokenizer, row: dict, max_length: int) -> list[int]:
    # Fix one template during evaluation so predictions are reproducible.
    prefix = translation_instruction(
        row["source_lang"], row["target_lang"], template_index=0
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    marker_ids = tokenizer(TRANSLATION_TARGET_MARKER, add_special_tokens=False)["input_ids"]
    source_ids = tokenizer(row["source"], add_special_tokens=False)["input_ids"]
    source_budget = max_length - len(prefix_ids) - len(marker_ids)
    if source_budget < 1:
        raise ValueError("max_input_length is too small for the translation prompt")
    return prefix_ids + source_ids[:source_budget] + marker_ids


def batches(iterator, size: int):
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    device = model.get_input_embeddings().weight.device

    paths = _discover_mt(args.data_dir, "test", args.language_pairs)
    rows = limit_per_direction(
        _parallel_rows(paths, args.direction), args.max_samples
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with ExitStack() as stack, tqdm(desc="Generating translations", unit="sample") as progress:
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

        for row_batch in batches(iter(rows), args.batch_size):
            prompt_rows = [
                build_prompt_ids(tokenizer, row, args.max_input_length)
                for row in row_batch
            ]
            width = max(map(len, prompt_rows))
            input_ids = torch.full(
                (len(prompt_rows), width), tokenizer.pad_token_id,
                dtype=torch.long, device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            # Left padding is required for batched generation by a decoder-only LM.
            for index, ids in enumerate(prompt_rows):
                length = len(ids)
                input_ids[index, -length:] = torch.tensor(ids, device=device)
                attention_mask[index, -length:] = 1
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            predictions = tokenizer.batch_decode(
                generated[:, width:], skip_special_tokens=True
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
