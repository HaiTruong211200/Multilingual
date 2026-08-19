"""Data preparation for both multilingual alignment and instruction FT."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .prompts import (
    TRANSLATION_TARGET_MARKER,
    summarization_instruction,
    translation_instruction,
)


def prepare_alignment_dataset(
    dataset: DatasetDict,
    tokenizer,
    prompt_format: str = "plain",
    enable_thinking: bool = False,
    training_mode: str = "finetune",
) -> DatasetDict:
    """Tokenize Stage 1 fully and store labels plus exact source/target spans."""
    if training_mode not in {"finetune", "continue"}:
        raise ValueError("training_mode must be 'finetune' or 'continue'")

    def tokenize_row(row: dict) -> dict:
        source, target = str(row["source"]), str(row["target"])
        instruction = translation_instruction(row["source_lang"], row["target_lang"])
        if prompt_format == "chat":
            if not tokenizer.chat_template:
                raise ValueError("prompt_format='chat' requires tokenizer.chat_template")
            user_content = instruction + source
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": target},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            user_start, target_char_start = rendered.rfind(user_content), rendered.rfind(target)
            if user_start < 0 or target_char_start < 0:
                raise ValueError("Chat template changed source/target text")
            source_char_start = user_start + len(instruction)
            source_char_end = source_char_start + len(source)
            target_char_end = target_char_start + len(target)
            add_special_tokens = False
        else:
            rendered = instruction + source + TRANSLATION_TARGET_MARKER + target
            source_char_start = len(instruction)
            source_char_end = source_char_start + len(source)
            target_char_start = source_char_end + len(TRANSLATION_TARGET_MARKER)
            target_char_end = target_char_start + len(target)
            add_special_tokens = True

        encoded = tokenizer(
            rendered,
            add_special_tokens=add_special_tokens,
        )

        # Tokenize four progressively longer prefixes. For plain prompts these
        # correspond to: instruction; instruction+source;
        # instruction+source+target marker; and the complete prompt.
        prompt_before_source = rendered[:source_char_start]
        prompt_through_source = rendered[:source_char_end]
        prompt_before_target = rendered[:target_char_start]
        prompt_through_target = rendered[:target_char_end]

        def token_length(text: str) -> int:
            return len(tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_tensors=None,
            )["input_ids"])

        source_start = token_length(prompt_before_source)
        source_end = token_length(prompt_through_source)
        target_start = token_length(prompt_before_target)
        target_end = token_length(prompt_through_target)
        if source_end <= source_start or target_end <= target_start:
            raise ValueError("Source or target has an empty token interval")
        input_ids = encoded["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
            input_ids.append(eos_id)
        model_limit = tokenizer.model_max_length
        if model_limit < 10**9 and len(input_ids) > model_limit:
            raise ValueError(
                f"Translation sample has {len(input_ids)} tokens, exceeding "
                f"model_max_length={model_limit}"
            )
        labels = (
            [-100] * target_start + input_ids[target_start:]
            if training_mode == "finetune"
            else input_ids.copy()
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "source_start_positions": source_start,
            "source_end_positions": source_end,
            "target_start_positions": target_start,
            "target_end_positions": target_end,
        }

    return DatasetDict({
        split: split_dataset.map(
            tokenize_row,
            desc=f"Tokenizing alignment {split}",
            load_from_cache_file=False,
        )
        for split, split_dataset in dataset.items()
    })


def prepare_translation_inference_dataset(
    rows: Iterable[dict],
    tokenizer,
    max_input_length: int = 512,
    prompt_format: str = "plain",
    enable_thinking: bool = False,
) -> Dataset:
    """Build and tokenize every translation prompt before collation/inference."""

    def tokenize_row(row: dict) -> dict:
        prefix = translation_instruction(
            row["source_lang"], row["target_lang"], template_index=0
        )
        source = str(row["source"])

        def fit_source_to_context(render_prompt) -> list[int]:
            """Tokenize a complete prompt once per candidate; never join token pieces."""
            full_ids = render_prompt(source)
            if len(full_ids) <= max_input_length:
                return full_ids
            empty_ids = render_prompt("")
            if len(empty_ids) >= max_input_length:
                raise ValueError("max_input_length is too small for the prompt template")

            # Keep the longest source character prefix whose complete rendered
            # prompt fits. Every candidate is tokenized as one whole sequence,
            # so BOS and boundary-sensitive tokens are always handled correctly.
            low, high = 0, len(source)
            best_ids = empty_ids
            while low <= high:
                middle = (low + high) // 2
                candidate_ids = render_prompt(source[:middle])
                if len(candidate_ids) <= max_input_length:
                    best_ids = candidate_ids
                    low = middle + 1
                else:
                    high = middle - 1
            return best_ids

        if prompt_format == "chat":
            if not tokenizer.chat_template:
                raise ValueError("prompt_format='chat' requires tokenizer.chat_template")

            def render(text: str) -> list[int]:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prefix + text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            input_ids = fit_source_to_context(render)
        else:
            def render(text: str) -> list[int]:
                complete_prompt = prefix + text + TRANSLATION_TARGET_MARKER
                return tokenizer(
                    complete_prompt,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]

            input_ids = fit_source_to_context(render)
        return {"input_ids": input_ids}

    dataset = Dataset.from_list(list(rows))
    return dataset.map(tokenize_row, desc="Tokenizing translation test dataset")


def _discover_mt(data_dir: str, split: str, language_pairs: str) -> list[Path]:
    root = Path(data_dir)
    requested = [x.strip() for x in language_pairs.split(",") if x.strip()]
    if not requested or requested == ["all"]:
        paths = sorted(root.glob(f"*/{split}.*.json"))
    else:
        paths = []
        for pair in requested:
            found = sorted((root / pair).glob(f"{split}.*.json"))
            if not found:
                raise FileNotFoundError(f"No {split} data for pair '{pair}' in {root}")
            paths.extend(found)
    if not paths:
        raise FileNotFoundError(f"No {split} translation files found under {root}")
    return paths


def _iter_json(path: Path) -> Iterator[dict]:
    """Read JSONL with a .json suffix as well as regular JSON arrays/objects."""
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            yield from json.load(handle)
            return
        try:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        except json.JSONDecodeError:
            handle.seek(0)
            payload = json.load(handle)
            rows = payload.get("data") if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                raise ValueError(f"Unsupported JSON structure in {path}")
            yield from rows


def _parallel_rows(paths: Iterable[Path], direction: str) -> Iterator[dict]:
    for path in paths:
        for line_number, row in enumerate(_iter_json(path), start=1):
            src_lang, tgt_lang = row.get("src_lang"), row.get("tgt_lang")
            translations = row.get("translation")
            if not src_lang or not tgt_lang or not isinstance(translations, dict):
                raise ValueError(f"Invalid translation schema at {path}:{line_number}")
            source = str(translations.get(src_lang, "")).strip()
            target = str(translations.get(tgt_lang, "")).strip()
            if not source or not target:
                continue
            if direction in {"forward", "both"}:
                yield {"source": source, "target": target, "source_lang": src_lang, "target_lang": tgt_lang}
            if direction in {"reverse", "both"}:
                yield {"source": target, "target": source, "source_lang": tgt_lang, "target_lang": src_lang}


def load_parallel_dataset(
    data_dir: str = "data/MT",
    language_pairs: str = "all",
    direction: str = "forward",
    train_file: Optional[str] = None,
    validation_file: Optional[str] = None,
) -> DatasetDict:
    if direction not in {"forward", "reverse", "both"}:
        raise ValueError("direction must be forward, reverse, or both")
    train_paths = [Path(train_file)] if train_file else _discover_mt(data_dir, "train", language_pairs)
    valid_paths = [Path(validation_file)] if validation_file else _discover_mt(data_dir, "valid", language_pairs)
    kwargs = {"direction": direction}
    return DatasetDict(
        train=Dataset.from_generator(_parallel_rows, gen_kwargs={"paths": train_paths, **kwargs}),
        validation=Dataset.from_generator(_parallel_rows, gen_kwargs={"paths": valid_paths, **kwargs}),
    )


def _load_json_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(path)
    return load_dataset("json", data_files=str(path), split="train")


def load_instruction_dataset(
    xlsum_dir: str = "data/XLSum/XLSum",
    bactrian_dir: str = "data/Bactrian-Multilingual_Instruction",
    languages: str = "en,km,my,th,vi",
    bactrian_validation_ratio: float = 0.01,
    seed: int = 42,
) -> DatasetDict:
    selected = [x.strip() for x in languages.split(",") if x.strip()]
    train_parts, validation_parts = [], []
    for lang in selected:
        xlsum_root = Path(xlsum_dir) / lang
        for split_name, destination in (("train", train_parts), ("validation", validation_parts)):
            path = xlsum_root / f"{split_name}.{lang}.json"
            if path.exists():
                raw = _load_json_dataset(path)
                destination.append(raw.map(
                    lambda row, code=lang: {
                        "instruction": summarization_instruction(code),
                        "input": str(row["text"]).strip(),
                        "output": str(row["summary"]).strip(),
                        "language": code,
                        "dataset_name": "xlsum",
                    },
                    remove_columns=raw.column_names,
                    desc=f"Preparing XLSum {split_name} ({lang})",
                ))

        bactrian_path = Path(bactrian_dir) / f"{lang}.json"
        if bactrian_path.exists():
            raw = _load_json_dataset(bactrian_path)
            normalized = raw.map(
                lambda row, code=lang: {
                    "instruction": str(row["instruction"]).strip(),
                    "input": str(row.get("input") or "").strip(),
                    "output": str(row["output"]).strip(),
                    "language": code,
                    "dataset_name": "bactrian",
                },
                remove_columns=raw.column_names,
                desc=f"Preparing Bactrian ({lang})",
            ).filter(lambda row: bool(row["instruction"] and row["output"]))
            split = normalized.train_test_split(test_size=bactrian_validation_ratio, seed=seed)
            train_parts.append(split["train"])
            validation_parts.append(split["test"])
    if not train_parts or not validation_parts:
        raise ValueError(f"No usable instruction data for languages: {languages}")
    return DatasetDict(
        train=concatenate_datasets(train_parts).shuffle(seed=seed),
        validation=concatenate_datasets(validation_parts).shuffle(seed=seed),
    )
