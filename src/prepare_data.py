"""Data preparation for both multilingual alignment and instruction FT."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .prompts import summarization_instruction


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
