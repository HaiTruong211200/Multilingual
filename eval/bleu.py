"""Sentence-level BLEU evaluation for translation JSONL files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _normalize_choices(values: list[str] | tuple[str, ...] | str | None) -> set[str] | None:
    """Normalize Python/CLI selections, accepting both spaces and commas."""
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    choices = {
        choice.strip().lower()
        for value in values
        for choice in value.split(",")
        if choice.strip()
    }
    return choices or None


def _select_input_files(
    input_dir: Path,
    files: list[str | Path] | tuple[str | Path, ...] | str | Path | None,
) -> list[Path]:
    """Resolve explicit files, or discover every translation JSONL file."""
    if files is None:
        return sorted(input_dir.glob("translation.*-*.jsonl"))
    if isinstance(files, (str, Path)):
        files = [files]

    selected = []
    for value in files:
        file_path = Path(value)
        if not file_path.is_absolute():
            file_path = input_dir / file_path
        if not file_path.is_file():
            raise FileNotFoundError(f"BLEU input file does not exist: {file_path}")
        selected.append(file_path)
    return selected


def evaluate_folder_sentence_bleu(
    input_dir: str | Path,
    split_at: int = 2000,
    output_csv: str | Path | None = None,
    language_pairs: list[str] | tuple[str, ...] | str | None = None,
    source_languages: list[str] | tuple[str, ...] | str | None = None,
    target_languages: list[str] | tuple[str, ...] | str | None = None,
    files: list[str | Path] | tuple[str | Path, ...] | str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate selected translation JSONL files and return a summary table.

    All selection arguments are optional. When several filters are supplied, a
    file must satisfy all of them. Values may be passed as lists or as
    comma-separated strings, for example ``language_pairs="en-vi,en-zh"``.
    """
    if split_at < 0:
        raise ValueError("split_at must be non-negative")
    try:
        import sacrebleu
    except ImportError as error:
        raise ImportError(
            "BLEU evaluation requires sacrebleu. Install it with "
            "'pip install sacrebleu'."
        ) from error

    input_dir = Path(input_dir)
    pair_filter = _normalize_choices(language_pairs)
    src_filter = _normalize_choices(source_languages)
    tgt_filter = _normalize_choices(target_languages)
    input_files = _select_input_files(input_dir, files)

    rows = []
    for file_path in input_files:
        match = re.fullmatch(r"translation\.([^.]+)-([^.]+)\.jsonl", file_path.name)
        if match is None:
            raise ValueError(
                "Input filename must match translation.<src>-<tgt>.jsonl: "
                f"{file_path.name}"
            )

        src_lang, tgt_lang = match.group(1).lower(), match.group(2).lower()
        pair = f"{src_lang}-{tgt_lang}"
        if pair_filter is not None and pair not in pair_filter:
            continue
        if src_filter is not None and src_lang not in src_filter:
            continue
        if tgt_filter is not None and tgt_lang not in tgt_filter:
            continue

        scores: list[float] = []
        skipped = 0
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    prediction = str(item.get("prediction", "")).strip()
                    reference = str(item.get("gold", "")).strip()
                    if not prediction or not reference:
                        skipped += 1
                        continue
                    scores.append(
                        sacrebleu.sentence_bleu(
                            prediction,
                            [reference],
                            tokenize="13a",
                            smooth_method="exp",
                            use_effective_order=True,
                        ).score
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    skipped += 1

        first, rest = scores[:split_at], scores[split_at:]
        row = {
            "file": file_path.name,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "pair": pair,
            "samples": len(scores),
            "skipped": skipped,
            f"sentence_bleu_0_{split_at}": (
                sum(first) / len(first) if first else float("nan")
            ),
            f"sentence_bleu_{split_at}_end": (
                sum(rest) / len(rest) if rest else float("nan")
            ),
        }
        rows.append(row)
        print(
            f"{row['pair']:8s} | 0-{split_at}: "
            f"{row[f'sentence_bleu_0_{split_at}']:.2f} | "
            f"{split_at}-end: {row[f'sentence_bleu_{split_at}_end']:.2f} | "
            f"N={row['samples']} | skipped={skipped}"
        )

    result = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"Saved BLEU summary: {output_csv.resolve()}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--split-at", type=int, default=2000)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--language-pairs",
        nargs="+",
        default=None,
        metavar="PAIR",
        help="Pairs to evaluate, e.g. en-vi en-zh or en-vi,en-zh.",
    )
    parser.add_argument(
        "--source-languages",
        nargs="+",
        default=None,
        metavar="LANG",
        help="Only evaluate these source languages.",
    )
    parser.add_argument(
        "--target-languages",
        nargs="+",
        default=None,
        metavar="LANG",
        help="Only evaluate these target languages.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Specific JSONL filenames/paths. Relative paths use --input-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_folder_sentence_bleu(
        input_dir=args.input_dir,
        split_at=args.split_at,
        output_csv=args.output_csv,
        language_pairs=args.language_pairs,
        source_languages=args.source_languages,
        target_languages=args.target_languages,
        files=args.files,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
