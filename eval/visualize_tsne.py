"""Reusable t-SNE visualization for hidden-state or embedding matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_multilingual_texts(
    data_file: str | Path,
    languages: Sequence[str] | str | None = None,
    *,
    text_column: str = "text",
    language_column: str = "language",
    max_samples_per_language: int | None = None,
) -> dict[str, list[str]]:
    """Load and filter multilingual text rows from CSV, JSON, or JSONL."""
    path = Path(data_file)
    if path.suffix.lower() == ".csv":
        records = pd.read_csv(path).to_dict("records")
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload if isinstance(payload, list) else payload.get("data", [])
    else:
        raise ValueError("data_file must be .csv, .json, or .jsonl")

    if isinstance(languages, str):
        languages = [value.strip() for value in languages.split(",") if value.strip()]
    selected = set(languages) if languages else None
    grouped: dict[str, list[str]] = {}
    for row in records:
        language = str(row.get(language_column, "")).strip()
        text = str(row.get(text_column, "")).strip()
        if not language or not text or (selected is not None and language not in selected):
            continue
        samples = grouped.setdefault(language, [])
        if max_samples_per_language is None or len(samples) < max_samples_per_language:
            samples.append(text)

    missing = selected - grouped.keys() if selected else set()
    if missing:
        raise ValueError(f"No samples found for languages: {sorted(missing)}")
    if not grouped:
        raise ValueError("No valid multilingual samples were loaded")
    return grouped


def extract_model_embeddings(
    model_name_or_path: str | Path,
    texts_by_language: Mapping[str, Sequence[str]],
    *,
    layer: int = -1,
    batch_size: int = 16,
    max_length: int | None = None,
    trust_remote_code: bool = False,
):
    """Extract masked-mean sentence embeddings from a base model or PEFT adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(model_name_or_path)
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.is_file():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path)
        tokenizer_name = peft_config.base_model_name_or_path
        model = AutoModelForCausalLM.from_pretrained(
            tokenizer_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        tokenizer_name = model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    device = next(model.parameters()).device

    texts: list[str] = []
    groups: list[str] = []
    for language, samples in texts_by_language.items():
        texts.extend(str(sample) for sample in samples)
        groups.extend([language] * len(samples))

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokenize_kwargs = {
                "padding": True,
                "truncation": max_length is not None,
                "return_tensors": "pt",
            }
            if max_length is not None:
                tokenize_kwargs["max_length"] = max_length
            batch = tokenizer(texts[start : start + batch_size], **tokenize_kwargs)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[layer]
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(pooled.float().cpu())

    return torch.cat(vectors).numpy(), groups, texts


def visualize_model_tsne(
    model_name_or_path: str | Path,
    texts_by_language: Mapping[str, Sequence[str]],
    *,
    languages: Sequence[str] | str | None = None,
    layer: int = -1,
    batch_size: int = 16,
    max_length: int | None = None,
    trust_remote_code: bool = False,
    **visualize_kwargs,
):
    """Select languages, extract model embeddings, and visualize their t-SNE map."""
    if isinstance(languages, str):
        languages = [value.strip() for value in languages.split(",") if value.strip()]
    if languages:
        missing = set(languages) - texts_by_language.keys()
        if missing:
            raise ValueError(f"No input texts supplied for languages: {sorted(missing)}")
        texts_by_language = {lang: texts_by_language[lang] for lang in languages}

    embeddings, groups, texts = extract_model_embeddings(
        model_name_or_path,
        texts_by_language,
        layer=layer,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )
    result = visualize_tsne(embeddings, groups=groups, labels=texts, **visualize_kwargs)
    result["embeddings"] = embeddings
    result["texts"] = texts
    return result


def _as_numpy(embeddings) -> np.ndarray:
    """Accept an array/tensor or load a numeric matrix from npy/npz/csv."""
    if isinstance(embeddings, (str, Path)):
        path = Path(embeddings)
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        elif path.suffix.lower() == ".npz":
            archive = np.load(path)
            if not archive.files:
                raise ValueError(f"No array found in {path}")
            array = archive[archive.files[0]]
        elif path.suffix.lower() == ".csv":
            array = pd.read_csv(path).select_dtypes(include=[np.number]).to_numpy()
        else:
            raise ValueError("Embedding file must be .npy, .npz, or .csv")
    elif hasattr(embeddings, "detach"):
        array = embeddings.detach().float().cpu().numpy()
    else:
        array = np.asarray(embeddings)

    if array.ndim != 2:
        raise ValueError(f"Expected [num_samples, hidden_size], got {array.shape}")
    if array.shape[0] < 3:
        raise ValueError("t-SNE requires at least three samples")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain NaN or infinity")
    return array.astype(np.float32, copy=False)


def visualize_tsne(
    embeddings,
    groups: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    *,
    output_path: str | Path | None = "outputs/eval/tsne.png",
    title: str = "t-SNE embedding visualization",
    perplexity: float = 30.0,
    learning_rate: float | str = "auto",
    max_iter: int = 1000,
    seed: int = 42,
    point_size: float = 45,
    annotate: bool = False,
    show: bool = True,
):
    """Project embeddings to 2-D, draw them, and return data plus figure."""
    matrix = _as_numpy(embeddings)
    sample_count = matrix.shape[0]
    if groups is not None and len(groups) != sample_count:
        raise ValueError("groups length must match number of embeddings")
    if labels is not None and len(labels) != sample_count:
        raise ValueError("labels length must match number of embeddings")

    effective_perplexity = min(float(perplexity), float(sample_count - 1))
    if effective_perplexity <= 0:
        raise ValueError("perplexity must be positive")
    try:
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise ImportError(
            "t-SNE visualization requires scikit-learn. Install it with "
            "'pip install scikit-learn'."
        ) from error
    coordinates = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        random_state=seed,
    ).fit_transform(matrix)

    frame = pd.DataFrame({"tsne_1": coordinates[:, 0], "tsne_2": coordinates[:, 1]})
    frame["group"] = list(groups) if groups is not None else "all"
    if labels is not None:
        frame["label"] = list(labels)

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=frame,
        x="tsne_1",
        y="tsne_2",
        hue="group",
        s=point_size,
        alpha=0.8,
        ax=axis,
    )
    axis.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    if annotate and labels is not None:
        for row in frame.itertuples(index=False):
            axis.annotate(
                row.label,
                (row.tsne_1, row.tsne_2),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )
    figure.tight_layout()

    saved_to = None
    if output_path is not None:
        saved_to = Path(output_path)
        saved_to.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_to, dpi=180, bbox_inches="tight")
        print(f"Saved t-SNE figure: {saved_to.resolve()}")
    if show:
        plt.show()
    return {
        "coordinates": frame,
        "figure": figure,
        "axis": axis,
        "saved_to": saved_to,
        "effective_perplexity": effective_perplexity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--embeddings", help="Precomputed .npy, .npz, or .csv")
    source.add_argument(
        "--model-name-or-path",
        help="Base/checkpoint model or PEFT adapter used to extract embeddings.",
    )
    parser.add_argument("--data-file", help="CSV/JSON/JSONL containing text rows.")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--max-samples-per-language", type=int, default=None)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-path", default="outputs/eval/tsne.png")
    parser.add_argument("--title", default="t-SNE embedding visualization")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning-rate", default="auto")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point-size", type=float, default=45)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw_kwargs = dict(
        output_path=args.output_path,
        title=args.title,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        seed=args.seed,
        point_size=args.point_size,
        show=not args.no_show,
    )
    if args.model_name_or_path:
        if not args.data_file:
            raise SystemExit("--data-file is required with --model-name-or-path")
        texts_by_language = load_multilingual_texts(
            args.data_file,
            args.languages,
            text_column=args.text_column,
            language_column=args.language_column,
            max_samples_per_language=args.max_samples_per_language,
        )
        visualize_model_tsne(
            args.model_name_or_path,
            texts_by_language,
            languages=args.languages,
            layer=args.layer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
            **draw_kwargs,
        )
    else:
        visualize_tsne(embeddings=args.embeddings, **draw_kwargs)


if __name__ == "__main__":
    main()
