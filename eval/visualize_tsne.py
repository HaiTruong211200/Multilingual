"""Visualize multilingual hidden states with the MT-pair t-SNE pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from tqdm.auto import tqdm


def read_json_or_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON array/object or a JSONL file (including JSONL named .json)."""
    path = Path(path)
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Empty data file: {path}")
    try:
        payload = json.loads(content)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "items", "records"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            return [payload]
    except json.JSONDecodeError:
        pass

    records = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path} at line {line_number}: {error}"
            ) from error
    return records


def extract_parallel_text(
    record: dict,
    lang1: str,
    lang2: str,
    path: str | Path,
    row_index: int,
) -> tuple[str, str]:
    """Extract and validate two translations from one MT record."""
    translation = record.get("translation")
    if not isinstance(translation, dict):
        raise ValueError(
            f"{path}, row {row_index}: missing translation dictionary; "
            f"available fields={list(record)}"
        )
    missing = [lang for lang in (lang1, lang2) if lang not in translation]
    if missing:
        raise ValueError(
            f"{path}, row {row_index}: translation is missing {missing}; "
            f"available languages={list(translation)}"
        )
    text1 = str(translation[lang1] or "").strip()
    text2 = str(translation[lang2] or "").strip()
    if not text1 or not text2:
        raise ValueError(f"{path}, row {row_index}: empty parallel sentence")
    return text1, text2


def load_parallel_mt50_json(
    data_dir: str | Path,
    language_pairs: Sequence[str] | str,
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    """Load selected MT pair folders into the notebook's long DataFrame format."""
    if isinstance(language_pairs, str):
        language_pairs = [
            pair.strip() for pair in language_pairs.split(",") if pair.strip()
        ]
    if not language_pairs or list(language_pairs) == ["all"]:
        language_pairs = sorted(
            path.name for path in Path(data_dir).iterdir() if path.is_dir()
        )
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    frames = []
    lengths: dict[str, int] = {}
    selected_files: dict[str, str] = {}
    for pair in language_pairs:
        parts = pair.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid language pair: {pair}")
        lang1, lang2 = parts
        pair_dir = Path(data_dir) / pair
        if not pair_dir.is_dir():
            raise FileNotFoundError(f"Language-pair folder not found: {pair_dir}")
        paths = sorted(pair_dir.glob(f"{split}.*.json"))
        if not paths:
            raise FileNotFoundError(f"No {split}.*.json in {pair_dir}")
        if len(paths) > 1:
            raise ValueError(
                f"Multiple {split} files in {pair_dir}: {[p.name for p in paths]}"
            )
        path = paths[0]
        records = read_json_or_jsonl(path)
        if max_samples is not None:
            records = records[:max_samples]

        pair_rows = {lang1: [], lang2: []}
        for row_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{path}, row {row_index}: expected JSON object")
            text1, text2 = extract_parallel_text(
                record, lang1, lang2, path, row_index
            )
            pair_rows[lang1].append(text1)
            pair_rows[lang2].append(text2)

        lengths[pair] = len(records)
        selected_files[pair] = str(path)
        for language in (lang1, lang2):
            frames.append(pd.DataFrame({
                "text": pair_rows[language],
                "sentence_id": np.arange(len(records), dtype=np.int64),
                "language": language,
                "language_pair": pair,
                "source_file": path.name,
            }))

    if not frames:
        raise ValueError("No MT samples were loaded")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(
        ["language_pair", "language", "sentence_id"]
    ).reset_index(drop=True)
    return frame, lengths, selected_files


def masked_mean_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool only valid (non-padding) tokens."""
    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    return (hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-9)


def resolve_model_checkpoint(model_name_or_path: str | Path) -> str:
    """Resolve the last Trainer checkpoint when an output directory is passed."""
    path = Path(model_name_or_path)
    if not path.is_dir():
        return str(model_name_or_path)
    # A directly loadable full model or adapter takes precedence.
    direct_markers = (
        "adapter_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if any((path / marker).exists() for marker in direct_markers):
        return str(path)

    from transformers.trainer_utils import get_last_checkpoint

    checkpoint = get_last_checkpoint(str(path))
    if checkpoint is None:
        raise FileNotFoundError(
            f"No loadable model or checkpoint-* directory found in: {path}"
        )
    print(f"Using last checkpoint: {checkpoint}")
    return checkpoint


def load_embedding_model(
    model_name_or_path: str | Path,
    *,
    trust_remote_code: bool = False,
):
    """Load a base/full model or a PEFT adapter and its base tokenizer."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_checkpoint(model_name_or_path)
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.is_file():
        from peft import PeftConfig, PeftModel

        config = PeftConfig.from_pretrained(model_path)
        tokenizer_name = config.base_model_name_or_path
        base = AutoModelForCausalLM.from_pretrained(
            tokenizer_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
    else:
        tokenizer_name = model_path
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=trust_remote_code
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def extract_all_layer_embeddings(
    texts: Sequence[str],
    tokenizer,
    model,
    *,
    batch_size: int = 32,
    max_length: int = 256,
) -> list[np.ndarray]:
    """Return one [num_sentences, hidden_dim] matrix for every hidden state."""
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    all_layers = None
    input_device = model.get_input_embeddings().weight.device
    for start in tqdm(
        range(0, len(texts), batch_size), desc="Extracting hidden states"
    ):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        outputs = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        if all_layers is None:
            all_layers = [[] for _ in hidden_states]
        for layer_index, hidden in enumerate(hidden_states):
            pooled = masked_mean_pooling(hidden, encoded["attention_mask"])
            all_layers[layer_index].append(pooled.float().cpu().numpy())
        del outputs, hidden_states, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if all_layers is None:
        raise ValueError("texts must not be empty")
    return [np.concatenate(parts, axis=0) for parts in all_layers]


def build_embeddings_dict(
    layer_embeddings: Sequence[np.ndarray | torch.Tensor],
    dataframe: pd.DataFrame,
    *,
    language_column: str = "language",
    sentence_id_column: str = "sentence_id",
    deduplicate_language: str | None = "en",
) -> dict[str, torch.Tensor]:
    """Convert list[layer][row, dim] to dict[language][layer, sample, dim]."""
    working = dataframe.reset_index(drop=True).copy()
    working["_embedding_row"] = np.arange(len(working))
    result = {}
    for language in sorted(working[language_column].unique()):
        rows = working[working[language_column] == language].copy()
        if language == deduplicate_language:
            rows = rows.drop_duplicates(subset=[sentence_id_column], keep="first")
        rows = rows.sort_values(sentence_id_column)
        positions = rows["_embedding_row"].to_numpy()
        tensors = [
            torch.as_tensor(layer)[positions].detach().float().cpu()
            for layer in layer_embeddings
        ]
        result[language] = torch.stack(tensors, dim=0)
    return result


def plot_multilingual_tsne_scatter(
    embeddings_dict: dict[str, torch.Tensor],
    *,
    sample_size: int | None = None,
    layers: Sequence[int] | None = None,
    perplexity: float = 30,
    cols: int = 3,
    figsize_scale: float = 4,
    random_state: int = 42,
    point_size: float = 14,
    alpha: float = 0.8,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot joint per-layer t-SNE projections with a global KDE background."""
    from sklearn.manifold import TSNE

    if not embeddings_dict:
        raise ValueError("embeddings_dict is empty")
    labels = list(embeddings_dict)
    first = next(iter(embeddings_dict.values()))
    if not torch.is_tensor(first) or first.ndim != 3:
        raise ValueError("Each embedding must be [num_layers, samples, hidden_dim]")
    num_layers, _, hidden_dim = first.shape
    for language, embedding in embeddings_dict.items():
        if (
            not torch.is_tensor(embedding)
            or embedding.ndim != 3
            or embedding.shape[0] != num_layers
            or embedding.shape[2] != hidden_dim
        ):
            raise ValueError(f"Inconsistent embedding shape for {language}")
    layers = list(range(7, num_layers)) if layers is None else list(layers)
    invalid = [layer for layer in layers if not 0 <= layer < num_layers]
    if invalid or not layers:
        raise ValueError(f"Invalid or empty layers: {invalid}")
    min_samples = min(value.shape[1] for value in embeddings_dict.values())
    sample_size = min_samples if sample_size is None else min(sample_size, min_samples)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    total_samples = sample_size * len(labels)
    effective_perplexity = min(perplexity, total_samples - 1)

    known_colors = {
        "en": "#1f77b4", "km": "#ff7f0e", "my": "#2ca02c",
        "th": "#d62728", "vi": "#9467bd", "zh": "#8c564b",
        "ko": "#e377c2",
    }
    fallback = sns.color_palette("husl", len(labels)).as_hex()
    palette = {
        language: known_colors.get(language, fallback[index])
        for index, language in enumerate(labels)
    }
    rows = (len(layers) + cols - 1) // cols
    figure, axes = plt.subplots(
        rows, cols, figsize=(figsize_scale * cols, figsize_scale * rows),
        squeeze=False,
    )
    axes = axes.flatten()
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor=palette[name],
            markeredgecolor="white", markersize=6, label=name,
        )
        for name in labels
    ]
    coordinates = {}
    for plot_index, layer in enumerate(layers):
        axis = axes[plot_index]
        matrix = torch.cat(
            [embeddings_dict[name][layer, :sample_size] for name in labels], dim=0
        ).numpy()
        projected = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(matrix)
        coordinates[layer] = projected
        if len(projected) >= 5:
            sns.kdeplot(
                x=projected[:, 0], y=projected[:, 1], fill=True, levels=12,
                thresh=0.03, bw_adjust=0.8, color="#BDBDBD", alpha=0.38,
                ax=axis, zorder=1,
            )
        start = 0
        for name in labels:
            points = projected[start : start + sample_size]
            axis.scatter(
                points[:, 0], points[:, 1], s=point_size,
                color=palette[name], alpha=alpha, edgecolors="white",
                linewidths=0.2, zorder=2,
            )
            start += sample_size
        axis.set(title=f"Layer {layer}", xlabel="x", ylabel="y")
        axis.grid(True, linewidth=0.4, alpha=0.25)
        axis.legend(handles=handles, loc="upper right", fontsize=8)
    for index in range(len(layers), len(axes)):
        axes[index].axis("off")
    figure.tight_layout()
    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, coordinates, saved_path


def visualize_model_tsne(
    model_name_or_path: str | Path,
    data_dir: str | Path,
    language_pairs: Sequence[str] | str,
    *,
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int = 32,
    max_length: int = 256,
    sample_size: int | None = 1000,
    layers: Sequence[int] | None = None,
    deduplicate_language: str | None = "en",
    perplexity: float = 30,
    cols: int = 3,
    output_path: str | Path | None = "outputs/eval/tsne.png",
    trust_remote_code: bool = False,
    show: bool = True,
) -> dict:
    """Run the complete notebook pipeline with one reusable function call."""
    dataframe, pair_lengths, selected_files = load_parallel_mt50_json(
        data_dir, language_pairs, split=split, max_samples=max_samples
    )
    model, tokenizer = load_embedding_model(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    layer_embeddings = extract_all_layer_embeddings(
        dataframe["text"].tolist(), tokenizer, model,
        batch_size=batch_size, max_length=max_length,
    )
    embeddings_dict = build_embeddings_dict(
        layer_embeddings, dataframe,
        deduplicate_language=deduplicate_language,
    )
    figure, coordinates, saved_path = plot_multilingual_tsne_scatter(
        embeddings_dict,
        sample_size=sample_size,
        layers=layers,
        perplexity=perplexity,
        cols=cols,
        output_path=output_path,
        show=show,
    )
    return {
        "dataframe": dataframe,
        "pair_lengths": pair_lengths,
        "selected_files": selected_files,
        "layer_embeddings": layer_embeddings,
        "embeddings_dict": embeddings_dict,
        "coordinates": coordinates,
        "figure": figure,
        "saved_path": saved_path,
    }


# Backward-compatible alias: this module now expects model + MT pair data.
visualize_tsne = plot_multilingual_tsne_scatter
extract_model_embeddings = extract_all_layer_embeddings
load_mt_pair_texts = load_parallel_mt50_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--language-pairs", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--deduplicate-language", default="en")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--output-path", default="outputs/eval/tsne.png")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = visualize_model_tsne(
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        language_pairs=args.language_pairs,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        sample_size=args.sample_size,
        layers=args.layers,
        deduplicate_language=args.deduplicate_language,
        perplexity=args.perplexity,
        cols=args.cols,
        output_path=args.output_path,
        trust_remote_code=args.trust_remote_code,
        show=not args.no_show,
    )
    print("Files:", result["selected_files"])
    print("Pair lengths:", result["pair_lengths"])
    print("Saved:", result["saved_path"])


if __name__ == "__main__":
    main()
