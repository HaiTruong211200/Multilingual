"""Reusable t-SNE visualization for hidden-state or embedding matrices."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    parser.add_argument("--embeddings", required=True, help=".npy, .npz, or .csv")
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
    visualize_tsne(
        embeddings=args.embeddings,
        output_path=args.output_path,
        title=args.title,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        seed=args.seed,
        point_size=args.point_size,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
