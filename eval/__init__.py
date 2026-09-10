"""Evaluation and representation-visualization utilities."""

from typing import Any

__all__ = [
    "evaluate_folder_sentence_bleu",
    "visualize_ot_alignment",
    "visualize_tsne",
]


def evaluate_folder_sentence_bleu(*args: Any, **kwargs: Any):
    """Lazily import and run the folder-level sentence BLEU evaluator."""
    from .bleu import evaluate_folder_sentence_bleu as implementation

    return implementation(*args, **kwargs)


def visualize_tsne(*args: Any, **kwargs: Any):
    """Lazily import and run the reusable t-SNE visualizer."""
    from .visualize_tsne import visualize_tsne as implementation

    return implementation(*args, **kwargs)


def visualize_ot_alignment(*args: Any, **kwargs: Any):
    """Lazily load and visualize an optimal-transport alignment plan."""
    from .visualize_ot_alignment import visualize_ot_alignment as implementation

    return implementation(*args, **kwargs)
