"""Evaluation and representation-visualization utilities."""

from typing import Any

__all__ = [
    "evaluate_folder_sentence_bleu",
    "build_embeddings_dict",
    "extract_all_layer_embeddings",
    "extract_model_embeddings",
    "load_mt_pair_texts",
    "load_parallel_mt50_json",
    "plot_multilingual_tsne_scatter",
    "resolve_model_checkpoint",
    "run_translation_inference",
    "visualize_ot_alignment",
    "visualize_model_tsne",
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


def load_mt_pair_texts(*args: Any, **kwargs: Any):
    from .visualize_tsne import load_mt_pair_texts as implementation

    return implementation(*args, **kwargs)


def extract_model_embeddings(*args: Any, **kwargs: Any):
    from .visualize_tsne import extract_model_embeddings as implementation

    return implementation(*args, **kwargs)


def load_parallel_mt50_json(*args: Any, **kwargs: Any):
    from .visualize_tsne import load_parallel_mt50_json as implementation

    return implementation(*args, **kwargs)


def extract_all_layer_embeddings(*args: Any, **kwargs: Any):
    from .visualize_tsne import extract_all_layer_embeddings as implementation

    return implementation(*args, **kwargs)


def build_embeddings_dict(*args: Any, **kwargs: Any):
    from .visualize_tsne import build_embeddings_dict as implementation

    return implementation(*args, **kwargs)


def plot_multilingual_tsne_scatter(*args: Any, **kwargs: Any):
    from .visualize_tsne import plot_multilingual_tsne_scatter as implementation

    return implementation(*args, **kwargs)


def resolve_model_checkpoint(*args: Any, **kwargs: Any):
    from .visualize_tsne import resolve_model_checkpoint as implementation

    return implementation(*args, **kwargs)


def visualize_model_tsne(*args: Any, **kwargs: Any):
    from .visualize_tsne import visualize_model_tsne as implementation

    return implementation(*args, **kwargs)


def run_translation_inference(*args: Any, **kwargs: Any):
    from .inference import run_translation_inference as implementation

    return implementation(*args, **kwargs)
