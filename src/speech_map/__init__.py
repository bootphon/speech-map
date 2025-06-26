"""Mean Average Precision over n-grams / words with speech features."""

from .core import KnnBackend, Pooling, build_embeddings_and_labels, knn, mean_average_precision

__all__ = ["KnnBackend", "Pooling", "build_embeddings_and_labels", "knn", "mean_average_precision"]
