"""Entry point for computing MAP."""

import argparse
from pathlib import Path

import torch

from .core import KnnBackend, Pooling, build_embeddings_and_labels, mean_average_precision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean Average Precision over n-grams / words with speech features")
    parser.add_argument("features", type=Path, help="Path to the directory with pre-computed features")
    parser.add_argument("jsonl", type=Path, help="Path to the JSONL file with annotations")
    parser.add_argument("--pooling", choices=Pooling._member_names_, default="MEAN", help="Pooling (default: MEAN)")
    parser.add_argument("--frequency", type=float, default=50, help="Feature frequency in Hz (default: 50 Hz)")
    parser.add_argument("--backend", choices=KnnBackend._member_names_, default="FAISS", help="KNN (default: FAISS)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings, labels = build_embeddings_and_labels(
        args.features,
        args.jsonl,
        pooling=Pooling[args.pooling],
        frequency=args.frequency,
        device=device,
        file_extension=".pt",
    )
    print("Computing MAP")
    score = mean_average_precision(embeddings, labels, knn_backend=KnnBackend[args.backend])
    print(f"{score:.2%}")
