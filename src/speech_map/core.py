"""Implementation of Mean Average Precision on speech features."""

import enum
from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


class KnnBackend(enum.Enum):
    """Enum to select the backend for k-NN computation."""

    FAISS = 1
    TORCH = 2


class Pooling(enum.Enum):
    """Pooling methods."""

    MEAN = 1
    MAX = 2
    MIN = 3
    HAMMING = 4


class FaissNotAvailableError(ImportError):
    """Raised when the faiss library is not available."""

    def __init__(self) -> None:
        super().__init__("Install faiss-cpu or faiss-gpu via conda first to use Faiss backend for k-NN")


class FeaturesDimensionError(ValueError):
    """To raise if input features have wrong dimension."""

    def __init__(self, fileid: str) -> None:
        super().__init__(f"Features from {fileid} are not 2D")


class MissingEmbeddingsError(ValueError):
    """To raise if some embeddings are missing."""

    def __init__(self, count: int) -> None:
        super().__init__(f"{count} embedding files are missing")


class NumberNeighborsError(ValueError):
    """To raise if the number of neighbors is larger than 2048."""

    def __init__(self, num: int) -> None:
        super().__init__(
            f"Maximum number of instances is {num + 1} > 2048, with Faiss backend on GPU."
            "Faiss on GPU can only compute k-NN with k <= 2048, therefore the behavior is not well-defined. "
            "Please subsample your dataset, or try to compute MAP on CPU or with torch backend."
        )


def hamming_window(x: Tensor) -> Tensor:
    """Apply the hamming window on the input Tensor."""
    window = torch.hamming_window(x.size(0), device=x.device)
    return (window @ x) / window.sum()


def match_pooling(name: Pooling) -> Callable[[Tensor], Tensor]:
    """Return the corresponding pooling function."""
    match name:
        case Pooling.MEAN:
            return partial(torch.mean, dim=0)
        case Pooling.MAX:
            return lambda x: torch.max(x, dim=0).values
        case Pooling.MIN:
            return lambda x: torch.min(x, dim=0).values
        case Pooling.HAMMING:
            return hamming_window
        case _:
            raise ValueError(name)


def find_all_files(root: str | Path, extension: str) -> dict[str, Path]:
    """Recursively find all files with the given `extension` in `root`."""
    return dict(sorted((p.stem, p) for p in Path(root).rglob(f"*{extension}")))


def segment_frontiers(frequency: float) -> tuple[pl.Expr, pl.Expr]:
    """Frontiers [start, end[ in the input features.

    See https://docs.cognitive-ml.fr/fastabx/advanced/slicing.html for more details.
    """
    start = (pl.col("onset") * frequency - 0.5).cast(pl.Float64).ceil().cast(pl.Int64).alias("start")
    end = ((pl.col("offset") * frequency - 0.5).cast(pl.Float64).floor().cast(pl.Int64) + 1).alias("end")
    return start, end


def read_annotations(source: str | Path) -> pl.DataFrame:
    """Read annotations from a JSONL file."""
    schema = pl.Schema(
        {
            "fileid": pl.String,
            "onset": pl.String,
            "offset": pl.String,
            "speaker": pl.String,
            "transcription": pl.String,
        }
    )
    annotations = pl.scan_ndjson(source, schema=schema).with_columns(
        pl.col("onset").str.to_decimal(), pl.col("offset").str.to_decimal()
    )
    mapping = annotations.select("transcription").unique().with_row_index("transcription_id")
    return annotations.join(mapping, on="transcription").collect()


def build_embeddings_and_labels(
    root: str | Path,
    jsonl: str | Path,
    *,
    pooling: Pooling = Pooling.MEAN,
    frequency: float = 50,
    feature_maker: Callable[[str | Path], Tensor] = torch.load,
    device: torch.device | None = None,
    file_extension: str = ".pt",
) -> tuple[Tensor, Tensor]:
    """Build the pooled embeddings and labels from the annotations and the pre-computed features.

    Args:
        root: Path to the directory with input files.
        jsonl: Path to the JSONL file with annotations.
        pooling: Pooling to use for the embeddings, either Pooling.MEAN, Pooling.MAX, Pooling.MIN, or Pooling.HAMMING.
        frequency: Frequency of the input features, used to compute the segment frontiers.
        feature_maker: Function to load the features, default is torch.load. You can use your own model here.
        device: Device to use for the embeddings, default is "cuda" if available, otherwise "cpu".
        file_extension: Extension of the input files, default is ".pt". If you use your own model,
                        you can change it to ".wav" for example.

    Returns:
        A tuple of two tensors: the pooled embeddings and the labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, jsonl = Path(root), Path(jsonl)
    pooling_fn = match_pooling(pooling)

    embeddings, segment_ids = [], []
    annotations = read_annotations(jsonl).with_columns(segment_frontiers(frequency))
    files = find_all_files(root, file_extension)
    if missing := set(annotations["fileid"].unique()) - set(files):
        raise MissingEmbeddingsError(len(missing))
    for fileid, path in tqdm(files.items(), desc="Pooling embeddings"):
        frames = feature_maker(path).to(device)
        if frames.ndim != 2:
            raise FeaturesDimensionError(fileid)
        for segment in annotations.filter(pl.col("fileid") == fileid).iter_rows(named=True):
            embeddings.append(pooling_fn(frames[segment["start"] : segment["end"]]))
            segment_ids.append(segment["transcription_id"])
    return torch.stack(embeddings), torch.tensor(segment_ids, device=device)


def faiss_knn(embeddings: Tensor, k: int) -> Tensor:
    """Compute the k-nearest neighbors with Faiss."""
    try:
        import faiss
        import faiss.contrib.torch_utils
    except ImportError as error:
        raise FaissNotAvailableError from error

    embeddings = F.normalize(embeddings, dim=1)
    if embeddings.is_cuda:
        indices = faiss.knn_gpu(faiss.StandardGpuResources(), embeddings, embeddings, k + 1)[1]
    else:
        indices = faiss.knn(embeddings, embeddings, k + 1)[1]
    return indices[:, 1:]


def torch_knn(embeddings: Tensor, k: int) -> Tensor:
    """Compute the k-nearest neighbors with PyTorch only."""
    embeddings = F.normalize(embeddings, dim=1)
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(float("-inf"))
    return torch.topk(similarity, k=k).indices


def knn(embeddings: Tensor, k: int, *, knn_backend: KnnBackend = KnnBackend.FAISS) -> Tensor:
    """Compute the k-nearest neighbors with the specified backend."""
    match knn_backend:
        case KnnBackend.FAISS:
            return faiss_knn(embeddings, k)
        case KnnBackend.TORCH:
            return torch_knn(embeddings, k)
        case _:
            raise ValueError(knn_backend)


def mean_average_precision_from_labels(y_pred: Tensor, y_true: Tensor, counts: Tensor) -> float:
    """Compute the MAP from the k-nearest neighbors labels and ground truth labels.

    Use the formula from equation (3) of: https://arxiv.org/abs/2003.08505
    Args:
        y_pred: Tensor of shape (n_samples, k) with predicted labels.
        y_true: Tensor of shape (n_samples,) with ground truth labels.
        counts: Tensor of shape (n_samples,) with the number of neighbors for each sample.

    Returns:
        The mean average precision score as a float.
    """
    correct = y_true.unsqueeze(dim=1) == y_pred
    cumulative_correct = torch.cumsum(correct, dim=1)
    k = torch.arange(1, y_pred.size(1) + 1, device=y_pred.device)
    precision_at_k = cumulative_correct * correct / k
    return (precision_at_k.sum(dim=1) / counts).mean().item()


def mean_average_precision(embeddings: Tensor, labels: Tensor, *, knn_backend: KnnBackend = KnnBackend.FAISS) -> float:
    """Compute the MAP from embeddings and labels.

    Args:
        embeddings: Tensor of shape (n_samples, d) with pooled embeddings.
        labels: Tensor of shape (n_samples,) with labels of the corresponding samples.
        knn_backend: Backend to compute k-NN, either KnnBackend.FAISS or KnnBackend.TORCH.

    Returns:
        The mean average precision score as a float.
    """
    _, inverse_indices, counts = torch.unique(labels, sorted=True, return_counts=True, return_inverse=True, dim=0)
    counts -= 1  # Remove itself
    if knn_backend == KnnBackend.FAISS and embeddings.is_cuda and counts.max() >= 2_048:
        raise NumberNeighborsError(counts.max().item())
    knn_indices = knn(embeddings, counts.max().item(), knn_backend=knn_backend)
    return mean_average_precision_from_labels(labels[knn_indices], labels, counts[inverse_indices])
