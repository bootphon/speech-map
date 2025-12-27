# Mean Average Precision over words or n-grams with speech features

Compute the Mean Average Precision (MAP) with speech features.

This is the MAP@R from equation (3) of https://arxiv.org/abs/2003.08505.

## Installation

This package is available on PyPI:

```bash
pip install speech-map
```

The naive PyTorch backend for the k-NN is enough to compute the MAP over words quickly.

You might want to use the Faiss backend if you compute the MAP over n-grams or if have a large number of
embeddings. In this case, since Faiss is not available on PyPI, you can install this package in a pixi or conda
environment. We recommend using pixi on Linux: clone this repository and run `pixi shell -e faiss-cpu` or
`pixi shell -e faiss-gpu`.

With conda, first install Faiss in your conda environment (be careful about your PyTorch and Faiss versions,
and from which channel they come from), and then install `speech-map` using pip.

## Usage

### CLI

```
❯ python -m speech_map --help
usage: __main__.py [-h] [--pooling {MEAN,MAX,MIN,HAMMING}] [--frequency FREQUENCY] [--backend {FAISS,TORCH}] features jsonl

Mean Average Precision over n-grams / words with speech features

positional arguments:
  features              Path to the directory with pre-computed features
  jsonl                 Path to the JSONL file with annotations

options:
  -h, --help            show this help message and exit
  --pooling {MEAN,MAX,MIN,HAMMING}
                        Pooling (default: MEAN)
  --frequency FREQUENCY
                        Feature frequency in Hz (default: 50 Hz)
  --backend {FAISS,TORCH}
                        KNN (default: TORCH)
```

### Python API

You most probably need only two functions: `build_embeddings_and_labels` and `mean_average_precision`.
Use them like this:

```python
from speech_map import build_embeddings_and_labels, mean_average_precision

embeddings, labels = build_embeddings_and_labels(path_to_features, path_to_jsonl)
print(mean_average_precision(embeddings, labels))
```

In this example, `path_to_features` is a path to a directory containing features stored in individual PyTorch
tensor files, and `path_to_jsonl` is the path to the JSONL annotations file.

You can also use those functions in a more advanced setting like this:

```python
from speech_map import Pooling, build_embeddings_and_labels, mean_average_precision

embeddings, labels = build_embeddings_and_labels(
    path_to_features,
    path_to_jsonl,
    pooling=Pooling.MAX,
    frequency=100,
    feature_maker=my_model,
    file_extension=".wav",
)
print(mean_average_precision(embeddings, labels))
```

This is a minimal package, and you can easily go through the code in `src/speech_map/core.py` if you want to check the details.

## Data

We distribute in `data` the words and n-grams annotations for LibriSpeech evaluation subsets. Decompress them with zstd.

We have not used the n-grams annotations recently; there is probably too much samples and they would need some clever subsampling.

## References

MAP for speech representations:

```bibtex
@inproceedings{carlin11_interspeech,
  title     = {Rapid evaluation of speech representations for spoken term discovery},
  author    = {Michael A. Carlin and Samuel Thomas and Aren Jansen and Hynek Hermansky},
  year      = {2011},
  booktitle = {Interspeech 2011},
  pages     = {821--824},
  doi       = {10.21437/Interspeech.2011-304},
  issn      = {2958-1796},
}
```

Data and original implementation:

```bibtex
@inproceedings{algayres20_interspeech,
  title     = {Evaluating the Reliability of Acoustic Speech Embeddings},
  author    = {Robin Algayres and Mohamed Salah Zaiem and Benoît Sagot and Emmanuel Dupoux},
  year      = {2020},
  booktitle = {Interspeech 2020},
  pages     = {4621--4625},
  doi       = {10.21437/Interspeech.2020-2362},
  issn      = {2958-1796},
}
```

