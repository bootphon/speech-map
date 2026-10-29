# LibriSpeech words and n-grams annotations

Annotations of words and phone n-grams in the LibriSpeech evaluation subsets (dev-clean, dev-other, test-clean,
test-other), to compute the MAP with `speech-map`.

They were used in:

- [Speech Sequence Embeddings using Nearest Neighbors Contrastive Learning](https://arxiv.org/abs/2204.05148)
  (Algayres et al., Interspeech 2022), where they were introduced, and
- [SpidR: Learning Fast and Stable Linguistic Units for Spoken Language Models Without Supervision](https://arxiv.org/abs/2512.20308)
  (Poli et al., TMLR 2025).

## Format

Each line of the JSONL files is a segment:

```json
{"fileid":"5338-24640-0002","speaker":"5338","onset":0.54,"offset":0.86,"transcription":"IY1,V,AH0,L"}
```

`onset` and `offset` are in seconds, and `transcription` is the sequence of phones (ARPAbet with stress), without
silences. Two segments with the same transcription are considered as instances of the same item.

## Build from TextGrids

`build.py` builds the annotations from forced alignments in TextGrid format, such as https://zenodo.org/records/2619474
obtained with the Montreal Forced Aligner (MFA). In an environment with `polars`:

```bash
python build.py path/to/librispeech_alignments.zip path/to/output
zstd --rm -f path/to/output/*.jsonl
```

You can also use your own alignments: give a directory with one subdirectory per split (`--splits`), containing
TextGrids in the long text format with a "words" and a "phones" tier, as written by MFA. The file id is the name of
the TextGrid, and the speaker is the first directory after the split (`split/speaker/.../fileid.TextGrid`).

### Words

`{split}-words.jsonl` contains the words that:

- are not in an utterance that contains an out-of-vocabulary word (`<unk>`),
- are not hapaxes: their spelling appears at least twice in the split,
- last at least 0.07 s and less than 0.96 s,
- and their transcription appears at least twice after the previous filters.

### N-grams

`{split}-ngrams.jsonl` contains the sequences of consecutive phones that:

- are not in an utterance that contains an out-of-vocabulary word (`<unk>`),
- do not cross a silence, nor a short pause longer than 0.15 s. The shorter pauses are not in the transcription,
- last at least 0.07 s and less than 1 s,
- and whose transcription appears at least twice.

## Distributed files

The files distributed in this directory are the original ones, from the
[original implementation](https://arxiv.org/abs/2204.05148). They follow the same rules, with a few differences:

- They come from Robin Algayres's re-alignment of LibriSpeech, available in the `librispeech` directory of
  [abx_sem_syn](https://gitlab.cognitive-ml.fr/ralgayres/abx_sem_syn). It has the same utterances and words as the
  Zenodo alignment from above (also without the utterances that contain an out-of-vocabulary word), but about 3% of the
  boundaries are shifted by 10 ms, and 0.2% of the words have another pronunciation.
- For the words of dev-other, test-clean and test-other, the transcriptions come from the Zenodo alignment and the
  boundaries from Robin's. This is why a few of these words have a transcription that does not match the phones of
  Robin's alignment.
- The durations are computed in floating point. This is why some words and n-grams of 0.07 s are missing, and why a
  few n-grams last 1.00 s.

`dev-clean-ngrams-subset.jsonl` is a random subset of `dev-clean-ngrams.jsonl`, restricted to 662 utterances and
without hapaxes. It was drawn without a fixed seed and cannot be rebuilt.

Compared to the annotations built with `build.py` from the Zenodo alignments, 99.8 to 100% of the distributed words
are also selected with the same transcription, and 93 to 96% with the same boundaries. `build.py` selects about 0.7%
more words and 1.3% more n-grams, mostly those of 0.07 s.

With SpidR (layer 6, mean pooling), the word MAP is 66.50% on dev-clean and 55.26% on dev-other with the
distributed files, and 66.40% and 55.14% with `build.py`.
