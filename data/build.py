"""Build words and n-grams annotations from forced alignments in TextGrid format.

The TextGrids are read from a zip archive or a directory, with one subdirectory per split. They must have a
WORDS_TIER and a PHONES_TIER, as in the outputs of the Montreal Forced Aligner. See README.md for more details.
"""  # ruff: ignore[implicit-namespace-package]

import argparse
import zipfile
from pathlib import Path

import polars as pl

SILENCES = ["", "sil", "SIL", "spn"]
WORDS_TIER, PHONES_TIER, UNKNOWN, PAUSE = "words", "phones", "<unk>", "sp"
MIN_DURATION, MAX_WORD_DURATION, MAX_NGRAM_DURATION, MAX_PAUSE = 70, 960, 1000, 150  # In milliseconds
DURATION = pl.col("offset") - pl.col("onset")


def read_files(source: Path, split: str) -> pl.DataFrame:
    """Read the path and content of all the TextGrids of a split."""
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            names = [
                name for name in archive.namelist() if name.startswith(f"{split}/") and name.endswith(".TextGrid")
            ]
            contents = [archive.read(name).decode() for name in names]
    else:
        paths = sorted((source / split).rglob("*.TextGrid"))
        names = [path.relative_to(source).as_posix() for path in paths]
        contents = [path.read_text() for path in paths]
    return pl.DataFrame({"path": names, "content": contents}, schema={"path": pl.String, "content": pl.String})


def milliseconds(column: pl.Expr) -> pl.Expr:
    """Convert a string in seconds to an integer in milliseconds."""
    return (column.cast(pl.Decimal(scale=3)) * 1000).cast(pl.Int64)


def seconds(column: pl.Expr) -> pl.Expr:
    """Convert an integer in milliseconds to a float in seconds."""
    return (column.cast(pl.Decimal(scale=3)) / 1000).cast(pl.Float64)


def read_textgrids(source: Path, split: str) -> pl.DataFrame:
    """Read the intervals of all the TextGrids of a split, without the utterances with an UNKNOWN word.

    The speaker is the first directory after the split, or the file id if the TextGrid is directly in the split.
    """
    parts = pl.col("path").str.split("/")
    return (
        read_files(source, split)
        .select(
            fileid=parts.list.last().str.strip_suffix(".TextGrid"),
            speaker=parts.list.get(1).str.strip_suffix(".TextGrid"),
            line=pl.col("content").str.split("\n"),
        )
        .explode("line")
        .with_columns(pl.col("line").str.strip_chars())
        .with_columns(
            tier=pl.col("line").str.extract(r'^name = "(.*)"$').forward_fill().over("fileid"),
            onset=milliseconds(pl.col("line").shift(2).str.extract(r"^xmin = (.*)$")),
            offset=milliseconds(pl.col("line").shift(1).str.extract(r"^xmax = (.*)$")),
            label=pl.col("line").str.extract(r'^text = "(.*)"$'),
        )
        .filter(pl.col("label").is_not_null(), ~(pl.col("label") == UNKNOWN).any().over("fileid"))
        .select("fileid", "speaker", "tier", "onset", "offset", "label")
    )


def build_words(intervals: pl.LazyFrame) -> pl.LazyFrame:
    """Words whose spelling and transcription are not hapaxes.

    They last at least MIN_DURATION and less than MAX_WORD_DURATION.
    """
    words = intervals.filter(pl.col("tier") == WORDS_TIER, pl.col("label") != "")  # ruff: ignore[compare-to-empty-string]
    phones = intervals.filter(pl.col("tier") == PHONES_TIER, ~pl.col("label").is_in([*SILENCES, PAUSE]))
    return (
        words.filter(pl.len().over("label") > 1)
        .filter(DURATION >= MIN_DURATION, DURATION < MAX_WORD_DURATION)
        .join(phones, on=["fileid", "speaker"], suffix="_phone")
        .filter(pl.col("onset_phone") >= pl.col("onset"), pl.col("offset_phone") <= pl.col("offset"))
        .group_by("fileid", "speaker", "onset", "offset")
        .agg(transcription=pl.col("label_phone").sort_by("onset_phone").str.join(","))
        .filter(pl.len().over("transcription") > 1)
        .sort("fileid", "onset", "offset")
        .select("fileid", "speaker", seconds(pl.col("onset")), seconds(pl.col("offset")), "transcription")
    )


def build_ngrams(intervals: pl.LazyFrame) -> pl.LazyFrame:
    """Sequences of consecutive phones whose transcription is not a hapax.

    They last at least MIN_DURATION and less than MAX_NGRAM_DURATION.
    They do not cross SILENCES, nor a PAUSE longer than MAX_PAUSE.
    Shorter pauses are not in the transcription.
    """
    phones = (
        intervals.filter(pl.col("tier") == PHONES_TIER)
        .sort("fileid", "onset")
        .with_columns(
            chunk=(pl.col("label").is_in(SILENCES) | ((pl.col("label") == PAUSE) & (DURATION > MAX_PAUSE)))
            .cum_sum()
            .over("fileid")
        )
        .filter(~pl.col("label").is_in([*SILENCES, PAUSE]))
        .with_columns(rank=pl.int_range(pl.len()).over("fileid", "chunk"))
    )
    chunks = phones.group_by("fileid", "chunk").agg(labels="label")
    return (
        phones.select("fileid", "speaker", "chunk", "onset", "rank")
        .join(phones.select("fileid", "chunk", "offset", "rank"), on=["fileid", "chunk"])
        .filter(pl.col("rank_right") >= pl.col("rank"), DURATION >= MIN_DURATION, DURATION < MAX_NGRAM_DURATION)
        .join(chunks, on=["fileid", "chunk"])
        .with_columns(
            transcription=pl.col("labels")
            .list.slice(pl.col("rank"), pl.col("rank_right") - pl.col("rank") + 1)
            .list.join(",")
        )
        .filter(pl.len().over("transcription") > 1)
        .sort("fileid", "onset", "offset")
        .select("fileid", "speaker", seconds(pl.col("onset")), seconds(pl.col("offset")), "transcription")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Zip archive or directory with one subdirectory of TextGrids per split",
    )
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev-clean", "dev-other", "test-clean", "test-other"],
        help="Splits to build (default: LibriSpeech dev and test)",
    )
    args = parser.parse_args()
    for split in args.splits:
        intervals = read_textgrids(args.source, split).lazy()
        build_words(intervals).sink_ndjson(args.output / f"{split}-words.jsonl")
        build_ngrams(intervals).sink_ndjson(args.output / f"{split}-ngrams.jsonl")
