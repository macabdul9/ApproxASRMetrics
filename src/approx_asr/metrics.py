"""Per-utterance WER/CER and error counts, and the CLI that adds them to ASR output CSVs."""

from __future__ import annotations

import argparse
import logging
import os
import re
import unicodedata
from pathlib import Path

import jiwer
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "WER",
    "CER",
    "nwErrors",
    "wRefLength",
    "ncErrors",
    "cRefLength",
    "nwSubstitutions",
    "nwDeletions",
    "nwInsertions",
    "ncSubstitutions",
    "ncDeletions",
    "ncInsertions",
)


def normalize_text(text: str) -> str:
    """Whisper's BasicTextNormalizer with default settings.

    Copied from `transformers.models.whisper.english_normalizer` so that scoring does not
    depend on importing transformers. Lowercases, drops bracketed and parenthesized spans,
    and replaces Unicode marks, symbols, and punctuation with spaces.
    """
    text = text.lower()
    text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
    text = re.sub(r"\(([^)]+?)\)", "", text)
    text = "".join(
        " " if unicodedata.category(char)[0] in "MSP" else char
        for char in unicodedata.normalize("NFKC", text)
    ).lower()
    return re.sub(r"\s+", " ", text)


def compute_metrics(reference: str, hypothesis: str) -> dict:
    """WER and CER in percent, with substitution/deletion/insertion counts.

    Both strings are normalized first. Reference lengths are counted after normalization.
    """
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    words = jiwer.process_words(reference=reference, hypothesis=hypothesis)
    chars = jiwer.process_characters(reference=reference, hypothesis=hypothesis)
    return {
        "WER": words.wer * 100,
        "CER": chars.cer * 100,
        "nwSubstitutions": words.substitutions,
        "nwInsertions": words.insertions,
        "nwDeletions": words.deletions,
        "nwErrors": words.substitutions + words.insertions + words.deletions,
        "ncErrors": chars.substitutions + chars.insertions + chars.deletions,
        "ncSubstitutions": chars.substitutions,
        "ncInsertions": chars.insertions,
        "ncDeletions": chars.deletions,
        "wRefLength": len(words.references[0]),
        "cRefLength": len(chars.references[0]),
    }


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add metric columns; rows with a missing reference or hypothesis get NaN."""
    rows = []
    for reference, hypothesis in zip(frame["reference"], frame["hypothesis"], strict=True):
        if pd.isnull(reference) or pd.isnull(hypothesis):
            rows.append({})
        else:
            rows.append(compute_metrics(reference, hypothesis))
    scores = pd.DataFrame(rows, index=frame.index, columns=list(METRIC_COLUMNS))
    return pd.concat([frame.drop(columns=list(METRIC_COLUMNS), errors="ignore"), scores], axis=1)


def score_directory(root_dir: Path) -> None:
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if not filename.endswith(".csv") or "checkpoint" in filename:
                continue
            path = Path(dirpath) / filename
            frame = pd.read_csv(path)
            if not {"reference", "hypothesis"} <= set(frame.columns):
                logger.warning("Skipping %s: no reference/hypothesis columns", path)
                continue
            tqdm.write(f"Scoring {path}")
            score_frame(frame).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add WER/CER and error counts to every model/dataset.csv under a directory "
        "(files are updated in place)"
    )
    parser.add_argument("--root-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    score_directory(args.root_dir)


if __name__ == "__main__":
    main()
