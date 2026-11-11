"""Build the 1K-utterance evaluation subsets, and load them for later stages.

Each subset keeps 1-30 s utterances whose normalized reference is non-empty, takes
1,000 examples after shuffling with seed 42, resamples audio to 16 kHz, and keeps
`audio`, `reference`, and an `id` column of the form `<name>_<index>`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from approx_asr.config import load_json
from approx_asr.metrics import normalize_text

logger = logging.getLogger(__name__)


def load_subset(name_or_path: str, entry: dict):
    """Load a subset from the Hub or from a directory written by `build`."""
    import datasets

    split = entry.get("split", "test")
    if Path(name_or_path).is_dir():
        return datasets.load_from_disk(name_or_path)[split]
    return datasets.load_dataset(
        name_or_path, entry.get("config", "default"), split=split, trust_remote_code=True
    )


def duration(example: dict) -> float:
    return example["audio"]["array"].shape[-1] / example["audio"]["sampling_rate"]


def resample(example: dict, target_sr: int) -> dict:
    import librosa

    example["audio"]["array"] = librosa.resample(
        y=example["audio"]["array"], orig_sr=example["audio"]["sampling_rate"], target_sr=target_sr
    )
    example["audio"]["sampling_rate"] = target_sr
    return example


def subset_name(source_id: str, config: str) -> str:
    name = source_id.split("/")[-1]
    if config not in ("default", "test"):
        name = f"{name}_{config}"
    return f"{name}_eval_1K"


def build_subset(source_id: str, entry: dict, args: argparse.Namespace):
    import datasets

    config = entry["config"]
    text_column = entry["text_column"]
    dataset = datasets.load_dataset(source_id, config, split=entry["split"], trust_remote_code=True)
    logger.info("Loaded %s: %d examples", source_id, len(dataset))
    dataset = dataset.filter(
        lambda x: args.min_duration <= duration(x) <= args.max_duration, num_proc=args.num_proc
    )
    dataset = dataset.filter(
        lambda x: len(normalize_text(x[text_column])) > 0, num_proc=args.num_proc
    )
    logger.info("After filtering: %d examples", len(dataset))
    # The paper subsets were drawn after two consecutive shuffles with the same seed.
    dataset = dataset.shuffle(seed=42).shuffle(seed=42)
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = dataset.rename_column(text_column, "reference")
    dataset = dataset.remove_columns(
        [column for column in dataset.column_names if column not in ("audio", "reference")]
    )
    prefix = subset_name(source_id, config).removesuffix("_eval_1K")
    dataset = dataset.add_column("id", [f"{prefix}_{i}" for i in range(len(dataset))])
    dataset = dataset.map(lambda x: resample(x, args.target_sr), num_proc=args.num_proc)
    return datasets.DatasetDict({"test": dataset})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("configs/source_datasets.json"),
        help="JSON mapping source dataset IDs to config, split, and text_column",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/subsets"))
    parser.add_argument(
        "--push-to-hub",
        metavar="NAMESPACE",
        default=None,
        help="Also push each subset to NAMESPACE/<name>_eval_1K as a private dataset",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--target-sr", type=int, default=16000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    for source_id, entry in load_json(args.source_config).items():
        name = subset_name(source_id, entry["config"])
        subset = build_subset(source_id, entry, args)
        subset.save_to_disk(args.output_dir / name)
        logger.info("Saved %s", args.output_dir / name)
        if args.push_to_hub:
            subset.push_to_hub(f"{args.push_to_hub}/{name}", private=True)


if __name__ == "__main__":
    main()
