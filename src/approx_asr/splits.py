"""Split each benchmark CSV into training (90%) and hyperparameter-tuning (10%) parts.

In-the-wild datasets are copied whole into the training directory; they are only ever
used as test sets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from approx_asr.config import dataset_id, load_json


def split_outputs(features_dir: Path, train_dir: Path, tuning_dir: Path, keep_whole: set[str]):
    for model_dir in sorted(path for path in features_dir.iterdir() if path.is_dir()):
        (train_dir / model_dir.name).mkdir(parents=True, exist_ok=True)
        (tuning_dir / model_dir.name).mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(model_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path)
            if csv_path.stem in keep_whole:
                frame.to_csv(train_dir / model_dir.name / csv_path.name, index=False)
                continue
            train, tuning = train_test_split(frame, test_size=0.1, random_state=42)
            train.to_csv(train_dir / model_dir.name / csv_path.name, index=False)
            tuning.to_csv(tuning_dir / model_dir.name / csv_path.name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features-dir", type=Path, default=Path("runs/features"))
    parser.add_argument("--train-dir", type=Path, default=Path("runs/splits/train"))
    parser.add_argument("--tuning-dir", type=Path, default=Path("runs/splits/tuning"))
    parser.add_argument(
        "--keep-whole-config",
        type=Path,
        default=Path("configs/datasets_wild.json"),
        help="Datasets copied without splitting",
    )
    args = parser.parse_args()
    keep_whole = {dataset_id(name) for name in load_json(args.keep_whole_config)}
    split_outputs(args.features_dir, args.train_dir, args.tuning_dir, keep_whole)


if __name__ == "__main__":
    main()
