from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICS = ("WER", "CER", "nwErrors", "ncErrors")


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n")


def load_dataset_configs(paths: list[str | Path]) -> dict:
    """Merge several dataset configs, keeping file order."""
    merged = {}
    for path in paths:
        merged.update(load_json(path))
    return merged


def dataset_id(name_or_path: str) -> str:
    """Return the short dataset name used for per-model CSV files.

    `macabdul9/Primock57_eval_1K` and `data/subsets/Primock57_eval_1K` both map to
    `Primock57_eval_1K`.
    """
    return name_or_path.rstrip("/").split("/")[-1]


def split_metrics(value: str) -> list[str]:
    return [metric for metric in value.split(",") if metric]
