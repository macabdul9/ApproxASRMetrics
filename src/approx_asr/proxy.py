"""Proxy-reference selection and proxy (pWER/pCER) features.

A target model's proxy is the best other model by mean WER across datasets. Its
hypothesis stands in for the missing reference transcript.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from approx_asr.config import write_json
from approx_asr.metrics import compute_metrics

# Placeholder used by the paper runs for missing (NaN) hypotheses.
MISSING_HYPOTHESIS = "no_audio"

PROXY_KEYS = (
    "nwSubstitutions",
    "nwDeletions",
    "nwInsertions",
    "nwErrors",
    "WER",
    "ncSubstitutions",
    "ncDeletions",
    "ncInsertions",
    "ncErrors",
    "CER",
)


def select_proxies(
    profile: dict, target_model: str, metric: str = "WER", count: int = 1
) -> list[str]:
    """Return the `count` best models by mean `metric`, never including the target.

    Ties keep the profile's key order. Models without the metric rank last.
    """
    key = f"{metric}_mean_all_datasets"
    candidates = [
        (model_id, scores.get(key, math.inf))
        for model_id, scores in profile.items()
        if model_id != target_model
    ]
    candidates.sort(key=lambda item: item[1])
    return [model_id for model_id, _ in candidates[:count]]


def proxy_metrics(target_hypotheses: pd.Series, proxy_hypotheses: pd.Series) -> pd.DataFrame:
    """Score each target hypothesis against the proxy hypothesis in the same row position.

    Rows are matched by position, not by audio ID, as in the paper runs; both CSVs come
    from the same dataset order. Returns `Ref_<metric>` columns indexed like the target.
    """
    target_hypotheses = target_hypotheses.fillna(MISSING_HYPOTHESIS)
    proxy_hypotheses = proxy_hypotheses.fillna(MISSING_HYPOTHESIS)
    rows = []
    for position in range(len(target_hypotheses)):
        # Kept from the paper runs: every "." in the proxy hypothesis is replaced, not only
        # a lone "." hypothesis. "Done." becomes "Doneno_audio", which normalizes to
        # "doneno audio", so proxies that emit sentence punctuation get inflated
        # pWER/pCER. These features produced the published numbers.
        pseudo_reference = str(proxy_hypotheses.iloc[position]).replace(".", MISSING_HYPOTHESIS)
        hypothesis = str(target_hypotheses.iloc[position])
        scores = compute_metrics(reference=pseudo_reference, hypothesis=hypothesis)
        rows.append({f"Ref_{key}": scores[key] for key in PROXY_KEYS})
    return pd.DataFrame(rows, index=target_hypotheses.index)


def profile_models(features_dir: Path) -> tuple[dict, dict]:
    """Compute per-dataset and mean corpus-level WER/CER for every model directory.

    The mean runs over every CSV in a model's directory, which in the paper covered both
    the benchmark and the in-the-wild subsets.
    """
    summary, per_dataset = {}, {}
    for model_dir in sorted(path for path in features_dir.iterdir() if path.is_dir()):
        per_dataset[model_dir.name] = {}
        for csv_path in sorted(model_dir.glob("*.csv")):
            if "checkpoint" in csv_path.name:
                continue
            frame = pd.read_csv(csv_path)
            per_dataset[model_dir.name][csv_path.name] = {
                "WER": frame["nwErrors"].sum() / frame["wRefLength"].sum() * 100,
                "CER": frame["ncErrors"].sum() / frame["cRefLength"].sum() * 100,
                "nwErrors": float(frame["nwErrors"].mean()),
                "ncErrors": float(frame["ncErrors"].mean()),
            }
        scores = pd.DataFrame(per_dataset[model_dir.name].values())
        summary[model_dir.name] = {
            f"{metric}_mean_all_datasets": round(float(scores[metric].mean()), 2)
            for metric in ("WER", "CER", "nwErrors", "ncErrors")
        }
    return summary, per_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile models by mean WER/CER to choose proxy references"
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/model_profile.json"))
    parser.add_argument("--per-dataset-output", type=Path, default=None)
    args = parser.parse_args()
    summary, per_dataset = profile_models(args.features_dir)
    write_json(args.output, summary)
    if args.per_dataset_output:
        write_json(args.per_dataset_output, per_dataset)


if __name__ == "__main__":
    main()
