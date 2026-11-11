"""Turn regression outputs into the paper's tables.

- `average-wild`: average each in-the-wild prediction over the leave-one-benchmark-out folds.
- `summarize`: corpus-level WER/CER, aWER/aCER, and MAEs per model and dataset.
- `ablation`: MAE mean and std over CV folds for ablation and eWER3 runs (Table 3).
- `compare-proxy`: regression vs. scoring against the proxy hypothesis directly (Table 4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from approx_asr.config import write_json

PREDICTION_COLUMNS = ["aWER", "aCER", "anwErrors", "ancErrors"]


def average_wild(root_dir: Path, columns: list[str]) -> None:
    """Write `<model>/<wild>_predicted.csv` with predictions averaged over fold folders."""
    for model_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        fold_dirs = sorted(path for path in model_dir.iterdir() if path.is_dir())
        if not fold_dirs:
            continue
        for wild_csv in sorted(fold_dirs[0].glob("*.csv")):
            frames = [pd.read_csv(fold / wild_csv.name) for fold in fold_dirs]
            averaged = frames[0].copy()
            averaged[columns] = sum(frame[columns] for frame in frames) / len(frames)
            averaged.to_csv(model_dir / wild_csv.name, index=False)


def corpus_metrics(frame: pd.DataFrame) -> dict:
    """Corpus-level rates and per-utterance MAEs, with negative predictions clipped to zero.

    aWER/aCER are predicted error counts normalized by the total reference length.
    MAE_WER/MAE_CER compare the per-utterance WER/CER regressors with the true rates.
    """
    frame = frame.copy()
    frame[PREDICTION_COLUMNS] = frame[PREDICTION_COLUMNS].clip(lower=0)
    words, chars = frame["wRefLength"].sum(), frame["cRefLength"].sum()
    return {
        "WER": frame["nwErrors"].sum() / words * 100,
        "CER": frame["ncErrors"].sum() / chars * 100,
        "aWER": frame["anwErrors"].sum() / words * 100,
        "aCER": frame["ancErrors"].sum() / chars * 100,
        "MAE_WER": mean_absolute_error(frame["WER"], frame["aWER"]),
        "MAE_CER": mean_absolute_error(frame["CER"], frame["aCER"]),
        "nwErrors": float(frame["nwErrors"].mean()),
        "ncErrors": float(frame["ncErrors"].mean()),
        "anwErrors": float(frame["anwErrors"].mean()),
        "ancErrors": float(frame["ancErrors"].mean()),
        "MAE_nwErrors": mean_absolute_error(frame["nwErrors"], frame["anwErrors"]),
        "MAE_ncErrors": mean_absolute_error(frame["ncErrors"], frame["ancErrors"]),
    }


def bootstrap(
    frame: pd.DataFrame, n_samples: int, iterations: int, independent: bool, seed: int = 42
) -> dict:
    """Mean and std of `corpus_metrics` over resamples of `n_samples` rows with replacement.

    By default every iteration uses `random_state=42`, as in the paper's analysis
    script, so all iterations draw the same resample: the reported mean is one
    500-row resample and the std is 0. `independent=True` draws a different resample
    per iteration.
    """
    if n_samples > len(frame):
        raise ValueError(f"Cannot draw {n_samples} rows from {len(frame)}")
    draws = [
        corpus_metrics(
            frame.sample(n=n_samples, replace=True, random_state=seed + i if independent else seed)
        )
        for i in range(iterations)
    ]
    draws = pd.DataFrame(draws)
    return {metric: {"mean": draws[metric].mean(), "std": draws[metric].std()} for metric in draws}


def summarize(root_dir: Path, output_dir: Path, n_samples: int, iterations: int, independent: bool):
    nested, rows = {}, []
    for model_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            dataset = csv_path.stem.removesuffix("_predicted")
            stats = bootstrap(pd.read_csv(csv_path), n_samples, iterations, independent)
            nested.setdefault(model_dir.name, {})[dataset] = stats
            row = {"model": model_dir.name, "dataset": dataset}
            for metric, values in stats.items():
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_std"] = values["std"]
            rows.append(row)
    write_json(output_dir / "summary.json", nested)
    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)


def summarize_ablation(root_dir: Path, output: Path) -> None:
    """Expects `<experiment>/<source_|target_model>/<metric>/fold_<k>/<dataset>.csv`."""
    results = {}
    for experiment_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        for model_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            model_results = {}
            for metric_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                metric = metric_dir.name
                fold_dirs = sorted(path for path in metric_dir.iterdir() if path.is_dir())
                for csv_path in sorted(fold_dirs[0].glob("*.csv")):
                    maes = []
                    for fold_dir in fold_dirs:
                        frame = pd.read_csv(fold_dir / csv_path.name)
                        predictions = np.maximum(0, frame[f"a{metric}"])
                        maes.append(mean_absolute_error(frame[metric], predictions))
                    model_results.setdefault(csv_path.stem, {})[metric] = {
                        "mean": float(np.mean(maes)),
                        "std": float(np.std(maes)),
                    }
            results.setdefault(experiment_dir.name, {})[model_dir.name] = model_results
    write_json(output, results)


def compare_proxy(root_dir: Path, metric: str, output: Path) -> pd.DataFrame:
    """MAE of the regression prediction vs. the proxy-scored metric, against the true metric.

    The direct baseline scores the target hypothesis against the proxy hypothesis
    (`Ref_<metric>`) and uses that as the estimate, with no regression.
    """
    rows = []
    for model_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path)
            rows.append(
                {
                    "model": model_dir.name,
                    "dataset": csv_path.stem.removesuffix("_predicted"),
                    "mae_regression": mean_absolute_error(
                        frame[metric], frame[f"a{metric}"].clip(lower=0)
                    ),
                    "mae_proxy": mean_absolute_error(frame[metric], frame[f"Ref_{metric}"]),
                }
            )
    table = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    wild = commands.add_parser("average-wild", help="Average in-the-wild predictions over folds")
    wild.add_argument("--root-dir", type=Path, default=Path("runs/regression"))
    wild.add_argument("--columns", nargs="+", default=PREDICTION_COLUMNS)

    summary = commands.add_parser("summarize", help="Corpus-level metrics per model and dataset")
    summary.add_argument("--root-dir", type=Path, default=Path("runs/regression"))
    summary.add_argument("--output-dir", type=Path, default=Path("runs/results"))
    summary.add_argument("--n-samples", type=int, default=500)
    summary.add_argument("--iterations", type=int, default=100)
    summary.add_argument(
        "--independent-resamples",
        action="store_true",
        help="Draw a new resample per iteration (the paper analysis reused one; see README)",
    )

    ablation = commands.add_parser("ablation", help="Fold mean/std MAE for ablation-style runs")
    ablation.add_argument("--root-dir", type=Path, default=Path("runs/ablation"))
    ablation.add_argument("--output", type=Path, default=Path("runs/results/ablation.json"))

    proxy = commands.add_parser("compare-proxy", help="Regression vs. direct proxy scoring")
    proxy.add_argument("--root-dir", type=Path, default=Path("runs/regression"))
    proxy.add_argument("--metric", default="nwErrors")
    proxy.add_argument("--output", type=Path, default=Path("runs/results/compare_proxy.csv"))

    args = parser.parse_args()
    if args.command == "average-wild":
        average_wild(args.root_dir, args.columns)
    elif args.command == "summarize":
        summarize(
            args.root_dir,
            args.output_dir,
            args.n_samples,
            args.iterations,
            args.independent_resamples,
        )
    elif args.command == "ablation":
        summarize_ablation(args.root_dir, args.output)
    else:
        table = compare_proxy(args.root_dir, args.metric, args.output)
        print(table.groupby("dataset")[["mae_regression", "mae_proxy"]].mean().to_string())


if __name__ == "__main__":
    main()
