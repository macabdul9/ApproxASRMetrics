"""Feature and proxy ablations with k-fold CV, plus transfer to an unseen target model (Table 3).

A regressor is trained on one source model's pooled benchmark data and evaluated on
held-out folds of the same model (IID), the same folds transcribed by another model
(OOD model), and the in-the-wild sets for both models.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from approx_asr.config import load_json, split_metrics
from approx_asr.experiments import check_disjoint, model_ids_or_all
from approx_asr.features import load_with_proxies
from approx_asr.proxy import select_proxies
from approx_asr.regressor import regressor_for_metric

logger = logging.getLogger(__name__)


def select_features(
    columns: list[str],
    target_metric: str,
    metrics: list[str],
    use_similarity: bool,
    num_proxies: int,
    mean_of_proxies: bool,
    proxy_features: str,
) -> list[str]:
    """Choose regression inputs for one target metric.

    `all-metrics` reproduces the original ablation script: the proxy columns of every
    metric in `metrics` are used as inputs for every target (e.g. both Ref_nwErrors and
    Ref_ncErrors when predicting nwErrors). `per-metric` uses only the target metric's
    proxy columns, as in the later rebuttal runs.
    """
    selected = ["similarity"] if use_similarity else []
    if num_proxies == 0:
        return selected
    feature_metrics = metrics if proxy_features == "all-metrics" else [target_metric]
    for metric in feature_metrics:
        if mean_of_proxies:
            selected.append(f"Ref_{metric}_mean")
        else:
            selected.extend(column for column in columns if column.startswith(f"Ref_{metric}_Ref"))
    return selected


def prediction_frame(frame: pd.DataFrame, metric: str, predictions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "audio_id": frame["audio_id"].to_numpy(),
            metric: frame[metric].to_numpy(),
            f"a{metric}": predictions,
            "wRefLength": frame["wRefLength"].to_numpy(),
            "cRefLength": frame["cRefLength"].to_numpy(),
        }
    )


def run_for_target(args: argparse.Namespace, ood_model_id: str) -> None:
    profile = load_json(args.model_profile)
    benchmarks = list(load_json(args.benchmark_config))
    wild = list(load_json(args.wild_config))
    hyperparameters = load_json(args.hyperparameters)
    check_disjoint(benchmarks, wild)
    metrics = split_metrics(args.metrics)
    rng = np.random.RandomState(args.seed)

    experiment = (
        f"sim_{args.use_similarity}_prc_{args.proxy_ref_count}"
        f"_mean_{args.mean_of_proxies}_ref_{args.ref_model_id}"
    )
    if args.proxy_features == "per-metric":
        experiment += "_per-metric"
    experiment_dir = args.output_dir / experiment
    logger.info("experiment=%s ood_model=%s", experiment, ood_model_id)

    for model_id in model_ids_or_all(args.model_ids, args.data_dir):
        if args.ref_model_id:
            proxy_ids = [args.ref_model_id]
        else:
            proxy_ids = select_proxies(
                profile, model_id, metric=args.proxy_metric, count=args.proxy_ref_count
            )

        def load(dataset_names, target, proxy_ids=proxy_ids):
            return load_with_proxies(
                args.data_dir, dataset_names, target, proxy_ids, args.mean_of_proxies
            )

        source_wild = load(wild, model_id)
        target_wild = load(wild, ood_model_id)
        source = pd.concat(load(benchmarks, model_id).values()).dropna()
        target = pd.concat(load(benchmarks, ood_model_id).values()).dropna()

        common_ids = set(source["audio_id"]) & set(target["audio_id"])
        source = source[source["audio_id"].isin(common_ids)]
        target = target[target["audio_id"].isin(common_ids)]
        permutation = rng.permutation(len(source))
        source = source.iloc[permutation].reset_index(drop=True)
        target = target.iloc[permutation].reset_index(drop=True)
        if not (source["audio_id"] == target["audio_id"]).all():
            raise ValueError(f"{model_id} and {ood_model_id} outputs are not in the same order")

        folds = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        for metric in metrics:
            columns = select_features(
                list(source.columns),
                metric,
                metrics,
                args.use_similarity,
                args.proxy_ref_count,
                args.mean_of_proxies,
                args.proxy_features,
            )
            logger.info("metric=%s features=%s", metric, columns)
            for fold, (train_index, test_index) in enumerate(folds.split(source[columns])):
                source_dir = experiment_dir / f"source_{model_id}" / metric / f"fold_{fold}"
                target_dir = experiment_dir / f"target_{ood_model_id}" / metric / f"fold_{fold}"
                source_dir.mkdir(parents=True, exist_ok=True)
                target_dir.mkdir(parents=True, exist_ok=True)

                pipeline = regressor_for_metric(
                    metric, hyperparameters, args.apply_tuned_hyperparameters
                )
                train = source.iloc[train_index]
                pipeline.fit(train[columns], train[metric])

                for frame, output_dir, name in (
                    (source, source_dir, model_id),
                    (target, target_dir, ood_model_id),
                ):
                    test = frame.iloc[test_index]
                    predictions = pipeline.predict(test[columns])
                    logger.info(
                        "model=%s fold=%d metric=%s mae=%.4f",
                        name,
                        fold,
                        metric,
                        mean_absolute_error(test[metric], predictions),
                    )
                    prediction_frame(test, metric, predictions).to_csv(
                        output_dir / "iid_test.csv", index=False
                    )
                for wild_sets, output_dir in ((source_wild, source_dir), (target_wild, target_dir)):
                    for wild_name, wild_frame in wild_sets.items():
                        predictions = pipeline.predict(wild_frame[columns])
                        prediction_frame(wild_frame, metric, predictions).to_csv(
                            output_dir / f"{wild_name}.csv", index=False
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("runs/splits/train"))
    parser.add_argument("--model-ids", nargs="+", default=None, help="Source models")
    parser.add_argument(
        "--ood-model-ids", nargs="+", required=True, help="Target models to transfer to"
    )
    parser.add_argument(
        "--ref-model-id", default=None, help="Fixed proxy; default selects by --proxy-metric"
    )
    parser.add_argument("--proxy-metric", default="WER")
    parser.add_argument("--metrics", default="nwErrors,ncErrors")
    parser.add_argument("--use-similarity", action="store_true")
    parser.add_argument("--proxy-ref-count", type=int, default=1)
    parser.add_argument(
        "--mean-of-proxies",
        action="store_true",
        help="Average proxy features over the selected proxies instead of one column each",
    )
    parser.add_argument(
        "--proxy-features",
        choices=["all-metrics", "per-metric"],
        default="all-metrics",
        help="all-metrics matches the original ablation script (see README)",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-profile", type=Path, default=Path("configs/model_profile.json"))
    parser.add_argument(
        "--benchmark-config", type=Path, default=Path("configs/datasets_bench.json")
    )
    parser.add_argument("--wild-config", type=Path, default=Path("configs/datasets_wild.json"))
    parser.add_argument(
        "--hyperparameters", type=Path, default=Path("configs/hyperparameters.json")
    )
    parser.add_argument("--apply-tuned-hyperparameters", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ablation"))
    args = parser.parse_args()
    if args.proxy_ref_count == 0 and not args.use_similarity:
        parser.error("--proxy-ref-count 0 needs --use-similarity; there would be no features")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for ood_model_id in args.ood_model_ids:
        run_for_target(args, ood_model_id)


if __name__ == "__main__":
    main()
