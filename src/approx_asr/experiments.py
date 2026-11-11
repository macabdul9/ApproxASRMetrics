"""Main regression experiments: leave-one-benchmark-out, training-size scaling, cross-lingual.

Each target metric is predicted from two features: the SONAR similarity between the
audio and the target hypothesis, and the same metric computed against the proxy
hypothesis (`Ref_<metric>`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from approx_asr.config import load_json, split_metrics, write_json
from approx_asr.features import load_with_proxy
from approx_asr.proxy import select_proxies
from approx_asr.regressor import build_regressor, regressor_for_metric

logger = logging.getLogger(__name__)


def feature_columns(metric: str) -> list[str]:
    return ["similarity", f"Ref_{metric}"]


def model_ids_or_all(model_ids: list[str] | None, data_dir: Path) -> list[str]:
    if model_ids:
        return model_ids
    return sorted(path.name for path in data_dir.iterdir() if path.is_dir())


def check_disjoint(benchmarks: list[str], wild: list[str]) -> None:
    overlap = set(benchmarks) & set(wild)
    if overlap:
        raise ValueError(f"Benchmark and in-the-wild configs overlap: {sorted(overlap)}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Leave one benchmark out per model, and predict the in-the-wild sets from each fold.

    Training uses the other benchmarks' training splits plus their tuning splits. The
    held-out benchmark is scored on its training split only, so tuning rows never reach
    a test set.
    """
    profile = load_json(args.model_profile)
    benchmarks = list(load_json(args.benchmark_config))
    wild = list(load_json(args.wild_config))
    hyperparameters = load_json(args.hyperparameters)
    check_disjoint(benchmarks, wild)
    metrics = split_metrics(args.metrics)

    for model_id in model_ids_or_all(args.model_ids, args.data_dir):
        proxy_id = select_proxies(profile, model_id, metric=args.proxy_metric)[0]
        logger.info("model=%s proxy=%s", model_id, proxy_id)
        wild_sets = load_with_proxy(args.data_dir, wild, model_id, proxy_id)
        train_sets = load_with_proxy(args.data_dir, benchmarks, model_id, proxy_id)
        tuning_sets = load_with_proxy(args.tuning_dir, benchmarks, model_id, proxy_id)
        output_dir = args.output_dir / model_id

        for held_out in train_sets:
            test_frame = train_sets[held_out].dropna()
            train_frame = pd.concat(
                [frame for name, frame in train_sets.items() if name != held_out]
                + [frame for name, frame in tuning_sets.items() if name != held_out]
            ).dropna()
            logger.info("held_out=%s train=%d test=%d", held_out, len(train_frame), len(test_frame))

            for metric in metrics:
                columns = feature_columns(metric)
                pipeline = regressor_for_metric(
                    metric, hyperparameters, args.apply_tuned_hyperparameters
                )
                pipeline.fit(train_frame[columns], train_frame[metric])
                test_frame[f"a{metric}"] = pipeline.predict(test_frame[columns])
                logger.info(
                    "model=%s held_out=%s metric=%s mae=%.4f",
                    model_id,
                    held_out,
                    metric,
                    mean_absolute_error(test_frame[metric], test_frame[f"a{metric}"]),
                )
                for wild_name, wild_frame in wild_sets.items():
                    wild_frame[f"a{metric}"] = pipeline.predict(wild_frame[columns])
                    logger.info(
                        "model=%s held_out=%s metric=%s wild=%s mae=%.4f",
                        model_id,
                        held_out,
                        metric,
                        wild_name,
                        mean_absolute_error(wild_frame[metric], wild_frame[f"a{metric}"]),
                    )

            output_dir.mkdir(parents=True, exist_ok=True)
            test_frame.to_csv(output_dir / f"{held_out}_predicted.csv", index=False)
            fold_dir = output_dir / held_out
            fold_dir.mkdir(exist_ok=True)
            for wild_name, wild_frame in wild_sets.items():
                wild_frame.to_csv(fold_dir / f"{wild_name}_predicted.csv", index=False)


def run_scaling(args: argparse.Namespace) -> None:
    """Train on N pooled benchmark utterances and evaluate on the in-the-wild sets (Fig. 3)."""
    profile = load_json(args.model_profile)
    benchmarks = list(load_json(args.benchmark_config))
    wild = list(load_json(args.wild_config))
    hyperparameters = load_json(args.hyperparameters)
    check_disjoint(benchmarks, wild)
    metrics = split_metrics(args.metrics)

    for model_id in model_ids_or_all(args.model_ids, args.data_dir):
        proxy_id = select_proxies(profile, model_id, metric=args.proxy_metric)[0]
        wild_sets = load_with_proxy(args.data_dir, wild, model_id, proxy_id)
        pooled = pd.concat(
            list(load_with_proxy(args.data_dir, benchmarks, model_id, proxy_id).values())
            + list(load_with_proxy(args.tuning_dir, benchmarks, model_id, proxy_id).values())
        )
        for num_samples in args.num_train_samples:
            train_frame = (
                pooled.sample(frac=1, random_state=args.seed)
                .reset_index(drop=True)
                .sample(n=min(num_samples, len(pooled)), random_state=args.seed)
                .dropna()
            )
            logger.info("model=%s train=%d", model_id, len(train_frame))
            for metric in metrics:
                columns = feature_columns(metric)
                pipeline = regressor_for_metric(
                    metric, hyperparameters, args.apply_tuned_hyperparameters
                )
                pipeline.fit(train_frame[columns], train_frame[metric])
                for wild_name, wild_frame in wild_sets.items():
                    wild_frame[f"a{metric}"] = pipeline.predict(wild_frame[columns])
                    logger.info(
                        "model=%s n=%d metric=%s wild=%s mae=%.4f",
                        model_id,
                        num_samples,
                        metric,
                        wild_name,
                        mean_absolute_error(wild_frame[metric], wild_frame[f"a{metric}"]),
                    )
            output_dir = args.output_dir / model_id / str(num_samples)
            output_dir.mkdir(parents=True, exist_ok=True)
            for wild_name, wild_frame in wild_sets.items():
                wild_frame.to_csv(output_dir / f"{wild_name}_predicted.csv", index=False)


def run_multilingual(args: argparse.Namespace) -> None:
    """Train on one dataset and test on each other dataset with a fixed proxy (Table 2)."""
    datasets = list(load_json(args.dataset_config))
    data = load_with_proxy(args.data_dir, datasets, args.model_id, args.proxy_model_id)
    output_dir = args.output_dir / args.model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for train_name, train_frame in data.items():
        train_frame = train_frame.dropna()
        results[train_name] = {}
        for metric in split_metrics(args.metrics):
            columns = feature_columns(metric)
            pipeline = build_regressor()
            pipeline.fit(train_frame[columns], train_frame[metric])
            results[train_name][metric] = {}
            for test_name, test_frame in data.items():
                if test_name == train_name:
                    continue
                test_frame = test_frame.dropna()
                predictions = pipeline.predict(test_frame[columns])
                test_frame[f"predicted_{metric}"] = predictions
                test_frame.to_csv(
                    output_dir / f"{train_name}_to_{test_name}_predicted.csv", index=False
                )
                mae = mean_absolute_error(test_frame[metric], predictions)
                results[train_name][metric][test_name] = mae
                logger.info(
                    "train=%s test=%s metric=%s mae=%.4f", train_name, test_name, metric, mae
                )
    output = args.output_dir / f"source_{args.model_id}_proxy_{args.proxy_model_id}.json"
    write_json(output, {args.model_id: results})
    print(json.dumps(results, indent=2))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("runs/splits/train"))
    parser.add_argument("--tuning-dir", type=Path, default=Path("runs/splits/tuning"))
    parser.add_argument("--model-ids", nargs="+", default=None, help="Default: every model")
    parser.add_argument("--metrics", default="WER,CER,nwErrors,ncErrors")
    parser.add_argument("--proxy-metric", default="WER", help="Metric used to rank proxies")
    parser.add_argument("--model-profile", type=Path, default=Path("configs/model_profile.json"))
    parser.add_argument(
        "--benchmark-config", type=Path, default=Path("configs/datasets_bench.json")
    )
    parser.add_argument("--wild-config", type=Path, default=Path("configs/datasets_wild.json"))
    parser.add_argument(
        "--hyperparameters", type=Path, default=Path("configs/hyperparameters.json")
    )
    parser.add_argument(
        "--apply-tuned-hyperparameters",
        action="store_true",
        help="Apply the tuned values. The paper numbers use the defaults (see README).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser("benchmark", help="Leave-one-benchmark-out and in-the-wild")
    add_common_arguments(benchmark)
    benchmark.add_argument("--output-dir", type=Path, default=Path("runs/regression"))
    benchmark.set_defaults(run=run_benchmark)

    scaling = commands.add_parser("scaling", help="Vary the number of training utterances")
    add_common_arguments(scaling)
    scaling.add_argument(
        "--num-train-samples",
        type=int,
        nargs="+",
        default=list(range(1000, 10001, 1000)),
    )
    scaling.add_argument("--seed", type=int, default=42)
    scaling.add_argument("--output-dir", type=Path, default=Path("runs/scaling"))
    scaling.set_defaults(run=run_scaling)

    multilingual = commands.add_parser("multilingual", help="Cross-lingual transfer")
    multilingual.add_argument("--data-dir", type=Path, required=True)
    multilingual.add_argument("--model-id", required=True)
    multilingual.add_argument("--proxy-model-id", required=True)
    multilingual.add_argument("--metrics", default="nwErrors")
    multilingual.add_argument(
        "--dataset-config", type=Path, default=Path("configs/datasets_de_en.json")
    )
    multilingual.add_argument("--output-dir", type=Path, default=Path("runs/multilingual"))
    multilingual.set_defaults(run=run_multilingual)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.run(args)


if __name__ == "__main__":
    main()
