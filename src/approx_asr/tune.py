"""Randomized hyperparameter search for the stacked regressor (5-fold CV)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from approx_asr.config import load_json, write_json
from approx_asr.experiments import model_ids_or_all
from approx_asr.features import load_with_proxy
from approx_asr.proxy import select_proxies
from approx_asr.regressor import build_regressor

logger = logging.getLogger(__name__)


def tuning_data(args: argparse.Namespace) -> pd.DataFrame:
    """Pool every model's tuning split, each with its own proxy features."""
    profile = load_json(args.model_profile)
    benchmarks = list(load_json(args.benchmark_config))
    frames = []
    for model_id in model_ids_or_all(args.model_ids, args.data_dir):
        proxy_id = select_proxies(profile, model_id, metric=args.metric)[0]
        logger.info("model=%s proxy=%s", model_id, proxy_id)
        frames.extend(load_with_proxy(args.data_dir, benchmarks, model_id, proxy_id).values())
    return pd.concat(frames).dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("runs/splits/tuning"))
    parser.add_argument("--model-ids", nargs="+", default=None, help="Default: every model")
    parser.add_argument("--metric", default="WER", help="Target metric; also ranks proxies")
    parser.add_argument("--search-space", type=Path, default=Path("configs/search_space.json"))
    parser.add_argument(
        "--scoring",
        default="neg_mean_absolute_error",
        choices=["neg_mean_absolute_error", "neg_mean_squared_error"],
    )
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--model-profile", type=Path, default=Path("configs/model_profile.json"))
    parser.add_argument(
        "--benchmark-config", type=Path, default=Path("configs/datasets_bench.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tuning"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data = tuning_data(args)
    search = RandomizedSearchCV(
        estimator=build_regressor(scoring=args.scoring),
        param_distributions=load_json(args.search_space),
        n_iter=args.n_iter,
        cv=5,
        scoring=args.scoring,
        random_state=42,
        n_jobs=args.n_jobs,
        verbose=2,
    )
    search.fit(data[["similarity", f"Ref_{args.metric}"]].to_numpy(), data[args.metric].to_numpy())
    logger.info("best_params=%s best_cv_score=%.4f", search.best_params_, -search.best_score_)
    output = args.output_dir / f"{args.metric}_{args.scoring}_best_params.json"
    write_json(output, search.best_params_)
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()
