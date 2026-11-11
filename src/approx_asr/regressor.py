"""The stacked regressor that maps (similarity, proxy metric) features to an error metric."""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_PARAMS = {
    "rf__n_estimators": 300,
    "rf__max_depth": 10,
    "rf__min_samples_split": 5,
    "gbr__n_estimators": 300,
    "gbr__learning_rate": 0.01,
    "gbr__max_depth": 5,
    "gbr__min_impurity_decrease": 0.01,
    "hgb__max_iter": 300,
    "hgb__learning_rate": 0.01,
    "hgb__max_depth": 5,
    "hgb__loss": "poisson",
}


def _clip_negative(y):
    return np.maximum(y, 0)


def _identity(y):
    return y


def build_regressor(
    tuned_params: dict | None = None, scoring: str = "neg_mean_absolute_error"
) -> Pipeline:
    """Standardize features, then stack RF, GBR, and Poisson HGB under a non-negative Ridge.

    `tuned_params` uses pipeline keys such as `stack__rf__n_estimators`, as written by
    `approx_asr.tune` and stored in `configs/hyperparameters.json`.
    """
    p = DEFAULT_PARAMS
    base_estimators = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=p["rf__n_estimators"],
                max_depth=p["rf__max_depth"],
                min_samples_split=p["rf__min_samples_split"],
                random_state=42,
            ),
        ),
        (
            "gbr",
            GradientBoostingRegressor(
                n_estimators=p["gbr__n_estimators"],
                learning_rate=p["gbr__learning_rate"],
                max_depth=p["gbr__max_depth"],
                min_impurity_decrease=p["gbr__min_impurity_decrease"],
                random_state=42,
            ),
        ),
        (
            "hgb",
            HistGradientBoostingRegressor(
                max_iter=p["hgb__max_iter"],
                learning_rate=p["hgb__learning_rate"],
                max_depth=p["hgb__max_depth"],
                loss=p["hgb__loss"],
                random_state=42,
            ),
        ),
    ]
    ridge = GridSearchCV(
        estimator=Ridge(positive=True),
        param_grid={"alpha": np.logspace(-3, 3, 10)},
        scoring=scoring,
        cv=5,
    )
    final_estimator = TransformedTargetRegressor(
        regressor=ridge, func=_clip_negative, inverse_func=_identity
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "stack",
                StackingRegressor(
                    estimators=base_estimators, final_estimator=final_estimator, passthrough=True
                ),
            ),
        ]
    )
    if tuned_params:
        pipeline.set_params(**tuned_params)
    return pipeline


def regressor_for_metric(
    metric: str,
    hyperparameters: dict,
    apply_tuned: bool,
    scoring: str = "neg_mean_absolute_error",
) -> Pipeline:
    """Build the regressor used for one target metric.

    The paper code merged the tuned `stack__*` keys into a dict of `rf__*`-style defaults
    and then read only the default keys, so the tuned values (including
    `min_samples_leaf`) were never applied and every published number uses
    DEFAULT_PARAMS. That remains the default here; `apply_tuned=True` applies
    `hyperparameters[metric]` to the pipeline.
    """
    if not apply_tuned:
        return build_regressor(scoring=scoring)
    if metric not in hyperparameters:
        raise KeyError(f"No tuned hyperparameters for {metric}; add them to the config")
    return build_regressor(hyperparameters[metric], scoring=scoring)
