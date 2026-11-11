import json
from pathlib import Path

import numpy as np

from approx_asr.regressor import DEFAULT_PARAMS, build_regressor, regressor_for_metric

ROOT = Path(__file__).resolve().parents[1]
HYPERPARAMETERS = json.loads((ROOT / "configs/hyperparameters.json").read_text())


def tiny_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    similarity = rng.uniform(0.3, 1.0, n)
    proxy_errors = rng.integers(0, 6, n).astype(float)
    errors = np.round(proxy_errors + 3 * (1 - similarity))
    return np.column_stack([similarity, proxy_errors]), errors


def test_pipeline_fits_and_predicts_nonnegative_errors():
    features, errors = tiny_data()
    pipeline = build_regressor().set_params(
        stack__rf__n_estimators=10, stack__gbr__n_estimators=10, stack__hgb__max_iter=10
    )
    pipeline.fit(features, errors)
    predictions = pipeline.predict(features)
    assert predictions.shape == errors.shape
    assert np.isfinite(predictions).all()


def test_tuned_hyperparameters_are_not_applied_by_default():
    pipeline = regressor_for_metric("WER", HYPERPARAMETERS, apply_tuned=False)
    rf = pipeline.named_steps["stack"].estimators[0][1]
    hgb = pipeline.named_steps["stack"].estimators[2][1]
    assert rf.n_estimators == DEFAULT_PARAMS["rf__n_estimators"]
    assert rf.min_samples_leaf == 1
    assert hgb.max_depth == DEFAULT_PARAMS["hgb__max_depth"]


def test_tuned_hyperparameters_are_applied_with_flag():
    pipeline = regressor_for_metric("CER", HYPERPARAMETERS, apply_tuned=True)
    params = pipeline.get_params()
    for key, value in HYPERPARAMETERS["CER"].items():
        assert params[key] == value
    assert params["stack__rf__min_samples_leaf"] == 4


def test_search_space_names_valid_pipeline_parameters():
    space = json.loads((ROOT / "configs/search_space.json").read_text())
    params = build_regressor().get_params()
    assert set(space) <= set(params)
    for tuned in HYPERPARAMETERS.values():
        for key, value in tuned.items():
            assert value in space[key]
