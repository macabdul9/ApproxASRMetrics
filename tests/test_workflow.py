"""End-to-end run of the regression and results stages on synthetic model outputs."""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from approx_asr import experiments, results
from approx_asr.config import write_json
from approx_asr.features import load_with_proxies

SENTENCES = ["the cat sat on the mat", "a quick brown fox", "speech is hard", "hello there world"]


def write_outputs(root, model, dataset, rng, noise, n=30):
    references = [SENTENCES[i % len(SENTENCES)] for i in range(n)]
    hypotheses = [
        " ".join(w for w in ref.split() if rng.random() > noise) or "uh" for ref in references
    ]
    from approx_asr.metrics import score_frame

    frame = pd.DataFrame(
        {"audio_id": [f"{dataset}_{i}" for i in range(n)], "reference": references}
    )
    frame["hypothesis"] = hypotheses
    frame = score_frame(frame)
    frame["similarity"] = 1 - frame["WER"] / 200 + rng.normal(0, 0.01, n)
    (root / model).mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / model / f"{dataset}.csv", index=False)


def test_benchmark_then_results(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    train, tuning = tmp_path / "train", tmp_path / "tuning"
    for model, noise in (("good", 0.05), ("bad", 0.4)):
        for dataset in ("benchA", "benchB", "wildA"):
            write_outputs(train, model, dataset, rng, noise)
        for dataset in ("benchA", "benchB"):
            write_outputs(tuning, model, dataset, rng, noise, n=10)
    write_json(tmp_path / "bench.json", {"ns/benchA": {}, "ns/benchB": {}})
    write_json(tmp_path / "wild.json", {"ns/wildA": {}})
    write_json(
        tmp_path / "profile.json",
        {
            "good": {"WER_mean_all_datasets": 5.0},
            "bad": {"WER_mean_all_datasets": 40.0},
        },
    )
    monkeypatch.setattr(experiments, "regressor_for_metric", lambda *a, **k: LinearRegression())
    args = argparse.Namespace(
        data_dir=train,
        tuning_dir=tuning,
        model_ids=["bad"],
        metrics="WER,CER,nwErrors,ncErrors",
        proxy_metric="WER",
        model_profile=tmp_path / "profile.json",
        benchmark_config=tmp_path / "bench.json",
        wild_config=tmp_path / "wild.json",
        hyperparameters=tmp_path / "profile.json",
        apply_tuned_hyperparameters=False,
        output_dir=tmp_path / "regression",
    )
    experiments.run_benchmark(args)

    model_dir = tmp_path / "regression" / "bad"
    assert sorted(p.name for p in model_dir.glob("*.csv")) == [
        "benchA_predicted.csv",
        "benchB_predicted.csv",
    ]
    results.average_wild(tmp_path / "regression", results.PREDICTION_COLUMNS)
    folds = [pd.read_csv(model_dir / d / "wildA_predicted.csv") for d in ("benchA", "benchB")]
    averaged = pd.read_csv(model_dir / "wildA_predicted.csv")
    np.testing.assert_allclose(averaged["aWER"], (folds[0]["aWER"] + folds[1]["aWER"]) / 2)

    results.summarize(tmp_path / "regression", tmp_path / "results", 20, 3, independent=False)
    summary = pd.read_csv(tmp_path / "results" / "summary.csv")
    assert set(summary["dataset"]) == {"benchA", "benchB", "wildA"}
    assert (summary["aWER_std"] == 0).all()

    table = results.compare_proxy(tmp_path / "regression", "nwErrors", tmp_path / "cmp.csv")
    assert len(table) == 3
    assert (table[["mae_regression", "mae_proxy"]] >= 0).all().all()


def test_multiple_proxies_per_proxy_and_mean_columns(tmp_path):
    rng = np.random.default_rng(1)
    for model, noise in (("target", 0.3), ("p1", 0.0), ("p2", 0.1)):
        write_outputs(tmp_path, model, "benchA", rng, noise, n=8)
    per_proxy = load_with_proxies(tmp_path, ["benchA"], "target", ["p1", "p2"], False)["benchA"]
    mean = load_with_proxies(tmp_path, ["benchA"], "target", ["p1", "p2"], True)["benchA"]
    assert {"Ref_nwErrors_Ref1", "Ref_nwErrors_Ref2"} <= set(per_proxy.columns)
    np.testing.assert_allclose(
        mean["Ref_nwErrors_mean"],
        (per_proxy["Ref_nwErrors_Ref1"] + per_proxy["Ref_nwErrors_Ref2"]) / 2,
    )


def test_ablation_writes_fold_outputs_that_results_can_summarize(tmp_path, monkeypatch):
    from approx_asr import ablation

    rng = np.random.default_rng(2)
    data = tmp_path / "train"
    for model, noise in (("src", 0.3), ("ood", 0.2), ("p1", 0.0), ("p2", 0.1)):
        for dataset in ("benchA", "wildA"):
            write_outputs(data, model, dataset, rng, noise, n=20)
    write_json(tmp_path / "bench.json", {"ns/benchA": {}})
    write_json(tmp_path / "wild.json", {"ns/wildA": {}})
    profile = {m: {"WER_mean_all_datasets": w} for m, w in (("p1", 1), ("p2", 2), ("ood", 3))}
    write_json(tmp_path / "profile.json", profile)
    monkeypatch.setattr(ablation, "regressor_for_metric", lambda *a, **k: LinearRegression())
    args = argparse.Namespace(
        data_dir=data,
        model_ids=["src"],
        ref_model_id=None,
        proxy_metric="WER",
        metrics="nwErrors,ncErrors",
        use_similarity=True,
        proxy_ref_count=2,
        mean_of_proxies=False,
        proxy_features="per-metric",
        n_folds=2,
        seed=42,
        model_profile=tmp_path / "profile.json",
        benchmark_config=tmp_path / "bench.json",
        wild_config=tmp_path / "wild.json",
        hyperparameters=tmp_path / "profile.json",
        apply_tuned_hyperparameters=False,
        output_dir=tmp_path / "ablation",
    )
    ablation.run_for_target(args, "ood")
    results.summarize_ablation(tmp_path / "ablation", tmp_path / "ablation.json")
    import json

    summary = json.loads((tmp_path / "ablation.json").read_text())
    experiment = summary["sim_True_prc_2_mean_False_ref_None_per-metric"]
    assert set(experiment) == {"source_src", "target_ood"}
    assert set(experiment["target_ood"]) == {"iid_test", "wildA"}
    assert set(experiment["target_ood"]["wildA"]) == {"nwErrors", "ncErrors"}
