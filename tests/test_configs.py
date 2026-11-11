import json
from pathlib import Path

from approx_asr.config import METRICS, dataset_id

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load(name):
    return json.loads((CONFIGS / name).read_text())


def test_every_configured_model_has_a_profile_entry():
    models = load("models.json")
    names = [name.split("/")[-1] for names in models.values() for name in names]
    profile = load("model_profile.json")
    assert len(names) == len(set(names))
    assert set(names) == set(profile)
    for scores in profile.values():
        assert {f"{metric}_mean_all_datasets" for metric in METRICS} <= set(scores)


def test_benchmark_and_wild_sets_are_disjoint_and_well_formed():
    benchmarks, wild = load("datasets_bench.json"), load("datasets_wild.json")
    assert not {dataset_id(n) for n in benchmarks} & {dataset_id(n) for n in wild}
    for name in [
        "datasets_bench.json",
        "datasets_wild.json",
        "datasets_wild_additional.json",
        "datasets_de.json",
        "datasets_de_en.json",
    ]:
        for entry in load(name).values():
            assert {"split", "text_column"} <= set(entry)
    for entry in load("source_datasets.json").values():
        assert {"config", "split", "text_column"} <= set(entry)


def test_hyperparameters_cover_every_target_metric():
    assert set(METRICS) <= set(load("hyperparameters.json"))
