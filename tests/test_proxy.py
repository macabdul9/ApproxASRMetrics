import pandas as pd
import pytest

from approx_asr.proxy import proxy_metrics, select_proxies

PROFILE = {
    "best": {"WER_mean_all_datasets": 10.0},
    "second": {"WER_mean_all_datasets": 12.0},
    "third": {"WER_mean_all_datasets": 20.0},
    "unprofiled": {},
}


def test_proxy_is_best_other_model():
    assert select_proxies(PROFILE, "third") == ["best"]
    assert select_proxies(PROFILE, "best") == ["second"]
    assert select_proxies(PROFILE, "best", count=3) == ["second", "third", "unprofiled"]


def test_proxy_metrics_score_target_against_proxy_by_position():
    target = pd.Series(["the cat sat", "a dog"], index=[5, 9])
    proxy = pd.Series(["the cat sat", "a big dog"])
    features = proxy_metrics(target, proxy)
    assert list(features.index) == [5, 9]
    assert features["Ref_nwErrors"].tolist() == [0, 1]
    assert features["Ref_nwDeletions"].tolist() == [0, 1]
    assert features["Ref_WER"].iloc[1] == pytest.approx(100 / 3)


def test_periods_in_proxy_are_replaced_as_in_the_paper_runs():
    # "done." becomes "doneno_audio" -> "doneno audio": one substitution, one deletion.
    features = proxy_metrics(pd.Series(["done"]), pd.Series(["done."]))
    assert features["Ref_nwErrors"].iloc[0] == 2
    assert features["Ref_nwSubstitutions"].iloc[0] == 1


def test_missing_hypotheses_use_placeholder():
    features = proxy_metrics(pd.Series([None]), pd.Series([None]))
    assert features["Ref_nwErrors"].iloc[0] == 0
