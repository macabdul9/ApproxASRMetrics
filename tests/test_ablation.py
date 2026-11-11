from approx_asr.ablation import select_features

COLUMNS = [
    "similarity",
    "Ref_nwErrors_Ref1",
    "Ref_ncErrors_Ref1",
    "Ref_nwErrors_Ref2",
    "Ref_ncErrors_Ref2",
]
METRICS = ["nwErrors", "ncErrors"]


def test_all_metrics_mode_uses_every_metrics_proxy_columns():
    features = select_features(COLUMNS, "nwErrors", METRICS, True, 2, False, "all-metrics")
    assert features == [
        "similarity",
        "Ref_nwErrors_Ref1",
        "Ref_nwErrors_Ref2",
        "Ref_ncErrors_Ref1",
        "Ref_ncErrors_Ref2",
    ]


def test_per_metric_mode_uses_only_target_metric():
    features = select_features(COLUMNS, "ncErrors", METRICS, False, 2, False, "per-metric")
    assert features == ["Ref_ncErrors_Ref1", "Ref_ncErrors_Ref2"]


def test_mean_of_proxies_and_similarity_only():
    assert select_features(COLUMNS, "nwErrors", METRICS, True, 3, True, "per-metric") == [
        "similarity",
        "Ref_nwErrors_mean",
    ]
    assert select_features(COLUMNS, "nwErrors", METRICS, True, 0, False, "all-metrics") == [
        "similarity"
    ]
