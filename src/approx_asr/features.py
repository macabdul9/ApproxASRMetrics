"""Load per-model ASR outputs and attach proxy-reference features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from approx_asr.config import dataset_id
from approx_asr.proxy import proxy_metrics


def read_outputs(data_dir: Path, model_id: str, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / model_id / f"{dataset_id(name)}.csv")


def load_with_proxy(
    data_dir: Path, dataset_names: list[str], model_id: str, proxy_id: str
) -> dict[str, pd.DataFrame]:
    """Return {dataset_id: frame} with `Ref_*` columns from one proxy, NaN rows dropped."""
    if model_id == proxy_id:
        raise ValueError(f"Proxy must differ from the target model: {model_id}")
    data = {}
    for name in dataset_names:
        frame = read_outputs(data_dir, model_id, name)
        proxy = read_outputs(data_dir, proxy_id, name)
        frame = pd.concat([frame, proxy_metrics(frame["hypothesis"], proxy["hypothesis"])], axis=1)
        data[dataset_id(name)] = frame.dropna()
    return data


def load_with_proxies(
    data_dir: Path,
    dataset_names: list[str],
    model_id: str,
    proxy_ids: list[str],
    mean_of_proxies: bool,
) -> dict[str, pd.DataFrame]:
    """Attach features from several proxies, either per proxy or averaged.

    Per-proxy columns are named `Ref_<metric>_Ref<k>` (k starts at 1, best proxy first);
    averaged columns are `Ref_<metric>_mean`. With no proxies the frame is returned
    without proxy features (similarity-only ablation).
    """
    data = {}
    for name in dataset_names:
        frame = read_outputs(data_dir, model_id, name)
        # Missing references were filled in the ablation runs; such rows have no metric
        # values and are dropped below either way.
        frame["reference"] = frame["reference"].fillna("empty_audio")
        per_proxy = [
            proxy_metrics(frame["hypothesis"], read_outputs(data_dir, proxy_id, name)["hypothesis"])
            for proxy_id in proxy_ids
        ]
        if per_proxy and mean_of_proxies:
            mean = sum(per_proxy) / len(per_proxy)
            frame = pd.concat([frame, mean.add_suffix("_mean")], axis=1)
        else:
            for k, scores in enumerate(per_proxy, start=1):
                frame = pd.concat([frame, scores.add_suffix(f"_Ref{k}")], axis=1)
        data[dataset_id(name)] = frame.dropna()
    return data
