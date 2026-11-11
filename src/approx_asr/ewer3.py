"""eWER3 baseline (Chowdhury and Ali, 2023), reimplemented with frozen encoders.

Utterance features are the first-position hidden states of frozen wav2vec2-base-960h
(speech) and roberta-base (hypothesis text), optionally reduced with PCA per
modality, and fed to the same stacked regressor as our method. Evaluation mirrors
`approx_asr.ablation`: k-fold CV on the source model, the same folds for an OOD model,
and the in-the-wild sets.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import AutoProcessor, RobertaModel, RobertaTokenizer, Wav2Vec2Model

from approx_asr.config import dataset_id, load_json
from approx_asr.regressor import build_regressor
from approx_asr.subsets import load_subset

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
LABEL_COLUMNS = ("nwErrors", "ncErrors", "wRefLength", "cRefLength")


class FrozenEncoders(torch.nn.Module):
    def __init__(self, text_model: str, speech_model: str):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(text_model)
        self.text_encoder = RobertaModel.from_pretrained(text_model).eval()
        self.speech_processor = AutoProcessor.from_pretrained(speech_model)
        self.speech_encoder = Wav2Vec2Model.from_pretrained(speech_model).eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state[:, 0, :]
        # wav2vec2 has no CLS token; the original implementation used the first frame.
        speech = self.speech_encoder(input_values=batch["input_values"]).last_hidden_state[:, 0, :]
        return speech, text


def load_split(name: str, entry: dict, root_dir: Path, model_id: str, metric: str):
    """Pair a subset's audio with one model's hypotheses and labels (missing labels -> 0)."""
    frame = pd.read_csv(root_dir / model_id / f"{dataset_id(name)}.csv")
    dataset = load_subset(name, entry)
    if list(dataset["id"]) != frame["audio_id"].tolist():
        raise ValueError(f"{model_id}/{dataset_id(name)}.csv is not aligned with the dataset")
    labels = frame[list(LABEL_COLUMNS)].fillna(0)
    labels.insert(0, "audio_id", frame["audio_id"])
    labels["hypothesis"] = frame["hypothesis"].astype(str)
    labels["label"] = frame[metric].fillna(0)
    return dataset, labels


def collate(batch: list[dict]) -> dict:
    """Right-pad audio and token sequences to the longest item in the batch."""
    audio_length = max(item["input_values"].shape[-1] for item in batch)
    token_length = max(item["input_ids"].shape[-1] for item in batch)

    def pad(tensor, length):
        return torch.nn.functional.pad(tensor, (0, length - tensor.shape[-1]))

    return {
        "input_values": torch.stack([pad(i["input_values"], audio_length) for i in batch]),
        "input_ids": torch.stack([pad(i["input_ids"], token_length) for i in batch]),
        "attention_mask": torch.stack([pad(i["attention_mask"], token_length) for i in batch]),
    }


def extract_features(encoders, dataset, hypotheses: list[str], device, batch_size: int = 64):
    def encode(index: int) -> dict:
        tokens = encoders.tokenizer(
            hypotheses[index], return_tensors="pt", truncation=True, max_length=128
        )
        audio = encoders.speech_processor(
            dataset[index]["audio"]["array"], return_tensors="pt", sampling_rate=SAMPLING_RATE
        )
        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "input_values": audio["input_values"][0],
        }

    speech_features, text_features = [], []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Extracting features"):
        indices = range(start, min(start + batch_size, len(dataset)))
        batch = {
            key: value.to(device) for key, value in collate([encode(i) for i in indices]).items()
        }
        speech, text = encoders(batch)
        speech_features.append(speech.cpu().numpy())
        text_features.append(text.cpu().numpy())
    return np.concatenate(speech_features), np.concatenate(text_features)


def featurize(encoders, configs: dict, args, model_id: str, device):
    """Return (speech, text, labels) for the concatenation of the given datasets."""
    speech, text, labels = [], [], []
    for name, entry in configs.items():
        dataset, frame = load_split(name, entry, args.root_dir, model_id, args.metric)
        if args.debug:
            dataset, frame = dataset.select(range(100)), frame.iloc[:100]
        s, t = extract_features(encoders, dataset, frame["hypothesis"].tolist(), device)
        speech.append(s)
        text.append(t)
        labels.append(frame)
    return np.concatenate(speech), np.concatenate(text), pd.concat(labels, ignore_index=True)


@dataclass
class FoldModel:
    """Per-modality PCA (optional) followed by the stacked regressor, fit on one fold."""

    pipeline: object
    pca_speech: PCA | None
    pca_text: PCA | None

    @classmethod
    def fit(cls, speech, text, labels, n_components: int | None, scoring: str) -> FoldModel:
        pca_speech = pca_text = None
        if n_components:
            pca_speech = PCA(n_components=n_components).fit(speech)
            pca_text = PCA(n_components=n_components).fit(text)
        model = cls(build_regressor(scoring=scoring), pca_speech, pca_text)
        model.pipeline.fit(model.inputs(speech, text), labels)
        return model

    def inputs(self, speech, text) -> np.ndarray:
        if self.pca_speech is not None:
            speech, text = self.pca_speech.transform(speech), self.pca_text.transform(text)
        return np.concatenate([speech, text], axis=1)

    def predict(self, speech, text) -> np.ndarray:
        return self.pipeline.predict(self.inputs(speech, text))


def write_predictions(model: FoldModel, features, metric: str, output_csv: Path, index=None):
    speech, text, labels = features
    if index is not None:
        speech, text, labels = speech[index], text[index], labels.iloc[index]
    predictions = model.predict(speech, text)
    result = labels.drop(columns="label")
    result[metric] = labels["label"].to_numpy()
    result[f"a{metric}"] = predictions
    result.to_csv(output_csv, index=False)
    return mean_absolute_error(result[metric], predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root-dir", type=Path, default=Path("runs/features"))
    parser.add_argument("--model-id", default="data2vec-audio-base-960h")
    parser.add_argument("--ood-model-id", default="wav2vec2-base-960h")
    parser.add_argument("--text-model", default="roberta-base")
    parser.add_argument("--speech-model", default="facebook/wav2vec2-base-960h")
    parser.add_argument(
        "--benchmark-config", type=Path, default=Path("configs/datasets_bench.json")
    )
    parser.add_argument("--wild-config", type=Path, default=Path("configs/datasets_wild.json"))
    parser.add_argument("--metric", default="nwErrors")
    parser.add_argument(
        "--loss",
        choices=["mean_absolute_error", "mean_squared_error"],
        default="mean_absolute_error",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--apply-pca", action="store_true")
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true", help="Use 100 utterances per dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ewer3"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoders = FrozenEncoders(args.text_model, args.speech_model).to(device)
    benchmarks = load_json(args.benchmark_config)
    wild = load_json(args.wild_config)
    source = featurize(encoders, benchmarks, args, args.model_id, device)
    target = featurize(encoders, benchmarks, args, args.ood_model_id, device)
    wild_sets = {
        dataset_id(name): (
            featurize(encoders, {name: entry}, args, args.model_id, device),
            featurize(encoders, {name: entry}, args, args.ood_model_id, device),
        )
        for name, entry in wild.items()
    }
    del encoders
    torch.cuda.empty_cache()

    # Same directory layout as approx_asr.ablation, so `approx_asr.results ablation` applies.
    experiment = f"pca_{args.n_components}" if args.apply_pca else "no_pca"
    experiment_dir = args.output_dir / f"{experiment}_{args.loss}"
    folds = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(folds.split(source[2])):
        source_dir = experiment_dir / f"source_{args.model_id}" / args.metric / f"fold_{fold}"
        target_dir = experiment_dir / f"target_{args.ood_model_id}" / args.metric / f"fold_{fold}"
        source_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        speech, text, labels = source
        model = FoldModel.fit(
            speech[train_index],
            text[train_index],
            labels["label"].to_numpy()[train_index],
            args.n_components if args.apply_pca else None,
            scoring=f"neg_{args.loss}",
        )

        mae = write_predictions(model, source, args.metric, source_dir / "iid_test.csv", test_index)
        logger.info("model=%s fold=%d metric=%s mae=%.4f", args.model_id, fold, args.metric, mae)
        mae = write_predictions(model, target, args.metric, target_dir / "iid_test.csv", test_index)
        logger.info(
            "model=%s fold=%d metric=%s mae=%.4f", args.ood_model_id, fold, args.metric, mae
        )
        for name, (source_wild, target_wild) in wild_sets.items():
            write_predictions(model, source_wild, args.metric, source_dir / f"{name}.csv")
            write_predictions(model, target_wild, args.metric, target_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
