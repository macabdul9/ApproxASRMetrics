"""SONAR speech-text cosine similarity between each utterance and a model's hypothesis.

Reads `<asr-dir>/<model>/<dataset>.csv` and writes the same rows with a `similarity`
column to `<output-dir>/<model>/<dataset>.csv`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from approx_asr.config import dataset_id, load_json
from approx_asr.subsets import load_subset

logger = logging.getLogger(__name__)


def create_encoders(speech_language: str = "eng", device: str = "cuda"):
    from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    device = torch.device(device)
    speech_encoder = SpeechToEmbeddingModelPipeline(
        encoder=f"sonar_speech_encoder_{speech_language}", device=device
    )
    text_encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device
    )
    return speech_encoder, text_encoder


def speech_embedding(audio, speech_encoder) -> torch.Tensor:
    embedding = speech_encoder.predict([torch.tensor(audio).unsqueeze(0)])
    return torch.nn.functional.normalize(embedding, p=2, dim=-1)


def text_embedding(text: str, text_encoder, language: str = "eng_Latn") -> torch.Tensor:
    embedding = text_encoder.predict([text], source_lang=language)
    return torch.nn.functional.normalize(embedding, p=2, dim=-1)


def add_similarity(frame: pd.DataFrame, dataset, speech_encoder, text_encoder, text_language):
    """Return the rows present in both the CSV and the dataset, in dataset order."""
    available = set(frame["audio_id"])
    dataset_ids = [audio_id for audio_id in dataset["id"] if audio_id in available]
    if len(dataset_ids) != len(dataset):
        logger.warning("%d dataset rows have no hypothesis", len(dataset) - len(dataset_ids))
        dataset = dataset.filter(lambda example: example["id"] in available)
    frame = frame.set_index("audio_id").loc[dataset_ids].reset_index()
    similarities = []
    for example, hypothesis in tqdm(
        zip(dataset, frame["hypothesis"], strict=True), total=len(frame)
    ):
        speech = speech_embedding(example["audio"]["array"], speech_encoder)
        text = text_embedding(str(hypothesis), text_encoder, text_language)
        similarities.append(float(torch.matmul(speech, text.T)))
    frame["similarity"] = similarities
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asr-dir", type=Path, default=Path("runs/asr"))
    parser.add_argument(
        "--models", nargs="+", default=None, help="Model directory names (default: all)"
    )
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/datasets_bench.json"))
    parser.add_argument("--speech-language", default="eng", help="SONAR speech encoder, e.g. deu")
    parser.add_argument("--text-language", default="eng_Latn", help="e.g. deu_Latn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/features"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    models = args.models or sorted(path.name for path in args.asr_dir.iterdir() if path.is_dir())
    speech_encoder, text_encoder = create_encoders(args.speech_language, args.device)
    for name, entry in load_json(args.dataset_config).items():
        dataset = None
        for model in models:
            input_csv = args.asr_dir / model / f"{dataset_id(name)}.csv"
            output_csv = args.output_dir / model / f"{dataset_id(name)}.csv"
            if not input_csv.exists() or output_csv.exists():
                continue
            if dataset is None:
                dataset = load_subset(name, entry)
            frame = add_similarity(
                pd.read_csv(input_csv), dataset, speech_encoder, text_encoder, args.text_language
            )
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_csv, index=False)
            logger.info("Saved %s", output_csv)


if __name__ == "__main__":
    main()
