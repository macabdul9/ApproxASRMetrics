"""Transcribe evaluation subsets with one or more ASR models.

Writes `<output-dir>/<model>/<dataset>.csv` with `audio_id`, `reference`, `hypothesis`.
Existing CSVs are skipped; interrupted runs resume from a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from approx_asr.asr_models import MODEL_TYPES
from approx_asr.config import dataset_id, load_json
from approx_asr.subsets import load_subset

logger = logging.getLogger(__name__)


def model_type_for(model_name: str, models_config: dict) -> str:
    for model_type, names in models_config.items():
        if model_name in names:
            return model_type
    raise KeyError(f"{model_name} is not listed in the models config; pass --model-type")


def transcribe_dataset(
    dataset,
    text_column: str,
    transcribe,
    loaded,
    language: str,
    output_csv: Path,
    checkpoint_every: int,
) -> None:
    checkpoint_csv = output_csv.with_name(f"{output_csv.stem}_checkpoint.csv")
    position_file = output_csv.with_name(f"{output_csv.stem}_checkpoint.json")
    rows, start = [], 0
    if checkpoint_csv.exists() and position_file.exists():
        rows = pd.read_csv(checkpoint_csv).to_dict("records")
        start = json.loads(position_file.read_text())["next_index"]

    for index in tqdm(range(start, len(dataset)), desc=output_csv.stem):
        example = dataset[index]
        try:
            hypothesis = transcribe(loaded, example["audio"]["array"], language)
        except Exception:
            # The paper runs skipped utterances that failed to decode. Proxy features pair
            # rows by position, so a skipped row misaligns the rest of that dataset.
            logger.exception("Failed on %s index %d", output_csv.stem, index)
            continue
        rows.append(
            {
                "audio_id": example.get("id", index),
                "reference": example[text_column],
                "hypothesis": hypothesis,
            }
        )
        if (index + 1) % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)
            position_file.write_text(json.dumps({"next_index": index + 1}))

    if rows:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    checkpoint_csv.unlink(missing_ok=True)
    position_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=None, help="Default: all configured models")
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), default=None)
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/datasets_bench.json"))
    parser.add_argument(
        "--language",
        default="en",
        help="Whisper language code (en, de); use eng/deu for SeamlessM4T",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--speechbrain-savedir",
        type=Path,
        default=Path("pretrained_models"),
        help="Where SpeechBrain stores downloaded checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/asr"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    models_config = load_json(args.models_config)
    model_names = args.models or [name for names in models_config.values() for name in names]
    datasets_config = load_json(args.dataset_config)
    for model_name in model_names:
        model_type = args.model_type or model_type_for(model_name, models_config)
        model_dir = args.output_dir / model_name.split("/")[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        pending = {
            name: entry
            for name, entry in datasets_config.items()
            if not (model_dir / f"{dataset_id(name)}.csv").exists()
        }
        if not pending:
            logger.info("Skipping %s: all datasets transcribed", model_name)
            continue
        loader, transcribe = MODEL_TYPES[model_type]
        loaded = loader(model_name, args.device, savedir=args.speechbrain_savedir)
        for name, entry in pending.items():
            dataset = load_subset(name, entry)
            dataset = dataset.select(range(min(args.num_samples, len(dataset))))
            output_csv = model_dir / f"{dataset_id(name)}.csv"
            transcribe_dataset(
                dataset,
                entry.get("text_column", "reference"),
                transcribe,
                loaded,
                args.language,
                output_csv,
                args.checkpoint_every,
            )
        del loaded


if __name__ == "__main__":
    main()
