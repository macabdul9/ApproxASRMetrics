"""Domain shift between LibriSpeech-clean and each evaluation set.

`embed` stores SONAR speech embeddings per dataset, `cmd` computes the central moment
discrepancy on them (Table 8), and `tvo` computes the vocabulary overlap of the
reference transcripts (Table 9).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from approx_asr.config import dataset_id, load_dataset_configs, write_json


def central_moment_discrepancy(x: np.ndarray, y: np.ndarray, n_moments: int = 5) -> float:
    """CMD of Zellinger et al. (2017) for samples shaped (n, dim), without range scaling.

    Sum of the L2 distance between means and between central moments of order 2..n_moments.
    """
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    centered_x, centered_y = x - mean_x, y - mean_y
    total = np.linalg.norm(mean_x - mean_y)
    for order in range(2, n_moments + 1):
        total += np.linalg.norm((centered_x**order).mean(axis=0) - (centered_y**order).mean(axis=0))
    return float(total)


def load_embeddings(embeddings_dir: Path, name: str, per_utterance: bool) -> np.ndarray:
    embeddings = np.load(embeddings_dir / f"{name}_speech_embeddings.npy")
    embeddings = embeddings.reshape(len(embeddings), -1)
    if per_utterance:
        return embeddings
    # Table 8 was computed on each dataset's mean embedding, a single row per dataset.
    # With one row, every central moment is zero and CMD reduces to the Euclidean
    # distance between the two mean embeddings.
    return embeddings.mean(axis=0, keepdims=True)


def run_cmd(args: argparse.Namespace) -> None:
    source = load_embeddings(args.embeddings_dir, args.source_domain, args.per_utterance)
    results = {}
    for name in load_dataset_configs(args.dataset_config):
        name = dataset_id(name)
        if name == args.source_domain:
            continue
        target = load_embeddings(args.embeddings_dir, name, args.per_utterance)
        results[name] = {"cmd": central_moment_discrepancy(source, target)}
    write_json(args.output, results)


def run_embed(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from approx_asr.similarity import create_encoders, speech_embedding
    from approx_asr.subsets import load_subset

    speech_encoder, _ = create_encoders(args.speech_language, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in load_dataset_configs(args.dataset_config).items():
        dataset = load_subset(name, entry)
        embeddings = [
            speech_embedding(example["audio"]["array"], speech_encoder).cpu().numpy()
            for example in tqdm(dataset, desc=dataset_id(name))
        ]
        np.save(args.output_dir / f"{dataset_id(name)}_speech_embeddings.npy", np.array(embeddings))


def vocabulary(texts, lowercase: bool = True) -> set[str]:
    words = set()
    for text in texts:
        words.update(re.findall(r"\w+", text.lower() if lowercase else text))
    return words


def total_vocabulary_overlap(source: set[str], target: set[str]) -> float:
    """Jaccard overlap |source & target| / |source | target| of two vocabularies."""
    union = source | target
    return len(source & target) / len(union) if union else 0.0


def run_tvo(args: argparse.Namespace) -> None:
    from approx_asr.subsets import load_subset

    config = load_dataset_configs(args.dataset_config)
    texts = {
        dataset_id(name): load_subset(name, entry)[entry["text_column"]]
        for name, entry in config.items()
    }
    source_vocab = vocabulary(texts.pop(args.source_domain), args.lowercase)
    results = {}
    for name, target_texts in texts.items():
        target_vocab = vocabulary(target_texts, args.lowercase)
        results[name] = {
            "tvo": total_vocabulary_overlap(source_vocab, target_vocab),
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "intersection_size": len(source_vocab & target_vocab),
        }
    write_json(args.output, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    embed = commands.add_parser("embed", help="Extract SONAR speech embeddings")
    embed.add_argument("--dataset-config", type=Path, nargs="+", required=True)
    embed.add_argument("--speech-language", default="eng")
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--output-dir", type=Path, default=Path("runs/embeddings"))
    embed.set_defaults(run=run_embed)

    cmd = commands.add_parser("cmd", help="Central moment discrepancy to a source domain")
    cmd.add_argument("--dataset-config", type=Path, nargs="+", required=True)
    cmd.add_argument("--embeddings-dir", type=Path, default=Path("runs/embeddings"))
    cmd.add_argument("--source-domain", default="librispeech_asr_test_clean_eval_1K")
    cmd.add_argument(
        "--per-utterance",
        action="store_true",
        help="Use all utterance embeddings instead of the per-dataset mean used in the paper",
    )
    cmd.add_argument("--output", type=Path, default=Path("runs/domain_shift/cmd.json"))
    cmd.set_defaults(run=run_cmd)

    tvo = commands.add_parser("tvo", help="Total vocabulary overlap to a source dataset")
    tvo.add_argument("--dataset-config", type=Path, nargs="+", required=True)
    tvo.add_argument("--source-domain", default="librispeech_asr_test_clean_eval_1K")
    tvo.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)
    tvo.add_argument("--output", type=Path, default=Path("runs/domain_shift/tvo.json"))
    tvo.set_defaults(run=run_tvo)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
