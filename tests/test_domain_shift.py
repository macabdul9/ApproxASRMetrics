import numpy as np
import pytest

from approx_asr.domain_shift import (
    central_moment_discrepancy,
    load_embeddings,
    total_vocabulary_overlap,
    vocabulary,
)


def test_cmd_is_zero_for_identical_samples_and_grows_with_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    assert central_moment_discrepancy(x, x) == pytest.approx(0.0)
    assert central_moment_discrepancy(x, x + 1.0) == pytest.approx(2.0)
    assert central_moment_discrepancy(x, 3 * x) > 0


def test_paper_cmd_reduces_to_distance_between_mean_embeddings(tmp_path):
    rng = np.random.default_rng(1)
    source = rng.normal(size=(50, 1, 8)).astype(np.float32)
    target = rng.normal(loc=0.5, size=(40, 1, 8)).astype(np.float32)
    np.save(tmp_path / "src_speech_embeddings.npy", source)
    np.save(tmp_path / "tgt_speech_embeddings.npy", target)
    cmd = central_moment_discrepancy(
        load_embeddings(tmp_path, "src", per_utterance=False),
        load_embeddings(tmp_path, "tgt", per_utterance=False),
    )
    expected = np.linalg.norm(source.mean(axis=0) - target.mean(axis=0))
    assert cmd == pytest.approx(expected, rel=1e-5)
    assert load_embeddings(tmp_path, "tgt", per_utterance=True).shape == (40, 8)


def test_total_vocabulary_overlap():
    source = vocabulary(["The cat sat.", "the dog"])
    target = vocabulary(["A cat ran!"])
    assert source == {"the", "cat", "sat", "dog"}
    assert total_vocabulary_overlap(source, target) == pytest.approx(1 / 6)
    assert total_vocabulary_overlap(set(), set()) == 0.0
    assert vocabulary(["The"], lowercase=False) == {"The"}
