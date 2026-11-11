import pytest

from approx_asr.metrics import compute_metrics, normalize_text, score_frame


def test_word_and_character_counts():
    scores = compute_metrics("the cat sat on the mat", "the cat sit on the mat")
    assert (scores["nwSubstitutions"], scores["nwDeletions"], scores["nwInsertions"]) == (1, 0, 0)
    assert scores["WER"] == pytest.approx(100 / 6)
    assert scores["wRefLength"] == 6
    assert scores["cRefLength"] == len("the cat sat on the mat")
    assert (scores["ncSubstitutions"], scores["ncErrors"]) == (1, 1)
    assert scores["CER"] == pytest.approx(100 / len("the cat sat on the mat"))

    scores = compute_metrics("the cat sat", "the sat")
    assert (scores["nwDeletions"], scores["nwErrors"]) == (1, 1)
    scores = compute_metrics("the cat sat", "the big cat sat down")
    assert (scores["nwInsertions"], scores["nwErrors"]) == (2, 2)
    assert scores["WER"] == pytest.approx(200 / 3)


def test_normalization_removes_case_punctuation_and_brackets():
    assert compute_metrics("Hello, World!", "hello world")["nwErrors"] == 0
    assert normalize_text("Yes [noise] (laughs) OK.").split() == ["yes", "ok"]


def test_score_frame_leaves_missing_hypotheses_empty():
    import pandas as pd

    frame = pd.DataFrame({"reference": ["a b", "c d"], "hypothesis": ["a b", None]})
    scored = score_frame(frame)
    assert scored.loc[0, "WER"] == 0
    assert scored[["WER", "nwErrors"]].iloc[1].isna().all()


def test_normalizer_matches_transformers_when_available():
    english_normalizer = pytest.importorskip("transformers.models.whisper.english_normalizer")
    reference = english_normalizer.BasicTextNormalizer()
    for text in ["Hello, World!", "Dr. Smith's [inaudible] 3.5%", "Ünïcode — dash (aside)"]:
        assert normalize_text(text) == reference(text)
