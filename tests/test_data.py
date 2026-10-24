"""
Tests for the dataset layer.

Everything here runs offline. The network call lives in one function and is
exercised by actually running the benchmark, not by a test that would fail
whenever Hugging Face is slow.
"""

import numpy as np
import pytest

from data import TARGET_RATE, Utterance, _pick_reference, _to_mono_float32, describe


def make(seconds: float = 2.0, reference: str = "раз два три") -> Utterance:
    return Utterance(
        uid="u1",
        audio=np.zeros(int(seconds * TARGET_RATE), dtype=np.float32),
        sample_rate=TARGET_RATE,
        reference=reference,
    )


def test_seconds_is_derived_not_stored():
    assert make(2.5).seconds == pytest.approx(2.5)


def test_raw_transcription_wins():
    """
    The raw field is preferred so that one normalisation path - ours - is the
    only one applied. Taking the corpus's pre-normalised field would mix two.
    """
    row = {"raw_transcription": "Дом 5.", "transcription": "дом пять"}
    assert _pick_reference(row) == "Дом 5."


def test_falls_back_through_known_field_names():
    assert _pick_reference({"transcription": "привет"}) == "привет"
    assert _pick_reference({"sentence": "привет"}) == "привет"


def test_blank_transcript_is_not_accepted():
    with pytest.raises(KeyError, match="schema has changed"):
        _pick_reference({"raw_transcription": "   "})


def test_missing_transcript_names_the_keys():
    with pytest.raises(KeyError, match="audio"):
        _pick_reference({"audio": {}, "num_samples": 1})


def test_stereo_is_mixed_to_mono():
    stereo = np.stack([np.ones(100), np.full(100, 3.0)], axis=1)
    mono = _to_mono_float32(stereo)
    assert mono.shape == (100,)
    assert mono.dtype == np.float32
    assert np.allclose(mono, 2.0)


def test_int_audio_becomes_float32():
    assert _to_mono_float32(np.ones(10, dtype=np.int16)).dtype == np.float32


def test_describe_reports_the_sample_size():
    text = describe([make(2.0, "раз два"), make(4.0, "три четыре пять")])
    assert "2 utterances" in text
    assert "5 reference words" in text
    assert "0.1 min" in text
