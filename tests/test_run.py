"""
Tests for the engine interface and the runner.

The Echo engine stands in for a model, so these run in milliseconds and never
touch the GPU or the network. What they check is the wiring: that every
utterance reaches the engine, that the channel is actually applied, and that
the table says what the numbers say.
"""

import numpy as np
import pytest

from data import TARGET_RATE, Utterance
from engines import Echo, Transcript, build
from run import run_one, summary_table


def utterance(reference: str = "раз два три четыре", seconds: float = 1.0):
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(int(seconds * TARGET_RATE))).astype(np.float32)
    return Utterance(
        uid="u", audio=audio, sample_rate=TARGET_RATE, reference=reference
    )


def test_real_time_factor():
    assert Transcript("x", seconds=2.0, audio_seconds=4.0).real_time_factor == 0.5


def test_engine_records_its_own_timing():
    result = Echo("привет").transcribe(np.zeros(TARGET_RATE, dtype=np.float32), TARGET_RATE)
    assert result.text == "привет"
    assert result.seconds >= 0
    assert result.audio_seconds == pytest.approx(1.0)


def test_perfect_engine_scores_zero():
    """A transcript identical to the reference must give WER 0."""
    reference = "раз два три четыре"
    row = run_one(Echo(reference), [utterance(reference)], "clean", quiet=True)
    assert row["wer"] == 0.0
    assert row["n_utterances"] == 1


def test_every_utterance_is_transcribed():
    rows = run_one(Echo("раз"), [utterance()] * 5, "narrowband", quiet=True)
    assert len(rows["pairs"]) == 5


def test_raw_pairs_are_kept_verbatim():
    """
    The saved transcripts must be the engine's actual output, not the
    normalised form, or nobody can apply their own normalisation later.
    """
    row = run_one(Echo("Привет, МИР!"), [utterance()], "clean", quiet=True)
    assert row["pairs"][0]["hypothesis"] == "Привет, МИР!"


def test_channel_is_recorded():
    row = run_one(Echo(""), [utterance()], "g711_alaw", quiet=True)
    assert row["channel"] == "g711_alaw"


def test_confidence_interval_brackets_the_point_estimate():
    refs = [utterance("раз два три четыре") for _ in range(8)]
    row = run_one(Echo("раз два три пять"), refs, "clean", quiet=True)
    assert row["wer_ci_low"] <= row["wer"] <= row["wer_ci_high"]


def test_summary_table_is_markdown_and_ordered():
    rows = [
        {
            "engine": "e",
            "channel": "g711_alaw",
            "wer": 0.5,
            "cer": 0.2,
            "wer_ci_low": 0.4,
            "wer_ci_high": 0.6,
            "median_real_time_factor": 0.3,
        },
        {
            "engine": "e",
            "channel": "clean",
            "wer": 0.1,
            "cer": 0.05,
            "wer_ci_low": 0.05,
            "wer_ci_high": 0.15,
            "median_real_time_factor": 0.3,
        },
    ]
    table = summary_table(rows)
    assert table.startswith("| engine |")
    # clean is the control and must come first, however the rows arrived.
    assert table.index("clean") < table.index("g711_alaw")
    assert "10.0%" in table and "50.0%" in table


def test_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="unknown engine"):
        build("kaldi")


def test_registry_builds_echo():
    assert isinstance(build("echo"), Echo)


def test_delta_table_compares_each_channel_to_clean():
    """A channel identical to clean must come out as not distinguishable."""
    from run import delta_table

    pairs = [
        {"uid": str(i), "reference": "раз два три четыре", "hypothesis": "раз два три четыре"}
        for i in range(10)
    ]
    rows = [
        {"engine": "e", "channel": "clean", "pairs": pairs},
        {"engine": "e", "channel": "g711_alaw", "pairs": pairs},
    ]
    table = delta_table(rows)
    assert "g711_alaw" in table
    assert "not distinguishable" in table
    assert "clean |" not in table.split("\n", 2)[2]


def test_delta_table_flags_a_worse_channel():
    from run import delta_table

    refs = [{"uid": str(i), "reference": "раз два три четыре"} for i in range(20)]
    clean = [{**r, "hypothesis": "раз два три четыре"} for r in refs]
    worse = [{**r, "hypothesis": "раз два три пять"} for r in refs]
    rows = [
        {"engine": "e", "channel": "clean", "pairs": clean},
        {"engine": "e", "channel": "narrowband", "pairs": worse},
    ]
    assert "worse" in delta_table(rows)


def test_delta_table_is_empty_without_the_control():
    from run import delta_table

    pairs = [{"uid": "1", "reference": "раз", "hypothesis": "раз"}]
    assert delta_table([{"engine": "e", "channel": "narrowband", "pairs": pairs}]) == ""
