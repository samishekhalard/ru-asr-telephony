"""
Tests for normalisation and error rates.

The first test is the one that matters: it is the real failure from the first
transcription this project ever made, turned into a regression test.
"""

import pytest

from metrics import ErrorRates, bootstrap_ci, error_rates, expand_numbers, normalize


def test_digits_and_words_are_the_same_after_normalising():
    """
    The failure that started this module.

    Whisper heard "номер сорок семь" and wrote "номер 47". That is a correct
    recognition, and a naive WER scores it as two errors. After expansion the
    two strings are identical and the error rate is zero.
    """
    reference = "Я хотел бы уточнить статус моей заявки номер сорок семь."
    hypothesis = "Я хотел бы уточнить статус моей заявки номер 47"

    assert normalize(reference) == normalize(hypothesis)
    assert error_rates([reference], [hypothesis]).wer == 0.0


def test_expansion_runs_in_both_directions():
    """Neither spelling is privileged: digits in the reference expand too."""
    assert normalize("номер 47") == normalize("номер сорок семь")


def test_normalisation_follows_the_public_benchmark():
    """Lowercase, ё -> е, no punctuation, single spaces."""
    assert normalize("Ещё,   ПРИВЕТ!") == "еще привет"


def test_expand_numbers_leaves_text_alone():
    assert expand_numbers("без чисел") == "без чисел"
    assert expand_numbers("дом 5") == "дом пять"


def test_wer_counts_real_errors():
    """A substitution in a four-word reference is 25%."""
    rates = error_rates(["один два три четыре"], ["один два три пять"])
    assert rates.wer == pytest.approx(0.25)
    assert rates.n_ref_words == 4
    assert rates.n_utterances == 1


def test_corpus_level_not_averaged_per_utterance():
    """
    Two utterances: one long and perfect, one short and wrong.

    Averaging per-utterance rates would give 50%. Corpus-level divides total
    errors by total reference words, which is 1 error in 7 words.
    """
    refs = ["раз два три четыре пять шесть", "семь"]
    hyps = ["раз два три четыре пять шесть", "восемь"]
    assert error_rates(refs, hyps).wer == pytest.approx(1 / 7)


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="hypotheses"):
        error_rates(["а", "б"], ["а"])


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="nothing to score"):
        error_rates([], [])


def test_bootstrap_interval_brackets_the_estimate():
    refs = ["один два три четыре"] * 10
    hyps = ["один два три пять"] * 9 + ["один два три четыре"]
    point = error_rates(refs, hyps).wer
    lo, hi = bootstrap_ci(refs, hyps, resamples=500)
    assert lo <= point <= hi


def test_bootstrap_is_reproducible():
    refs = ["один два три четыре"] * 10
    hyps = ["один два три пять"] * 5 + ["один два три четыре"] * 5
    assert bootstrap_ci(refs, hyps, resamples=200) == bootstrap_ci(
        refs, hyps, resamples=200
    )


def test_error_rates_str_is_readable():
    rates = ErrorRates(wer=0.1234, cer=0.0456, n_utterances=3, n_ref_words=20)
    text = str(rates)
    assert "12.34%" in text and "3 utterances" in text
