"""
Tests for the paired comparison.

The point of pairing is sensitivity: a small consistent effect buried in
utterances of wildly different difficulty should be detectable paired, and
invisible when two independent intervals are compared for overlap. The last
test here demonstrates exactly that, because it is the reason the function
exists.
"""

import pytest

from metrics import Delta, bootstrap_ci, paired_delta_ci


def test_no_difference_gives_zero_delta():
    refs = ["раз два три четыре"] * 10
    result = paired_delta_ci(refs, refs, refs, resamples=300)
    assert result.delta == 0.0
    assert not result.significant


def test_positive_delta_means_b_is_worse():
    refs = ["раз два три четыре"] * 20
    good = refs
    bad = ["раз два три пять"] * 20
    result = paired_delta_ci(refs, good, bad, resamples=300)
    assert result.delta > 0
    assert result.significant


def test_sign_flips_when_arguments_swap():
    refs = ["раз два три четыре"] * 20
    good, bad = refs, ["раз два три пять"] * 20
    forward = paired_delta_ci(refs, good, bad, resamples=300)
    backward = paired_delta_ci(refs, bad, good, resamples=300)
    assert forward.delta == pytest.approx(-backward.delta)


def test_interval_brackets_the_point_estimate():
    refs = ["раз два три четыре"] * 15
    b = ["раз два три пять"] * 7 + ["раз два три четыре"] * 8
    result = paired_delta_ci(refs, refs, b, resamples=300)
    assert result.ci_low <= result.delta <= result.ci_high


def test_reproducible():
    refs = ["раз два три четыре"] * 10
    b = ["раз два три пять"] * 10
    assert paired_delta_ci(refs, refs, b, resamples=200) == paired_delta_ci(
        refs, refs, b, resamples=200
    )


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="lengths differ"):
        paired_delta_ci(["а", "б"], ["а", "б"], ["а"])


def test_delta_str_flags_an_interval_containing_zero():
    assert "includes zero" in str(Delta(delta=0.01, ci_low=-0.01, ci_high=0.03))
    assert "includes zero" not in str(Delta(delta=0.02, ci_low=0.01, ci_high=0.03))


def test_pairing_detects_what_overlapping_intervals_miss():
    """
    The whole argument for this function, as a test.

    Twenty utterances of very different difficulty. Condition B is slightly
    but consistently worse: it adds one extra error to each of the hard ones.
    Because utterance difficulty dominates the spread, the two independent
    intervals overlap and a naive reader concludes "no difference". The
    paired interval, which cancels that shared difficulty, excludes zero.
    """
    easy_ref = "раз два"
    hard_ref = "раз два три четыре пять шесть семь восемь"
    hard_hyp = "раз два три четыре пять шесть семь ДЕВЯТЬ"
    hard_hyp_worse = "раз два три четыре пять ВОСЕМЬ семь ДЕВЯТЬ"

    refs, a, b = [], [], []
    for i in range(24):
        if i % 3:
            refs.append(easy_ref)
            a.append(easy_ref)
            b.append(easy_ref)
        else:
            refs.append(hard_ref)
            a.append(hard_hyp)
            b.append(hard_hyp)

    # B picks up one extra error on four of the eight hard utterances: small
    # next to how much the easy/hard mix moves the overall rate.
    worsened = 0
    for i in range(24):
        if i % 3 == 0 and worsened < 4:
            b[i] = hard_hyp_worse
            worsened += 1

    lo_a, hi_a = bootstrap_ci(refs, a, resamples=400)
    lo_b, hi_b = bootstrap_ci(refs, b, resamples=400)
    intervals_overlap = lo_b <= hi_a and lo_a <= hi_b

    paired = paired_delta_ci(refs, a, b, resamples=400)

    assert intervals_overlap, "setup should produce overlapping intervals"
    assert paired.significant, "pairing should still detect the effect"
    assert paired.delta > 0
