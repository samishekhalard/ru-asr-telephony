"""
Text normalisation and error rates for comparing ASR engines.

Why this file exists
--------------------
A word error rate is only meaningful next to the normalisation that produced
it. Two engines can transcribe the same audio equally well and score very
differently because one writes "47" and the other writes "сорок семь".

The normalisation here is deliberately the same as the public Russian ASR
benchmark (lowercase, ё -> е, strip punctuation, collapse whitespace) so that
clean-audio numbers from this repository can be checked against theirs. One
step is added on top, and it is the reason this module is not three lines:

    numbers are expanded into words, on BOTH sides of the comparison.

Telephony is full of order numbers, amounts and dates. Without expansion a
correct recognition of "номер сорок семь" as "номер 47" is scored as two
errors, and every engine that prefers digits is punished for a formatting
choice rather than a mistake. The expansion is applied to the reference too,
so neither spelling is privileged.

Known limits of the expansion, stated because they affect the numbers:
  - Russian numerals inflect. num2words produces the nominative cardinal, so
    "заявки номер сорок семь" normalises cleanly, but an inflected form in the
    reference ("второго марта") will not match an expanded digit ("два").
    This inflates WER slightly on dates and ordinals, equally for all engines.
  - Decimals and digit groups separated by punctuation are expanded per run of
    digits: "3.5" becomes "три пять", not "три целых пять десятых".
Both are measured the same way for every engine, so comparisons between
engines stay fair even where the absolute number is pessimistic.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import jiwer
from num2words import num2words

# Anything that is not a word character or whitespace. Unicode-aware, so
# Cyrillic survives and punctuation does not.
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def expand_numbers(text: str) -> str:
    """Replace every run of digits with its Russian cardinal in words."""

    def replace(match: re.Match[str]) -> str:
        value = int(match.group())
        try:
            return num2words(value, lang="ru")
        except (NotImplementedError, OverflowError):
            # Absurdly large values have no spelling; leave them as digits so
            # the failure is visible in the diff rather than silently dropped.
            return match.group()

    return _DIGITS.sub(replace, text)


def normalize(text: str) -> str:
    """
    Put a transcript into the form the error rates are computed on.

    Order matters: numbers are expanded before punctuation is stripped, so a
    decimal point still separates digit runs rather than gluing them together.
    """
    text = text.lower()
    text = text.replace("ё", "е")
    text = expand_numbers(text)
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


@dataclass(frozen=True)
class ErrorRates:
    """Corpus-level rates plus the counts they were computed from."""

    wer: float
    cer: float
    n_utterances: int
    n_ref_words: int

    def __str__(self) -> str:
        return (
            f"WER {self.wer:.2%}  CER {self.cer:.2%}  "
            f"({self.n_utterances} utterances, {self.n_ref_words} ref words)"
        )


def error_rates(references: list[str], hypotheses: list[str]) -> ErrorRates:
    """
    Corpus-level WER and CER over aligned reference/hypothesis lists.

    Corpus-level means total errors divided by total reference words, not the
    mean of per-utterance rates. The two differ: averaging per-utterance rates
    lets one short mangled utterance count as much as a long correct one.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"{len(references)} references but {len(hypotheses)} hypotheses"
        )
    if not references:
        raise ValueError("nothing to score")

    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]

    # An empty reference has no words to divide by; jiwer raises rather than
    # guessing, which is correct, so drop those pairs and say how many.
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        raise ValueError("every reference normalised to an empty string")
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]

    return ErrorRates(
        wer=jiwer.wer(refs, hyps),
        cer=jiwer.cer(refs, hyps),
        n_utterances=len(refs),
        n_ref_words=sum(len(r.split()) for r in refs),
    )


def bootstrap_ci(
    references: list[str],
    hypotheses: list[str],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Confidence interval for corpus WER, by resampling utterances with
    replacement.

    A WER printed without an interval invites the reader to believe a
    difference that a few utterances could erase. With samples this small the
    interval is usually the most honest number in the table.

    The seed is fixed so the interval is reproducible.
    """
    if resamples < 1:
        raise ValueError("resamples must be positive")

    rng = random.Random(seed)
    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    n = len(pairs)

    draws = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        draws.append(jiwer.wer([p[0] for p in sample], [p[1] for p in sample]))

    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lo = draws[int(tail * resamples)]
    hi = draws[min(int((1.0 - tail) * resamples), resamples - 1)]
    return lo, hi


@dataclass(frozen=True)
class Delta:
    """A difference in WER between two conditions, with its interval."""

    delta: float  # positive means `b` is worse than `a`
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def __str__(self) -> str:
        verdict = "" if self.significant else " (interval includes zero)"
        return (
            f"{self.delta:+.2%} [{self.ci_low:+.2%}, {self.ci_high:+.2%}]{verdict}"
        )


def paired_delta_ci(
    references: list[str],
    hypotheses_a: list[str],
    hypotheses_b: list[str],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Delta:
    """
    Difference in WER between two conditions measured on the SAME utterances.

    Why paired rather than two separate intervals
    ---------------------------------------------
    Comparing two independent confidence intervals and checking whether they
    overlap is the wrong test, and it is conservative in a way that hides real
    effects. Most of the spread in either interval comes from utterances being
    unequally hard - a sentence full of proper nouns is hard in every
    condition. That shared difficulty cancels when the same resampled set of
    utterances is scored under both conditions.

    So: resample utterance indices once per draw, score both conditions on
    exactly that set, and take the difference. What survives is the effect of
    the condition itself.

    A positive delta means `b` has the higher error rate.
    """
    if not (len(references) == len(hypotheses_a) == len(hypotheses_b)):
        raise ValueError(
            f"lengths differ: {len(references)} references, "
            f"{len(hypotheses_a)} and {len(hypotheses_b)} hypotheses"
        )
    if resamples < 1:
        raise ValueError("resamples must be positive")

    refs = [normalize(r) for r in references]
    a = [normalize(h) for h in hypotheses_a]
    b = [normalize(h) for h in hypotheses_b]

    triples = [(r, x, y) for r, x, y in zip(refs, a, b) if r]
    if not triples:
        raise ValueError("every reference normalised to an empty string")
    n = len(triples)

    point = jiwer.wer([t[0] for t in triples], [t[2] for t in triples]) - jiwer.wer(
        [t[0] for t in triples], [t[1] for t in triples]
    )

    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [triples[rng.randrange(n)] for _ in range(n)]
        refs_s = [t[0] for t in sample]
        draws.append(
            jiwer.wer(refs_s, [t[2] for t in sample])
            - jiwer.wer(refs_s, [t[1] for t in sample])
        )

    draws.sort()
    tail = (1.0 - confidence) / 2.0
    return Delta(
        delta=point,
        ci_low=draws[int(tail * resamples)],
        ci_high=draws[min(int((1.0 - tail) * resamples), resamples - 1)],
    )
