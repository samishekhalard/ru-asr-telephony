"""
Loading reference speech with known transcripts.

Which corpus and why
--------------------
FLEURS is read news sentences recorded in studio conditions, with a human
transcript for every utterance, released under CC-BY. It is the corpus the
public Russian ASR benchmark used for its studio numbers, which matters here:
the clean-channel results from this repository can be checked against theirs,
and if they disagree the fault is in this code, not in the finding.

Studio audio is also the right starting point for this particular question.
The experiment asks what a telephone line costs, so it needs input that has
not already been degraded by something else.

Streaming, on purpose
---------------------
The split is downloaded lazily and only as far as the requested sample size.
Pulling a whole speech corpus to measure fifty utterances wastes bandwidth and
makes the experiment harder for anyone else to repeat.

Decoding without torchcodec
---------------------------
`datasets` 5.x decodes audio columns through torchcodec, a heavy dependency
pinned to particular torch builds. The audio column is therefore left
undecoded and the bytes are read with soundfile, which is already needed
elsewhere. One less thing for anyone reproducing this to install, and one
less version conflict waiting to happen.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

# The Hub id and config. Kept as module constants because if the dataset moves
# this is the only line that should need changing.
DATASET = "google/fleurs"
CONFIG = "ru_ru"
TARGET_RATE = 16_000


@dataclass(frozen=True)
class Utterance:
    """One audio clip and the words a human wrote down for it."""

    uid: str
    audio: np.ndarray  # float32, mono, TARGET_RATE
    sample_rate: int
    reference: str

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sample_rate


def _to_mono_float32(array) -> np.ndarray:
    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim > 1:
        # Datasets occasionally hand back (samples, channels).
        audio = audio.mean(axis=1)
    return audio


def _decode_audio(field: dict) -> tuple[np.ndarray, int]:
    """
    Turn an undecoded `datasets` audio cell into samples and a rate.

    With `decode=False` the cell is `{"bytes": ..., "path": ...}`. Streaming
    from the Hub fills in the bytes; a local cache may give only a path.
    """
    raw = field.get("bytes")
    source = io.BytesIO(raw) if raw else field.get("path")
    if source is None:
        raise ValueError(
            f"audio cell has neither bytes nor path: keys {sorted(field)}"
        )
    audio, rate = sf.read(source, dtype="float32", always_2d=False)
    return _to_mono_float32(audio), int(rate)


def _pick_reference(row: dict) -> str:
    """
    FLEURS ships two transcript fields and they are not interchangeable.

    `raw_transcription` keeps the original punctuation, capitalisation and
    digits. `transcription` is pre-normalised by the corpus authors in a way
    that is not the same as ours. Taking the raw one and applying our own
    normalisation keeps a single, documented normalisation path for every
    engine and every channel.
    """
    for field in ("raw_transcription", "transcription", "sentence", "text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(
        f"no transcript field in row with keys {sorted(row)}; "
        "the dataset schema has changed"
    )


def load_utterances(
    limit: int = 50,
    *,
    split: str = "test",
    min_seconds: float = 1.0,
    max_seconds: float = 30.0,
) -> list[Utterance]:
    """
    Stream `limit` utterances from FLEURS Russian.

    Very short clips are dropped because a one-word reference makes WER swing
    between 0 and 100 percent and adds noise without adding information. Very
    long ones are dropped because they would dominate a corpus-level rate.
    Both bounds are applied identically to every channel and engine.
    """
    # Imported here so the tests need neither network nor datasets installed.
    from datasets import Audio, load_dataset

    stream = load_dataset(DATASET, CONFIG, split=split, streaming=True)
    # Hand back raw bytes instead of decoded arrays; see the module docstring.
    stream = stream.cast_column("audio", Audio(decode=False))

    kept: list[Utterance] = []
    for index, row in enumerate(stream):
        if len(kept) >= limit:
            break

        audio, rate = _decode_audio(row["audio"])
        if rate != TARGET_RATE:
            raise ValueError(
                f"expected {TARGET_RATE} Hz from {DATASET}, got {rate}; "
                "the degradation chain assumes 16 kHz input"
            )

        seconds = len(audio) / rate
        if not (min_seconds <= seconds <= max_seconds):
            continue

        kept.append(
            Utterance(
                uid=str(row.get("id", index)),
                audio=audio,
                sample_rate=rate,
                reference=_pick_reference(row),
            )
        )

    if not kept:
        raise RuntimeError(
            f"no usable utterances in {DATASET}/{CONFIG} split={split}"
        )
    return kept


def describe(utterances: list[Utterance]) -> str:
    """One line for the README, so the sample size is never left implicit."""
    total = sum(u.seconds for u in utterances)
    words = sum(len(u.reference.split()) for u in utterances)
    return (
        f"{len(utterances)} utterances, {total / 60:.1f} min of audio, "
        f"{words} reference words"
    )
