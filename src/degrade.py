"""
Simulating what a telephone line does to speech.

Why simulate rather than record
-------------------------------
Real call-centre recordings are personal data and are not published. The
public Russian ASR benchmarks are all studio, audiobook or podcast audio at
16 kHz. So this repository takes clean 16 kHz speech with known transcripts
and passes it through the two things a phone call actually does to it.

That is an approximation, and the README says so. What it buys is a
controlled comparison: the same utterances, the same reference text, one
variable changed at a time.

The two things, kept separate on purpose
----------------------------------------
A phone call degrades speech in two distinct ways, and lumping them together
hides which one the engines are actually losing to:

  1. Band limiting. The PSTN carries roughly 300-3400 Hz at an 8 kHz sample
     rate. Everything above ~4 kHz is gone - which is most of the energy that
     distinguishes с, ш, ф and ц. No codec involved.

  2. Codec quantisation. G.711 companding squeezes 14 bits into 8 using a
     logarithmic curve, trading precision in loud passages for precision in
     quiet ones. This adds noise that band limiting alone does not.

`narrowband` applies only the first. `g711_ulaw` and `g711_alaw` apply both.
Running all three tells you how much of the damage is bandwidth and how much
is the codec, which a single "telephony" number cannot.

Everything returns audio at the original sample rate, because the ASR models
expect 16 kHz. The information lost at 8 kHz does not come back when it is
upsampled - that is the point.
"""

from __future__ import annotations

import numpy as np
import soxr

PSTN_RATE = 8_000

# G.711 constants, from ITU-T Recommendation G.711.
_MU = 255.0
_A = 87.6
_SEG_MAX = 8


def _as_float32(audio: np.ndarray) -> np.ndarray:
    """Accept int16 or float, return float32 in roughly [-1, 1]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return np.asarray(audio, dtype=np.float32)


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Band-limited resampling. soxr, so the anti-alias filter is not ours."""
    if from_rate == to_rate:
        return _as_float32(audio)
    return soxr.resample(_as_float32(audio), from_rate, to_rate).astype(np.float32)


def to_narrowband(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Down to 8 kHz and back, which throws away everything above ~4 kHz.

    The round trip is deliberate: the models want 16 kHz input, so the signal
    has to come back up. It returns at the original length and rate, carrying
    only the information a phone line would have carried.
    """
    narrow = resample(audio, sample_rate, PSTN_RATE)
    wide = resample(narrow, PSTN_RATE, sample_rate)
    return _fit_length(wide, len(audio))


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Resampling can be a sample or two off; keep frames aligned."""
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def mulaw_roundtrip(audio: np.ndarray) -> np.ndarray:
    """
    G.711 µ-law encode and decode, the North American PSTN codec.

    µ-law is logarithmic companding: 14 bits of linear PCM into 8, with the
    resolution concentrated where quiet speech lives. The round trip is what
    matters here - we never transmit anything, we only want the quantisation
    noise the codec leaves behind.
    """
    x = np.clip(_as_float32(audio), -1.0, 1.0)
    magnitude = np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    encoded = np.sign(x) * magnitude
    # 8 bits, signed: 256 levels across [-1, 1].
    quantised = np.round(encoded * 127.0) / 127.0
    decoded = np.sign(quantised) * (np.expm1(np.abs(quantised) * np.log1p(_MU)) / _MU)
    return decoded.astype(np.float32)


def alaw_roundtrip(audio: np.ndarray) -> np.ndarray:
    """
    G.711 A-law encode and decode, the European PSTN codec.

    Worth running beside µ-law because Russian telephony is A-law territory,
    and the two curves differ most exactly where quiet speech sits.
    """
    x = np.clip(_as_float32(audio), -1.0, 1.0)
    abs_x = np.abs(x)
    ln_a = 1.0 + np.log(_A)

    small = abs_x < (1.0 / _A)
    magnitude = np.where(
        small,
        _A * abs_x / ln_a,
        (1.0 + np.log(np.maximum(_A * abs_x, 1e-12))) / ln_a,
    )
    encoded = np.sign(x) * magnitude
    quantised = np.round(encoded * 127.0) / 127.0

    abs_q = np.abs(quantised)
    small_q = abs_q < (1.0 / ln_a)
    decoded_mag = np.where(
        small_q,
        abs_q * ln_a / _A,
        np.exp(abs_q * ln_a - 1.0) / _A,
    )
    return (np.sign(quantised) * decoded_mag).astype(np.float32)


CHANNELS = ("clean", "narrowband", "g711_ulaw", "g711_alaw")


def apply_channel(
    audio: np.ndarray, sample_rate: int, channel: str = "clean"
) -> np.ndarray:
    """
    Pass audio through one named channel profile.

    clean        untouched, the control
    narrowband   8 kHz band limiting only, no codec
    g711_ulaw    band limiting plus µ-law quantisation (North American PSTN)
    g711_alaw    band limiting plus A-law quantisation (European PSTN)
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")

    audio = _as_float32(audio)
    if channel == "clean":
        return audio

    # Companding happens at 8 kHz, where a real line would apply it, then the
    # result is brought back up for the model.
    narrow = resample(audio, sample_rate, PSTN_RATE)
    if channel == "g711_ulaw":
        narrow = mulaw_roundtrip(narrow)
    elif channel == "g711_alaw":
        narrow = alaw_roundtrip(narrow)
    wide = resample(narrow, PSTN_RATE, sample_rate)
    return _fit_length(wide, len(audio))
