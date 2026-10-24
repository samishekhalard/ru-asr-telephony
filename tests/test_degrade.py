"""
Tests for the telephony channel simulation.

These assert the physics, not the output of a particular library version: a
narrowband channel must remove high frequencies, a codec must add
quantisation noise without destroying the signal, and every profile must
hand back audio the model can still read.
"""

import numpy as np
import pytest

from degrade import (
    CHANNELS,
    PSTN_RATE,
    alaw_roundtrip,
    apply_channel,
    mulaw_roundtrip,
    to_narrowband,
)

SR = 16_000


def tone(freq: float, seconds: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def energy(audio: np.ndarray) -> float:
    return float(np.sum(audio.astype(np.float64) ** 2))


def test_narrowband_removes_what_a_phone_line_removes():
    """
    A 6 kHz tone cannot survive an 8 kHz channel; a 1 kHz tone must.

    This is the whole premise of the experiment, so it is the first test.
    """
    high = tone(6_000)
    low = tone(1_000)

    assert energy(to_narrowband(high, SR)) < 0.01 * energy(high)
    assert energy(to_narrowband(low, SR)) > 0.80 * energy(low)


def test_narrowband_keeps_length_and_rate():
    """Frames must stay aligned with the reference transcript's timing."""
    audio = tone(1_000, seconds=1.3)
    assert len(to_narrowband(audio, SR)) == len(audio)


@pytest.mark.parametrize("codec", [mulaw_roundtrip, alaw_roundtrip])
def test_codec_roundtrip_preserves_the_signal(codec):
    """
    Companding is lossy but not destructive: the waveform must still
    correlate strongly with the original.
    """
    audio = tone(1_000)
    out = codec(audio)
    correlation = np.corrcoef(audio, out)[0, 1]
    assert correlation > 0.99


@pytest.mark.parametrize("codec", [mulaw_roundtrip, alaw_roundtrip])
def test_codec_actually_quantises(codec):
    """
    If the round trip were lossless the experiment would be measuring
    nothing. There must be a measurable difference.
    """
    audio = tone(1_000)
    out = codec(audio)
    assert not np.allclose(audio, out, atol=1e-6)


@pytest.mark.parametrize("codec", [mulaw_roundtrip, alaw_roundtrip])
def test_codec_handles_silence_and_extremes(codec):
    """Silence stays silence; full scale does not wrap around."""
    assert np.allclose(codec(np.zeros(100, dtype=np.float32)), 0.0, atol=1e-6)
    loud = np.array([1.0, -1.0] * 50, dtype=np.float32)
    assert np.all(np.abs(codec(loud)) <= 1.0001)


@pytest.mark.parametrize("channel", CHANNELS)
def test_every_channel_returns_usable_audio(channel):
    """Same length, same rate, finite values, nothing clipped out of range."""
    audio = tone(800, seconds=0.75)
    out = apply_channel(audio, SR, channel)
    assert len(out) == len(audio)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0001


def test_clean_is_really_untouched():
    audio = tone(1_000)
    assert np.allclose(apply_channel(audio, SR, "clean"), audio)


def test_codec_degrades_more_than_bandwidth_alone():
    """
    g711 must differ from narrowband, otherwise the codec stage is a no-op
    and the three-way comparison in the README would be meaningless.
    """
    audio = tone(1_000)
    narrow = apply_channel(audio, SR, "narrowband")
    ulaw = apply_channel(audio, SR, "g711_ulaw")
    assert not np.allclose(narrow, ulaw, atol=1e-4)


def test_int16_input_is_accepted():
    """Datasets hand out int16 as often as float."""
    audio = (tone(1_000) * 32767).astype(np.int16)
    out = apply_channel(audio, SR, "g711_alaw")
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 1.0001


def test_unknown_channel_is_refused():
    with pytest.raises(ValueError, match="unknown channel"):
        apply_channel(tone(1_000), SR, "bluetooth")


def test_pstn_rate_is_eight_k():
    """Guard against someone 'improving' the constant."""
    assert PSTN_RATE == 8_000
