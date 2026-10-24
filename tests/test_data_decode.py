"""
Tests for decoding audio cells without torchcodec.

A real wav is synthesised in memory, so these run offline and still exercise
the actual soundfile path the loader uses.
"""

import io

import numpy as np
import pytest
import soundfile as sf

from data import TARGET_RATE, _decode_audio


def wav_bytes(seconds: float = 0.5, rate: int = TARGET_RATE, channels: int = 1):
    t = np.arange(int(seconds * rate)) / rate
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_decodes_bytes_to_float32_mono():
    audio, rate = _decode_audio({"bytes": wav_bytes(), "path": None})
    assert rate == TARGET_RATE
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(TARGET_RATE * 0.5, abs=2)


def test_stereo_cell_is_mixed_down():
    audio, _ = _decode_audio({"bytes": wav_bytes(channels=2), "path": None})
    assert audio.ndim == 1


def test_falls_back_to_path_when_bytes_absent(tmp_path):
    """A locally cached row may carry only a filename."""
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes())
    audio, rate = _decode_audio({"bytes": None, "path": str(path)})
    assert rate == TARGET_RATE
    assert len(audio) > 0


def test_cell_with_neither_is_refused():
    with pytest.raises(ValueError, match="neither bytes nor path"):
        _decode_audio({"bytes": None, "path": None})
