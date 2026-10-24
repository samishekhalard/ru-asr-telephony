"""
ASR engines behind one interface.

Every engine takes 16 kHz mono float32 and returns text plus the wall-clock
seconds it took. Nothing else: no per-engine normalisation, no post-editing,
no language-specific fixups. Whatever an engine emits goes into the same
normaliser as everything else, so a comparison between two engines is a
comparison of the engines and not of the glue around them.

Timing is recorded because the public Russian ASR benchmark explicitly does
not measure it, and a voice assistant on a phone line cares about latency as
much as accuracy. The numbers describe one laptop and say nothing about a
server, which is why they are reported as a real-time factor rather than as
milliseconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Transcript:
    """What an engine returned, and what it cost."""

    text: str
    seconds: float  # wall clock spent transcribing
    audio_seconds: float

    @property
    def real_time_factor(self) -> float:
        """Below 1.0 means faster than real time."""
        return self.seconds / self.audio_seconds if self.audio_seconds else float("nan")


class Engine:
    """Base class. Subclasses implement `_transcribe`."""

    name = "engine"

    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        started = time.perf_counter()
        text = self._transcribe(audio, sample_rate)
        return Transcript(
            text=text,
            seconds=time.perf_counter() - started,
            audio_seconds=len(audio) / sample_rate,
        )


class WhisperMLX(Engine):
    """
    Whisper through MLX, running on the Apple Silicon GPU.

    `large-v3-turbo` is the checkpoint the public benchmark used, so the
    clean-channel number from here can be compared with theirs directly.

    The language is pinned to Russian rather than detected. Detection on
    narrowband audio is itself unreliable, and letting it vary would add a
    second changing thing to an experiment that is meant to change one.
    """

    def __init__(self, repo: str = "mlx-community/whisper-large-v3-turbo"):
        self.repo = repo
        self.name = f"whisper:{repo.rsplit('/', 1)[-1]}"

    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import mlx_whisper

        if sample_rate != 16_000:
            raise ValueError(f"whisper wants 16 kHz, got {sample_rate}")

        result = mlx_whisper.transcribe(
            audio.astype(np.float32),
            path_or_hf_repo=self.repo,
            language="ru",
            temperature=0.0,  # greedy, so a rerun gives the same text
            verbose=None,
        )
        return str(result.get("text", "")).strip()


class Echo(Engine):
    """
    A stand-in that returns a fixed string.

    Used by the tests to exercise the runner without a model, and useful for
    checking that the pipeline is wired correctly before spending an hour of
    GPU time on it.
    """

    name = "echo"

    def __init__(self, text: str = ""):
        self.text = text

    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return self.text


# Registry, so run.py takes engine names rather than imports.
BUILDERS: dict[str, Callable[[], Engine]] = {
    "whisper": WhisperMLX,
    "echo": Echo,
}


def build(name: str) -> Engine:
    if name not in BUILDERS:
        raise ValueError(f"unknown engine {name!r}; expected one of {sorted(BUILDERS)}")
    return BUILDERS[name]()
