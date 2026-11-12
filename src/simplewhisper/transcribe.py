"""Thin wrapper around faster-whisper."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger("simplewhisper")


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcription:
    source: Path
    model: str
    language: Optional[str]
    language_probability: float
    duration: float
    segments: List[Segment] = field(default_factory=list)
    task: str = "transcribe"


ProgressCallback = Callable[[Segment, object], None]


def load_model(
    name: str,
    *,
    device: str = "auto",
    compute_type: str = "default",
    model_dir: Optional[Path] = None,
):
    """Load (and on first use download) a faster-whisper model."""
    from faster_whisper import WhisperModel

    if compute_type == "default":
        # faster-whisper's "default" infers float16 from the weights, which is
        # slow on CPU (and triggers a conversion warning). int8 is the
        # recommended CPU type.
        resolved_device = device
        if device == "auto":
            try:
                import ctranslate2

                resolved_device = (
                    "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                )
            except Exception:
                resolved_device = "cpu"
        if resolved_device == "cpu":
            compute_type = "int8"

    kwargs = {}
    if model_dir is not None:
        kwargs["download_root"] = str(model_dir)

    return WhisperModel(name, device=device, compute_type=compute_type, **kwargs)


def transcribe_media(
    model,
    media: Path,
    *,
    language: Optional[str] = None,
    task: str = "transcribe",
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> Transcription:
    """Transcribe one media file, collecting subtitle-ready segments."""
    raw_segments, info = model.transcribe(
        str(media),
        language=language,
        task=task,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
    )

    segments: List[Segment] = []
    for raw in raw_segments:
        words = []
        if word_timestamps:
            for raw_word in raw.words or []:
                words.append(
                    Word(
                        start=float(raw_word.start),
                        end=float(raw_word.end),
                        text=raw_word.word,
                    )
                )
        segment = Segment(
            start=float(raw.start),
            end=float(raw.end),
            text=raw.text.strip(),
            words=words,
        )
        segments.append(segment)
        if progress is not None:
            progress(segment, info)

    return Transcription(
        source=media,
        model=getattr(model, "model_size_or_path", ""),
        language=getattr(info, "language", language),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        segments=segments,
        task=task,
    )
