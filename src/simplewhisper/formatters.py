"""Render a transcription as SRT, VTT, plain text, JSON or TSV."""

from __future__ import annotations

import json
from typing import Iterable, List

from .transcribe import Segment, Transcription

OUTPUT_FORMATS = ("srt", "vtt", "txt", "json", "tsv")


def timestamp(seconds: float, decimal_mark: str = ",") -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_mark}{millis:03d}"


def _clean(segments: Iterable[Segment]) -> List[Segment]:
    return [segment for segment in segments if segment.text.strip()]


def _labeled(segment: Segment) -> str:
    """Prefix the text with the speaker name when one is known."""
    text = segment.text.strip()
    if segment.speaker:
        return f"{segment.speaker}: {text}"
    return text


def format_srt(segments: Iterable[Segment]) -> str:
    blocks = []
    for index, segment in enumerate(_clean(segments), start=1):
        start = timestamp(segment.start, ",")
        end = timestamp(segment.end, ",")
        blocks.append(f"{index}\n{start} --> {end}\n{_labeled(segment)}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_vtt(segments: Iterable[Segment]) -> str:
    blocks = []
    for segment in _clean(segments):
        start = timestamp(segment.start, ".")
        end = timestamp(segment.end, ".")
        blocks.append(f"{start} --> {end}\n{_labeled(segment)}")
    body = "\n\n".join(blocks)
    return "WEBVTT\n\n" + (body + "\n" if body else "")


def format_txt(segments: Iterable[Segment]) -> str:
    lines = [_labeled(segment) for segment in _clean(segments)]
    return "\n".join(lines) + ("\n" if lines else "")


def format_tsv(segments: Iterable[Segment]) -> str:
    lines = []
    for segment in _clean(segments):
        text = _labeled(segment).replace("\t", " ").replace("\r", " ").replace("\n", " ")
        lines.append(
            f"{timestamp(segment.start, '.')}\t{timestamp(segment.end, '.')}\t{text}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def format_json(result: Transcription) -> str:
    payload = {
        "source": str(result.source),
        "model": result.model,
        "task": result.task,
        "language": result.language,
        "language_probability": round(result.language_probability, 4),
        "duration": round(result.duration, 3),
        "segments": [
            {
                "id": index,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "speaker": segment.speaker,
            }
            for index, segment in enumerate(result.segments)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render(result: Transcription, output_format: str) -> str:
    if output_format == "srt":
        return format_srt(result.segments)
    if output_format == "vtt":
        return format_vtt(result.segments)
    if output_format == "txt":
        return format_txt(result.segments)
    if output_format == "tsv":
        return format_tsv(result.segments)
    if output_format == "json":
        return format_json(result)
    raise ValueError(f"unsupported output format: {output_format}")
