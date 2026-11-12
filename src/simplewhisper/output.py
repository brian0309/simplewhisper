"""Resolve output paths and write transcript files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from . import formatters
from .config import OUTPUT_FORMATS
from .transcribe import Transcription

FORMAT_EXTENSIONS = {
    "srt": ".srt",
    "vtt": ".vtt",
    "txt": ".txt",
    "json": ".json",
    "tsv": ".tsv",
}


def expand_formats(formats: Iterable[str]) -> List[str]:
    """Expand ``all`` and remove duplicates while keeping order."""
    expanded: List[str] = []
    for item in formats:
        candidates = OUTPUT_FORMATS if item == "all" else (item,)
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def write_outputs(
    result: Transcription,
    formats: Sequence[str],
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Write every requested format and return ``(written, skipped)``."""
    destination_dir = Path(output_dir).expanduser() if output_dir else result.source.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    skipped: List[Path] = []

    for output_format in expand_formats(formats):
        destination = destination_dir / (result.source.stem + FORMAT_EXTENSIONS[output_format])
        if destination.exists() and not overwrite:
            skipped.append(destination)
            continue

        content = formatters.render(result, output_format)
        with open(destination, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        written.append(destination)

    return written, skipped
