"""Audio extraction helpers.

ffmpeg is used when available because it handles almost every container and
codec. When it is missing we pass the original file to faster-whisper, whose
bundled PyAV decoder can read audio from most audio and video files.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger("simplewhisper")


def find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def extract_audio(source: Path, ffmpeg: str) -> Path:
    """Extract a mono 16 kHz WAV track into a temporary file."""
    handle, temp_name = tempfile.mkstemp(prefix="simplewhisper-", suffix=".wav")
    os.close(handle)
    destination = Path(temp_name)

    command = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        destination.unlink(missing_ok=True)
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(
            "ffmpeg could not extract audio"
            + (f": {detail}" if detail else "")
        ) from exc
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"could not run ffmpeg: {exc}") from exc

    return destination


def prepare_media(source: Path, use_ffmpeg: bool = True) -> Tuple[Path, bool]:
    """Return ``(media_path, is_temporary)`` ready for transcription."""
    if not use_ffmpeg:
        return source, False

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        log.debug("ffmpeg not found; using the built-in PyAV decoder")
        return source, False

    try:
        return extract_audio(source, ffmpeg), True
    except RuntimeError as exc:
        log.warning("%s; trying the built-in decoder instead", exc)
        return source, False
