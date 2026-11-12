"""Audio playback for the desktop mini player.

The mini player needs three things: a way to play a media file, a way to
pause and seek, and a position that can be sampled while the audio runs so
the transcript can follow along.

Two backends provide that:

* ``ffplay`` (part of ffmpeg) plays the file itself. It understands every
  container ffmpeg supports and can seek inside a file, so nothing has to be
  decoded in Python.
* Without ffmpeg the audio is decoded with PyAV into a temporary WAV file
  and handed to the platform player: ``winsound`` on Windows, ``afplay`` on
  macOS and ``paplay``/``gst-play`` on Linux.

Both backends measure the position with a monotonic clock, which is enough to
follow along in the transcript.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import List, Optional

# Keep the console window from flashing up on Windows when ffplay is started.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Status values returned by the backends.
RUNNING = "running"
FINISHED = "finished"
FAILED = "failed"

# The playback speeds the mini player offers.
SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


class PlaybackError(RuntimeError):
    """Raised when the audio cannot be played."""


def find_ffplay() -> Optional[str]:
    """Return the path of ``ffplay`` when ffmpeg is installed."""
    return shutil.which("ffplay")


def _atempo_chain(speed: float) -> Optional[str]:
    """Build the ffmpeg filter that changes the playback speed.

    ``atempo`` takes a factor between 0.5 and 2.0, so a bigger change is
    made by chaining several of them.
    """
    if abs(speed - 1.0) < 1e-6:
        return None

    factors: List[float] = []
    remaining = float(speed)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:g}" for factor in factors)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def _interleaved_samples(frame) -> bytes:
    """Return one audio frame as interleaved 16-bit little-endian samples."""
    import numpy as np

    array = frame.to_ndarray()
    if array.dtype.kind == "f":
        # Floating point samples run from -1 to 1.
        array = array * 32767.0
    elif array.dtype.kind == "i" and array.dtype.itemsize > 2:
        # Wider integers keep the full scale; keep the two top bytes.
        array = array >> (8 * (array.dtype.itemsize - 2))

    if array.ndim == 1:
        samples = array
    elif array.shape[0] == 1:
        # Packed formats already hold the channels one after the other.
        samples = array.reshape(-1)
    else:
        # Planar formats keep one channel after the other; interleave them.
        samples = array.T.reshape(-1)

    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def decode_to_wav(source: Path, destination: Path) -> float:
    """Decode the audio track of ``source`` into a 16-bit WAV file.

    Returns the duration of the decoded audio in seconds.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - PyAV ships with faster-whisper
        raise PlaybackError(
            "playing audio needs ffmpeg; install ffmpeg and put ffplay on the PATH"
        ) from exc

    try:
        container = av.open(str(source))
    except Exception as exc:
        raise PlaybackError(f"could not read {source.name}: {exc}") from exc

    with container:
        stream = next(
            (item for item in container.streams if item.type == "audio"), None
        )
        if stream is None:
            raise PlaybackError(f"{source.name} has no audio track")

        rate = int(stream.codec_context.sample_rate or 44100)
        channels = int(stream.codec_context.channels or 1)
        frames = 0

        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            for frame in container.decode(stream):
                data = _interleaved_samples(frame)
                if not data:
                    continue
                handle.writeframes(data)
                frames += len(data) // (2 * channels)

    if frames == 0:
        raise PlaybackError(f"{source.name} has no audio to play")

    return frames / float(rate or 1)


def _slice_wav(source: Path, destination: Path, offset: float, rate_scale: float = 1.0) -> None:
    """Copy ``source`` into ``destination``, starting ``offset`` seconds in.

    The copy is played at ``rate_scale`` times the speed of the original, by
    giving it a different frame rate. That keeps the fallback simple, at the
    price of a pitch that shifts with the speed.
    """
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        rate = params.framerate
        total = params.nframes
        reader.setpos(max(0, min(int(offset * rate), total)))

        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(params.nchannels)
            writer.setsampwidth(params.sampwidth)
            writer.setframerate(max(1, int(round(rate * rate_scale))))
            while True:
                chunk = reader.readframes(65536)
                if not chunk:
                    break
                writer.writeframes(chunk)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class _FfplayBackend:
    """Play the file with ``ffplay``, which can also seek inside it."""

    name = "ffplay"

    def __init__(self, executable: str) -> None:
        self.executable = executable
        self._process: Optional[subprocess.Popen] = None
        self._offset = 0.0
        self._started = 0.0
        self._duration = 0.0
        self._speed = 1.0

    # -- lifecycle -----------------------------------------------------
    def prepare(self, path: Path) -> float:
        """ffplay reads the file itself, so there is nothing to prepare."""
        return 0.0

    def start(self, path: Path, offset: float, duration: float, speed: float = 1.0) -> None:
        self.stop()
        self._speed = float(speed) if speed > 0 else 1.0

        command = [
            self.executable,
            "-nodisp",  # no video window
            "-autoexit",  # quit when the file ends
            "-vn",  # no video at all
            "-loglevel",
            "error",
        ]
        if offset > 0.01:
            command += ["-ss", f"{offset:.3f}"]
        chain = _atempo_chain(self._speed)
        if chain:
            command += ["-af", chain]
        command.append(str(path))

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise PlaybackError(f"could not start ffplay: {exc}") from exc

        self._offset = offset
        self._started = time.monotonic()
        self._duration = duration

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=3)
        except Exception:  # pragma: no cover - the player may already be gone
            try:
                process.kill()
            except Exception:
                pass

    # -- state ---------------------------------------------------------
    @property
    def position(self) -> float:
        if self._process is None:
            return self._offset
        elapsed = time.monotonic() - self._started
        position = self._offset + elapsed * self._speed
        if self._duration > 0:
            position = min(position, self._duration)
        return position

    @property
    def status(self) -> str:
        process = self._process
        if process is None:
            return FINISHED
        code = process.poll()
        if code is None:
            return RUNNING
        return FINISHED if code == 0 else FAILED


class _NativeBackend:
    """Decode the audio once and hand it to the platform player."""

    name = "native"

    def __init__(self) -> None:
        self._source: Optional[Path] = None
        self._wav: Optional[Path] = None
        self._tail: Optional[Path] = None
        self._offset = 0.0
        self._started = 0.0
        self._duration = 0.0
        self._process: Optional[subprocess.Popen] = None
        self._playing = False
        self._ready = False
        self._generation = 0
        self._speed = 1.0
        self._lock = threading.RLock()

    # -- lifecycle -----------------------------------------------------
    def prepare(self, path: Path) -> float:
        """Decode ``path`` into a temporary WAV file and return its duration."""
        with self._lock:
            if self._source == path and self._wav is not None and self._wav.exists():
                self._ready = True
                return self._duration

            self._cleanup_files()
            generation = self._generation

        handle, name = tempfile.mkstemp(prefix="simplewhisper-play-", suffix=".wav")
        os.close(handle)
        wav = Path(name)
        try:
            duration = decode_to_wav(path, wav)
        except Exception:
            wav.unlink(missing_ok=True)
            raise

        with self._lock:
            if generation != self._generation:
                # Another file took over while this one was being decoded.
                wav.unlink(missing_ok=True)
                return self._duration
            self._source = path
            self._wav = wav
            self._duration = duration
            self._ready = True
            atexit.register(self._cleanup_files)
        return duration

    def start(self, path: Path, offset: float, duration: float, speed: float = 1.0) -> None:
        with self._lock:
            if not self._ready or self._wav is None:
                raise PlaybackError("the audio is still being prepared")

            self.stop()
            self._offset = offset
            self._started = time.monotonic()
            self._speed = float(speed) if speed > 0 else 1.0
            if duration > 0:
                self._duration = duration

            if offset > 0.01 or abs(self._speed - 1.0) > 1e-6:
                handle, name = tempfile.mkstemp(prefix="simplewhisper-tail-", suffix=".wav")
                os.close(handle)
                tail = Path(name)
                try:
                    _slice_wav(self._wav, tail, offset, rate_scale=self._speed)
                except Exception as exc:
                    tail.unlink(missing_ok=True)
                    raise PlaybackError(f"could not seek in the audio: {exc}") from exc
                self._tail = tail
                target = tail
            else:
                self._tail = None
                target = self._wav

            self._play(target)
            self._playing = True

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is not None:
                try:
                    process.terminate()
                    process.wait(timeout=3)
                except Exception:  # pragma: no cover - the player may already be gone
                    try:
                        process.kill()
                    except Exception:
                        pass
            elif self._playing and os.name == "nt":
                try:
                    import winsound

                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:  # pragma: no cover - no sound device
                    pass
            self._playing = False

    def _play(self, wav: Path) -> None:
        if os.name == "nt":
            try:
                import winsound
            except ImportError as exc:  # pragma: no cover - Windows only
                raise PlaybackError("winsound is not available") from exc
            try:
                winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except RuntimeError as exc:
                raise PlaybackError(f"could not play the audio: {exc}") from exc
            return

        command = self._platform_command(wav)
        if command is None:
            raise PlaybackError(
                "no audio player found; install ffmpeg (ffplay), paplay or gst-play"
            )
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise PlaybackError(f"could not start {command[0]}: {exc}") from exc

    def _platform_command(self, wav: Path) -> Optional[List[str]]:
        if sys.platform == "darwin":
            if shutil.which("afplay"):
                return ["afplay", str(wav)]
            return None
        for candidate in ("paplay", "gst-play-1.0"):
            if shutil.which(candidate):
                return [candidate, str(wav)]
        return None

    # -- state ---------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def position(self) -> float:
        if not self._playing:
            return self._offset
        elapsed = time.monotonic() - self._started
        position = self._offset + elapsed * self._speed
        if self._duration > 0:
            position = min(position, self._duration)
        return position

    @property
    def status(self) -> str:
        if not self._playing:
            return FINISHED
        process = self._process
        if process is not None:
            code = process.poll()
            if code is not None:
                return FINISHED if code == 0 else FAILED
        if self._duration > 0 and self.position >= self._duration:
            return FINISHED
        return RUNNING

    def _cleanup_files(self) -> None:
        with self._lock:
            self._generation += 1
            for path in (self._tail, self._wav):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:  # pragma: no cover - best effort
                        pass
            self._tail = None
            self._wav = None
            self._source = None
            self._ready = False


# --------------------------------------------------------------------------- #
# Player
# --------------------------------------------------------------------------- #


class AudioPlayer:
    """Play one media file at a time and report where the audio is."""

    def __init__(self) -> None:
        self._backend = None
        self._path: Optional[Path] = None
        self._duration = 0.0
        self._offset = 0.0
        self._playing = False
        self._finished = False
        self._error = ""
        self._speed = 1.0
        self._lock = threading.RLock()

    # -- loading -------------------------------------------------------
    def load(self, path: Path, duration: float = 0.0) -> None:
        """Point the player at a file and reset the position."""
        with self._lock:
            self.stop()
            self._path = Path(path)
            self._duration = max(0.0, float(duration))
            self._offset = 0.0
            self._finished = False
            self._error = ""

    def set_duration(self, duration: float) -> None:
        """Update the duration once it is known."""
        with self._lock:
            if duration and duration > 0:
                self._duration = float(duration)

    def set_speed(self, speed: float) -> None:
        """Change the playback speed, keeping the place in the audio."""
        with self._lock:
            speed = float(speed)
            if speed <= 0:
                return
            self._speed = speed
            if self._playing and self._backend is not None and self._path is not None:
                position = self._backend.position
                self._backend.start(self._path, position, self._duration, speed)

    def prepare(self) -> None:
        """Get the file ready in the background so playback starts at once."""
        with self._lock:
            backend = self._ensure_backend()
            path = self._path
        if path is None:
            return
        if isinstance(backend, _FfplayBackend):
            backend.prepare(path)
            return

        def worker() -> None:
            try:
                duration = backend.prepare(path)
            except PlaybackError as exc:
                with self._lock:
                    self._error = str(exc)
                return
            except Exception as exc:  # pragma: no cover - defensive
                with self._lock:
                    self._error = str(exc)
                return
            with self._lock:
                if self._path == path and self._duration <= 0 and duration > 0:
                    self._duration = duration

        threading.Thread(target=worker, daemon=True).start()

    # -- transport -----------------------------------------------------
    def play(self) -> None:
        with self._lock:
            if self._path is None:
                raise PlaybackError("there is nothing to play")
            if self._finished or (
                self._duration > 0 and self._offset >= self._duration - 0.05
            ):
                self._offset = 0.0
            backend = self._ensure_backend()
            backend.start(self._path, self._offset, self._duration, self._speed)
            self._playing = True
            self._finished = False

    def pause(self) -> None:
        with self._lock:
            if not self._playing:
                return
            position = self.position
            if self._backend is not None:
                self._backend.stop()
            self._offset = position
            self._playing = False

    def stop(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.stop()
            self._offset = 0.0
            self._playing = False
            self._finished = False

    def seek(self, seconds: float) -> None:
        with self._lock:
            if self._path is None:
                return
            target = max(0.0, float(seconds))
            if self._duration > 0:
                target = min(target, self._duration)
            if self._playing and self._backend is not None:
                self._backend.start(self._path, target, self._duration, self._speed)
            else:
                self._offset = target

    # -- state ---------------------------------------------------------
    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def position(self) -> float:
        with self._lock:
            if self._playing and self._backend is not None:
                return self._backend.position
            return self._offset

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_ready(self) -> bool:
        with self._lock:
            if self._backend is None:
                return False
            if isinstance(self._backend, _FfplayBackend):
                return True
            return self._backend.is_ready

    @property
    def error(self) -> str:
        return self._error

    def poll(self) -> str:
        """Update the state and report what happened.

        Returns an empty string while the audio plays, ``finished`` when the
        file ended and ``failed`` when the player stopped with an error.
        """
        with self._lock:
            if not self._playing or self._backend is None:
                return ""
            status = self._backend.status
            if status == RUNNING:
                return ""
            position = self._backend.position
            self._playing = False
            self._offset = position
            if status == FAILED:
                return "failed"
            self._finished = True
            return "finished"

    # -- helpers -------------------------------------------------------
    def _ensure_backend(self):
        if self._backend is None:
            executable = find_ffplay()
            if executable:
                self._backend = _FfplayBackend(executable)
            else:
                self._backend = _NativeBackend()
        return self._backend
