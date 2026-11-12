"""Speaker diarization services.

sherpa-onnx runs the whole diarization pipeline inside a single pybind11
call, and that call holds the Python GIL from start to finish. A worker
thread therefore blocks every other thread in the interpreter, which is
why the desktop window used to stop responding while the speakers were
being detected.

:class:`SpeakerService` puts the work in a child process, where the GIL
does not matter, and streams the progress back over a pipe so the window
can keep drawing while the job runs. :class:`InlineSpeakerService` is the
fallback for the rare case where a child process cannot be started.

Both classes offer the same interface, so callers do not need to know
which one they got:

    service = open_speaker_service(model_dir)
    turns = service.diarize(media, num_speakers=2, progress=report)
    service.close()
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from pathlib import Path
from typing import Callable, List, Optional

from .diarize import DEFAULT_THRESHOLD, SpeakerTurn, load_diarizer
from .errors import JobCancelled

log = logging.getLogger("simplewhisper")

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]

# How long to wait for the child process to load the speaker models.
READY_TIMEOUT = 120.0

# How often the parent looks for new messages from the child.
POLL_INTERVAL = 0.05


def _send(connection, kind: str, request_id: int, payload) -> None:
    """Send one message to the parent, ignoring a closed pipe."""
    try:
        connection.send((kind, request_id, payload))
    except (BrokenPipeError, EOFError, OSError):
        pass


def _child_main(requests, results, model_dir, threshold, num_threads) -> None:
    """Body of the child process.

    Loads the speaker models once and then answers diarization requests
    until the parent closes the pipe.
    """
    try:
        diarizer = load_diarizer(
            Path(model_dir), threshold=threshold, num_threads=num_threads
        )
    except Exception as exc:
        _send(results, "ready_failed", 0, f"{type(exc).__name__}: {exc}")
        return

    _send(results, "ready", 0, None)

    while True:
        try:
            request = requests.recv()
        except (EOFError, OSError):
            break
        if request is None:
            break

        media, num_speakers, request_id = request
        try:

            def report(fraction: float) -> None:
                _send(results, "progress", request_id, float(fraction))

            turns = diarizer.diarize(
                Path(media), num_speakers=num_speakers, progress=report
            )
            _send(results, "result", request_id, turns)
        except Exception as exc:
            _send(results, "error", request_id, f"{type(exc).__name__}: {exc}")


class SpeakerService:
    """Speaker diarization running in a child process.

    The heavy lifting happens in another process, so the GIL that
    sherpa-onnx holds never blocks the window that started the job.
    """

    # True when the work happens in another process.
    runs_in_process = True

    def __init__(
        self,
        model_dir: Path,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        num_threads: Optional[int] = None,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._threshold = threshold
        self._num_threads = num_threads
        self._next_id = 0
        self._closed = False

        # The spawn context is used on purpose: forking a process that
        # already holds a loaded model is not safe on every platform.
        context = multiprocessing.get_context("spawn")
        request_recv, request_send = context.Pipe(duplex=False)
        result_recv, result_send = context.Pipe(duplex=False)

        self._requests = request_send
        self._results = result_recv
        self._process = context.Process(
            target=_child_main,
            args=(
                request_recv,
                result_send,
                str(self._model_dir),
                threshold,
                num_threads,
            ),
            daemon=True,
            name="simplewhisper-speakers",
        )
        self._process.start()

        # The child has its own copies of the pipe ends now.
        request_recv.close()
        result_send.close()

    # ------------------------------------------------------------------ #

    def wait_ready(self, timeout: float = READY_TIMEOUT) -> None:
        """Block until the child has loaded the speaker models."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise RuntimeError("the speaker model process did not start in time")
            if self._results.poll(min(remaining, POLL_INTERVAL)):
                break
            # The parent keeps its own copy of the pipe open, so a child that
            # dies never sends an end-of-file. Check that it is still alive.
            if not self._process.is_alive():
                self.close()
                raise RuntimeError("the speaker model process stopped while starting")
        try:
            kind, _request_id, payload = self._results.recv()
        except EOFError as exc:
            self.close()
            raise RuntimeError("the speaker model process stopped while starting") from exc
        if kind == "ready_failed":
            self.close()
            raise RuntimeError(f"could not load the speaker models: {payload}")
        if kind != "ready":
            self.close()
            raise RuntimeError("the speaker model process sent an unexpected reply")
        log.info("speaker models loaded in a separate process")

    def diarize(
        self,
        media: Path,
        *,
        num_speakers: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
        cancel: Optional[CancelCheck] = None,
    ) -> List[SpeakerTurn]:
        """Return the speaker turns found in ``media``."""
        if self._closed:
            raise RuntimeError("the speaker service has been closed")

        self._next_id += 1
        request_id = self._next_id

        try:
            self._requests.send((str(media), num_speakers, request_id))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError("the speaker model process is not running") from exc

        while True:
            if cancel is not None and cancel():
                self.cancel()
                raise JobCancelled("the speaker detection was cancelled")

            if not self._results.poll(POLL_INTERVAL):
                if not self._process.is_alive():
                    raise RuntimeError("the speaker model process stopped unexpectedly")
                continue

            try:
                kind, reply_id, payload = self._results.recv()
            except EOFError as exc:
                raise RuntimeError("the speaker model process stopped unexpectedly") from exc
            if reply_id != request_id:
                continue  # a leftover reply from an earlier request
            if kind == "progress":
                if progress is not None:
                    progress(payload)
                continue
            if kind == "result":
                return payload
            if kind == "error":
                raise RuntimeError(payload)

    def cancel(self) -> None:
        """Stop the child process straight away."""
        self._stop()

    def close(self) -> None:
        """Shut the child process down."""
        if self._closed:
            return
        self._closed = True
        try:
            self._requests.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        try:
            self._requests.close()
        except OSError:
            pass
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        try:
            self._results.close()
        except OSError:
            pass

    def _stop(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


class InlineSpeakerService:
    """Speaker diarization in the current process.

    This is the fallback for the rare case where a child process cannot
    be started. The results are the same, but the window may stop
    responding while the job runs, because sherpa-onnx holds the GIL for
    the whole job.
    """

    # False when the work happens in this process.
    runs_in_process = False

    def __init__(
        self,
        model_dir: Path,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        num_threads: Optional[int] = None,
    ) -> None:
        self._diarizer = load_diarizer(
            Path(model_dir), threshold=threshold, num_threads=num_threads
        )

    def wait_ready(self, timeout: float = READY_TIMEOUT) -> None:
        """Nothing to wait for, the models are already loaded."""
        return None

    def diarize(
        self,
        media: Path,
        *,
        num_speakers: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
        cancel: Optional[CancelCheck] = None,
    ) -> List[SpeakerTurn]:
        """Return the speaker turns found in ``media``."""
        if cancel is not None and cancel():
            raise JobCancelled("the speaker detection was cancelled")
        return self._diarizer.diarize(
            media, num_speakers=num_speakers, progress=progress
        )

    def cancel(self) -> None:
        """Nothing to stop, the work runs in this thread."""
        return None

    def close(self) -> None:
        """Nothing to clean up."""
        return None


def open_speaker_service(
    model_dir: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    num_threads: Optional[int] = None,
) -> SpeakerService:
    """Open the best available speaker service.

    The out-of-process service is preferred, because it keeps the caller
    responsive. If the child process cannot be started the work falls
    back to this process, which gives the same results but can make a
    window stop responding.
    """
    try:
        service = SpeakerService(
            model_dir, threshold=threshold, num_threads=num_threads
        )
        service.wait_ready()
        return service
    except Exception as exc:
        log.warning(
            "could not start the speaker process (%s); falling back to this one", exc
        )
        return InlineSpeakerService(
            model_dir, threshold=threshold, num_threads=num_threads
        )
