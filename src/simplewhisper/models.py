"""Model catalog and download helpers.

No model weights live in this repository. faster-whisper downloads them from
Hugging Face on first use and caches them locally.
"""

from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ModelInfo:
    name: str
    repo: str
    params: str
    download: str
    memory: str
    note: str


MODEL_CATALOG: Tuple[ModelInfo, ...] = (
    ModelInfo(
        "tiny",
        "Systran/faster-whisper-tiny",
        "39 M",
        "~75 MB",
        "~1 GB",
        "fastest; rough drafts and smoke tests",
    ),
    ModelInfo(
        "base",
        "Systran/faster-whisper-base",
        "74 M",
        "~145 MB",
        "~1 GB",
        "fast drafts, light on CPU",
    ),
    ModelInfo(
        "small",
        "Systran/faster-whisper-small",
        "244 M",
        "~484 MB",
        "~2 GB",
        "best speed/quality balance on CPU",
    ),
    ModelInfo(
        "medium",
        "Systran/faster-whisper-medium",
        "769 M",
        "~1.5 GB",
        "~4 GB",
        "more accurate; slow on CPU",
    ),
    ModelInfo(
        "large-v3",
        "Systran/faster-whisper-large-v3",
        "1.55 B",
        "~3.1 GB",
        "~6 GB",
        "most accurate; GPU recommended",
    ),
    ModelInfo(
        "large-v3-turbo",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "809 M",
        "~1.6 GB",
        "~4 GB",
        "near large-v3 quality, much faster; GPU recommended",
    ),
    ModelInfo(
        "distil-large-v3",
        "Systran/faster-distil-whisper-large-v3",
        "756 M",
        "~1.5 GB",
        "~4 GB",
        "English-only, large-class accuracy for less compute",
    ),
)

MODEL_NAMES = tuple(info.name for info in MODEL_CATALOG)
_BY_NAME = {info.name: info for info in MODEL_CATALOG}

# Same file set faster-whisper requests when it downloads a model itself.
_MODEL_FILE_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
)

# Called with (bytes_downloaded, bytes_total) while a model is downloading.
DownloadCallback = Callable[[int, int], None]


class _NullStream:
    """A file-like object that swallows everything written to it."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


_NULL_STREAM = _NullStream()


class _ByteTracker:
    """Turn the hub's progress bars into a single "bytes so far" number.

    huggingface_hub drives progress through ``tqdm`` instances it creates
    itself, so a subclass is installed to funnel every byte count into this
    tracker. The tracker keeps the largest ``n`` seen rather than summing
    bars: the hub creates one bar for the bytes on the wire and another for
    the bytes written to disk, so summing them would count bytes twice.
    """

    def __init__(self, callback: Optional[DownloadCallback] = None, known_total: int = 0) -> None:
        self._callback = callback
        self._lock = threading.Lock()
        self._done = 0
        self._total = known_total
        self._known_total = known_total
        self._last_emit = 0.0
        self._sealed = False

    def observe(self, done: int, total: int) -> None:
        with self._lock:
            if self._sealed:
                return
            # The size reported by the Hub API is the real size of the model.
            # The hub's own bars grow their total as files are discovered, so
            # they are only used when the exact size is unknown.
            if not self._known_total and total > self._total:
                self._total = total
            if done > self._done:
                self._done = done
            current = self._done
            overall = self._total
            now = time.monotonic()
            if now - self._last_emit < 0.1:
                return
            self._last_emit = now

        if self._callback is not None:
            self._callback(current, overall)

    def seal(self) -> None:
        """Stop reporting progress.

        The hub closes its progress bars after the download has finished, and
        that close would otherwise emit one more progress update.
        """
        with self._lock:
            self._sealed = True


def _make_progress_class(tracker: _ByteTracker):
    """Build a tqdm subclass that records byte progress into ``tracker``.

    Two details matter here. Plain tqdm rejects the ``name=`` keyword the
    hub passes to its own tqdm subclass, so it is popped. And the bar is
    never disabled: tqdm's ``update()`` returns immediately when disabled,
    which would freeze the byte counter at zero.
    """
    from tqdm.auto import tqdm as base_tqdm

    class _ProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs.setdefault("file", _NULL_STREAM)
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            super().update(n)
            self._report()

        def close(self):
            self._report()
            super().close()

        def display(self, msg=None, pos=None):
            # The GUI and the CLI draw their own status line.
            return None

        def _report(self) -> None:
            unit = str(getattr(self, "unit", "") or "")
            if not unit.startswith("B"):
                return
            tracker.observe(
                int(getattr(self, "n", 0) or 0),
                int(getattr(self, "total", 0) or 0),
            )

    return _ProgressTqdm


def find_model(name: str) -> Optional[ModelInfo]:
    return _BY_NAME.get(name)


def _remote_files(repo: str) -> List[Tuple[str, int]]:
    """List the model files in a repo together with their sizes.

    Returns an empty list when the metadata cannot be fetched (for example
    when running fully offline), in which case the caller falls back to the
    plain download.
    """
    import fnmatch

    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except Exception:
        return []

    files: List[Tuple[str, int]] = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if any(fnmatch.fnmatch(name, pattern) for pattern in _MODEL_FILE_PATTERNS):
            files.append((name, int(getattr(sibling, "size", 0) or 0)))
    return files


def _remote_file_size(repo: str, filename: str) -> int:
    """Return the size of one file in a repo, or 0 when it cannot be fetched."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except Exception:
        return 0

    for sibling in info.siblings:
        if sibling.rfilename == filename:
            return int(getattr(sibling, "size", 0) or 0)
    return 0


def model_cache_path(name: str, model_dir: Path) -> Optional[Path]:
    """Return the cached snapshot for a model, or None when it is not downloaded."""
    info = _BY_NAME.get(name)
    if info is None:
        return None

    repo_dir = model_dir / ("models--" + info.repo.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        try:
            for snapshot in snapshots.iterdir():
                if (snapshot / "model.bin").is_file():
                    return snapshot
        except OSError:
            pass

    # Older faster-whisper versions saved the files directly into a folder.
    for candidate in (model_dir / info.name, repo_dir, model_dir):
        if (candidate / "model.bin").is_file():
            return candidate
    return None


def cached_model_names(model_dir: Path) -> Set[str]:
    """Best-effort detection of models already present in the cache."""
    cached: Set[str] = set()
    if not model_dir.is_dir():
        return cached

    for info in MODEL_CATALOG:
        if model_cache_path(info.name, model_dir) is not None:
            cached.add(info.name)
    return cached


def download_model_files(
    name: str, model_dir: Path, progress: Optional[DownloadCallback] = None
) -> Path:
    """Download one model into the cache and return the snapshot path."""
    from faster_whisper.utils import download_model
    from huggingface_hub import snapshot_download

    if name not in _BY_NAME:
        raise ValueError(
            f"unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}"
        )

    model_dir.mkdir(parents=True, exist_ok=True)

    # faster-whisper has used both ``cache_dir`` and ``output_dir`` over time.
    parameters = inspect.signature(download_model).parameters
    if "cache_dir" in parameters:
        kwargs = {"cache_dir": str(model_dir)}
    else:
        kwargs = {"local_dir": str(model_dir)}

    info = _BY_NAME[name]

    # The exact size of the files is known before the download starts, which
    # gives a stable denominator for the progress bar. When the metadata
    # cannot be fetched (for example when offline) the progress is reported
    # without a total instead.
    remote = _remote_files(info.repo)
    total_bytes = sum(size for _, size in remote)

    tracker = _ByteTracker(progress, total_bytes)
    tqdm_class = _make_progress_class(tracker)

    path = snapshot_download(
        info.repo,
        allow_patterns=list(_MODEL_FILE_PATTERNS),
        tqdm_class=tqdm_class,
        **kwargs,
    )

    # The hub closes its bars after the download finishes, so stop tracking
    # before the final report to avoid one stray update afterwards.
    tracker.seal()

    if progress is not None:
        # The transfer bar counts bytes on the wire, which can be slightly
        # below the size on disk, so the finished download is reported as
        # complete rather than as a fraction just under 100%.
        progress(total_bytes, total_bytes)

    return Path(path)


def model_table_rows(cached: Set[str]) -> List[Tuple[str, ...]]:
    rows = []
    for info in MODEL_CATALOG:
        name = info.name + (" *" if info.name in cached else "")
        rows.append((name, info.params, info.download, info.memory, info.note))
    return rows


# --------------------------------------------------------------------------- #
# Diarization models
# --------------------------------------------------------------------------- #

# The speaker models live in public repos, so they can be fetched without a
# Hugging Face account or token.
DIARIZATION_FILES = (
    ("csukuangfj/sherpa-onnx-pyannote-segmentation-3-0", "model.onnx"),
    (
        "csukuangfj/speaker-embedding-models",
        "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    ),
)


def download_diarization_models(
    model_dir: Path, progress: Optional[DownloadCallback] = None
) -> Path:
    """Download the speaker diarization models into the cache.

    Returns the folder the model files were written to.
    """
    from huggingface_hub import hf_hub_download

    folder = Path(model_dir) / "diarization"
    folder.mkdir(parents=True, exist_ok=True)

    pending = [
        (repo, filename)
        for repo, filename in DIARIZATION_FILES
        if not (folder / filename).is_file()
    ]
    if not pending:
        if progress is not None:
            total = sum((folder / name).stat().st_size for _, name in DIARIZATION_FILES)
            progress(total, total)
        return folder

    total = sum(_remote_file_size(repo, filename) for repo, filename in pending)
    tracker = _ByteTracker(progress, total)
    tqdm_class = _make_progress_class(tracker)

    for repo, filename in pending:
        hf_hub_download(
            repo,
            filename,
            local_dir=str(folder),
            tqdm_class=tqdm_class,
        )

    tracker.seal()
    if progress is not None:
        progress(total, total)

    return folder
