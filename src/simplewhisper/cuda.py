"""GPU support.

CTranslate2 does the maths on the GPU through cuBLAS. The CUDA runtime is
linked into CTranslate2 itself, so a GPU installation is only ever missing
one thing: the cuBLAS library.

This module looks for that library in the places it is usually installed
and, when it is nowhere to be found, can fetch it from NVIDIA. The download
is 548 MB, so it never happens without the user saying yes.

Only Windows is handled for the download; the other systems get cuBLAS
through the NVIDIA pip packages, which are found like any other library.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import default_cuda_dir

log = logging.getLogger("simplewhisper")

# The library CTranslate2 loads when it runs on a GPU.
CUBLAS_DLL = "cublas64_12.dll"

# The NVIDIA redistributable that holds it. The file is large because
# cublasLt64_12.dll alone is 660 MB.
CUBLAS_URL = (
    "https://developer.download.nvidia.com/compute/cuda/redist/"
    "libcublas/windows-x86_64/libcublas-windows-x86_64-12.8.3.14-archive.zip"
)
CUBLAS_SHA256 = "a2f990cf61f0086d942632f2455727240baa9378c2e9fa2bdda56ef81f6cf8ad"
CUBLAS_SIZE = 574528660

# The driver has to be new enough for the CUDA 12 runtime that is linked
# into CTranslate2.
MINIMUM_DRIVER = "525.60.13"

# Called with the number of bytes done and the total.
ProgressCallback = Callable[[int, int], None]

# Asked whether the download may happen; returns True when it may.
AskCallback = Callable[[str], bool]

# The handles of the folders that were added to the DLL search path. They
# have to stay alive for as long as the folders are needed.
_dll_handles: List[object] = []


class GpuUnavailable(RuntimeError):
    """The GPU was asked for, but it cannot be used."""


@dataclass
class GpuPlan:
    """What to do about the GPU."""

    requested: str  # "auto", "cpu" or "cuda"
    device: str  # "cuda" or "cpu"
    reason: str  # a short explanation
    needs_download: bool = False  # the CUDA libraries are missing
    has_gpu: bool = False  # a GPU is present and can be used
    folder: Optional[Path] = None  # where the libraries were found


# --------------------------------------------------------------------------- #
# Finding the libraries
# --------------------------------------------------------------------------- #


def cuda_cache_folder() -> Path:
    """The folder the CUDA libraries are unpacked into."""
    return default_cuda_dir() / "bin"


def _toolkit_roots() -> List[Path]:
    """The folders where the CUDA toolkit is installed."""
    roots: List[Path] = []
    for variable in ("ProgramFiles", "ProgramW6432"):
        base = os.environ.get(variable)
        if base:
            roots.append(Path(base) / "NVIDIA GPU Computing Toolkit" / "CUDA")

    # The installer sets CUDA_PATH to the version folder itself.
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        roots.append(Path(cuda_path))
        roots.append(Path(cuda_path).parent)

    return roots


def _site_packages_folders() -> List[Path]:
    """The site-packages folders of the Python installations on this machine."""
    folders: List[Path] = []

    def add(candidate: Path) -> None:
        if candidate.is_dir():
            folders.append(candidate)

    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        if not prefix:
            continue
        add(Path(prefix) / "Lib" / "site-packages")
        add(Path(prefix) / "lib" / "site-packages")

    local = os.environ.get("LOCALAPPDATA")
    if local:
        root = Path(local) / "Programs" / "Python"
        for candidate in root.glob("Python*/Lib/site-packages"):
            add(candidate)

    roaming = os.environ.get("APPDATA")
    if roaming:
        root = Path(roaming) / "Python"
        for candidate in root.glob("Python*/site-packages"):
            add(candidate)

    for candidate in Path("C:/").glob("Python*/Lib/site-packages"):
        add(candidate)

    return folders


def cublas_folders() -> List[Path]:
    """The folders that may hold the CUDA libraries, best one first."""
    folders: List[Path] = []

    # Our own cache.
    folders.append(cuda_cache_folder())

    # A folder next to the executable, so the libraries can be dropped in
    # by hand instead of being downloaded.
    folders.append(Path(sys.executable).resolve().parent / "cuda")

    # The CUDA toolkit.
    for root in _toolkit_roots():
        folders.extend(sorted(root.glob("v*/bin"), reverse=True))
        folders.append(root / "bin")

    # PyTorch and the NVIDIA pip packages, in every Python on the machine.
    for site in _site_packages_folders():
        folders.append(site / "torch" / "lib")
        folders.append(site / "nvidia" / "cublas" / "bin")
        folders.append(site / "nvidia" / "cublas" / "lib" / "bin")

    return [folder for folder in folders if folder.is_dir()]


def find_cublas() -> Optional[Path]:
    """The folder holding the CUDA libraries, or None when there is none."""
    for folder in cublas_folders():
        if (folder / CUBLAS_DLL).is_file():
            return folder
    return None


def add_dll_folder(folder: Path) -> bool:
    """Put a folder on the DLL search path so the loader finds the libraries."""
    if os.name != "nt":
        return False
    folder = Path(folder)
    if not folder.is_dir():
        return False
    try:
        _dll_handles.append(os.add_dll_directory(str(folder)))
    except (AttributeError, OSError):
        return False
    return True


def activate_cublas(folder: Path) -> None:
    """Make the CUDA libraries in ``folder`` available to CTranslate2."""
    if add_dll_folder(folder):
        log.info("using the CUDA libraries in %s", folder)


# --------------------------------------------------------------------------- #
# The GPU itself
# --------------------------------------------------------------------------- #


def driver_present() -> bool:
    """True when an NVIDIA driver is installed."""
    if os.name != "nt":
        return True
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return (system / "nvcuda.dll").exists()


def driver_version() -> Optional[str]:
    """The version of the NVIDIA driver, when it can be read."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def gpu_device_count() -> Tuple[int, Optional[str]]:
    """The number of usable GPUs, and the reason when there are none."""
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # the CUDA runtime could not start
        return 0, str(exc)
    return int(count or 0), None


def cuda_runtime_error() -> Optional[str]:
    """The reason CUDA does not work, or None when it does."""
    try:
        import ctranslate2

        ctranslate2.get_supported_compute_types("cuda")
    except Exception as exc:
        return str(exc)
    return None


def plan_gpu(device: str = "auto") -> GpuPlan:
    """Work out whether the GPU can be used, without changing anything."""
    if device == "cpu":
        return GpuPlan(requested="cpu", device="cpu", reason="the CPU was requested")

    count, error = gpu_device_count()
    if count == 0:
        if error:
            reason = f"the GPU could not be used ({error})"
        elif not driver_present():
            reason = "no NVIDIA driver was found"
        else:
            reason = "no NVIDIA GPU was found"
        return GpuPlan(requested="auto", device="cpu", reason=reason)

    folder = find_cublas()
    if folder is not None:
        return GpuPlan(
            requested=device,
            device="cuda",
            reason="the GPU is ready",
            needs_download=False,
            has_gpu=True,
            folder=folder,
        )

    return GpuPlan(
        requested=device,
        device="cuda",
        reason="the CUDA libraries are missing",
        needs_download=True,
        has_gpu=True,
    )


# --------------------------------------------------------------------------- #
# Downloading the libraries
# --------------------------------------------------------------------------- #


def download_reachable(timeout: float = 15.0) -> Optional[str]:
    """Check that the download server answers; None when it does."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(CUBLAS_URL, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return None if response.status == 200 else f"server answered {response.status}"
    except urllib.error.HTTPError as exc:
        return f"server answered {exc.code}"
    except Exception as exc:
        return str(exc)


def install_cublas(progress: Optional[ProgressCallback] = None) -> Path:
    """Download the CUDA libraries from NVIDIA and unpack them into the cache."""
    folder = cuda_cache_folder()
    folder.mkdir(parents=True, exist_ok=True)

    handle, name = tempfile.mkstemp(prefix="simplewhisper-cublas-", suffix=".zip")
    os.close(handle)
    archive = Path(name)

    try:
        _download(archive, progress)
        _verify(archive, CUBLAS_SHA256)
        _extract(archive, folder)
    finally:
        try:
            archive.unlink()
        except OSError:
            pass

    if not (folder / CUBLAS_DLL).is_file():
        raise RuntimeError("the downloaded archive did not contain the CUDA libraries")

    activate_cublas(folder)
    return folder


def _download(target: Path, progress: Optional[ProgressCallback]) -> None:
    import urllib.request

    request = urllib.request.Request(CUBLAS_URL, headers={"User-Agent": "simplewhisper"})
    done = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        with target.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, CUBLAS_SIZE)


def _verify(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest().lower() != expected.lower():
        raise RuntimeError("the downloaded file is damaged; please try again")


def _extract(archive: Path, folder: Path) -> None:
    """Copy the DLLs out of the archive into ``folder``."""
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = Path(entry.filename)
            if name.suffix.lower() != ".dll":
                continue
            # The libraries sit in the bin folder of the archive; the rest
            # is headers, libraries and source code.
            if name.parent.name != "bin":
                continue
            with bundle.open(entry) as source:
                with (folder / name.name).open("wb") as target:
                    shutil.copyfileobj(source, target)


# --------------------------------------------------------------------------- #
# Putting it together
# --------------------------------------------------------------------------- #


def prepare_gpu(
    device: str,
    *,
    ask: Optional[AskCallback] = None,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[str, Optional[str]]:
    """Work out which device to use, and make it usable.

    Returns the device to use ("cuda" or "cpu") and a note for the user
    when something was changed or could not be done.

    The ``ask`` callback is called with the question when the CUDA
    libraries have to be downloaded; it returns True when the user agrees.
    Without it, or when the user says no, the CPU is used.

    Raises :class:`GpuUnavailable` when the GPU was asked for explicitly
    and cannot be used.
    """
    plan = plan_gpu(device)

    if not plan.has_gpu:
        # The normal case on a machine without an NVIDIA GPU: nothing to
        # say, and nothing to download. An explicit -d cuda still fails,
        # so the user knows their request was not honoured.
        if device == "cuda":
            raise GpuUnavailable(
                f"{plan.reason}; update the NVIDIA driver or use -d cpu"
            )
        return "cpu", None

    if not plan.needs_download:
        if plan.folder is not None:
            activate_cublas(plan.folder)
        return "cuda", None

    # The GPU is there, but the libraries that make it work are not.
    question = (
        "An NVIDIA GPU was found, but the CUDA libraries it needs are missing.\n"
        "They are 548 MB and are downloaded once from NVIDIA.\n"
        "Without them the transcription runs on the CPU."
    )
    if ask is not None and ask(question):
        try:
            install_cublas(progress=progress)
        except Exception as exc:
            if device == "cuda":
                raise GpuUnavailable(f"the CUDA libraries could not be installed: {exc}")
            return "cpu", f"the CUDA libraries could not be installed ({exc}); using the CPU"
        return "cuda", None

    if device == "cuda":
        raise GpuUnavailable(
            "the CUDA libraries are missing and the download was declined; "
            "use -d cpu or let the download happen"
        )
    return "cpu", "the CUDA libraries are missing; using the CPU"


def gpu_report() -> List[Tuple[str, str]]:
    """The state of the GPU, as rows of label and value."""
    rows: List[Tuple[str, str]] = []

    version = driver_version()
    if driver_present():
        rows.append(("NVIDIA driver", version or "installed"))
        if version:
            older = version.split(".")[0] < MINIMUM_DRIVER.split(".")[0]
            if older:
                rows.append(("Driver version", f"older than {MINIMUM_DRIVER}"))
    else:
        rows.append(("NVIDIA driver", "not found"))

    count, error = gpu_device_count()
    if count:
        rows.append(("CUDA devices", f"{count}"))
    elif error:
        rows.append(("CUDA devices", f"none ({error})"))
    else:
        rows.append(("CUDA devices", "none"))

    runtime_error = cuda_runtime_error()
    if runtime_error:
        rows.append(("CUDA runtime", runtime_error))

    folder = find_cublas()
    rows.append(("CUDA libraries", str(folder) if folder else "not found"))

    if count and folder is None:
        rows.append(("Download size", f"{CUBLAS_SIZE / 1024 / 1024:.0f} MB"))
    return rows


__all__ = [
    "CUBLAS_DLL",
    "CUBLAS_SIZE",
    "CUBLAS_URL",
    "GpuPlan",
    "GpuUnavailable",
    "MINIMUM_DRIVER",
    "activate_cublas",
    "cuda_cache_folder",
    "cuda_runtime_error",
    "driver_present",
    "driver_version",
    "download_reachable",
    "find_cublas",
    "gpu_device_count",
    "gpu_report",
    "install_cublas",
    "plan_gpu",
    "prepare_gpu",
]
