# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for SimpleWhisper.

Run it through build.bat, or directly:

    .venv\\Scripts\\python -m PyInstaller --noconfirm SimpleWhisper.spec

The build produces two executables that share one folder:

* ``SimpleWhisper.exe``     - the command line interface.
* ``SimpleWhisper-GUI.exe`` - the desktop window, without a console window.

Two environment variables change the result:

``SIMPLEWHISPER_ONEFILE=1``
    Build two single file executables instead of a folder. The files are
    much bigger and start slower, because everything is unpacked into a
    temporary folder on every run.

``SIMPLEWHISPER_DEBUG=1``
    Keep the console window open on errors and skip the compression.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# SPECPATH is the folder that holds this spec file; PyInstaller sets it.
ROOT = Path(SPECPATH).resolve()
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

ONEFILE = os.environ.get("SIMPLEWHISPER_ONEFILE", "0") == "1"
DEBUG = os.environ.get("SIMPLEWHISPER_DEBUG", "0") == "1"

# --------------------------------------------------------------------------- #
# Files that PyInstaller cannot find on its own
# --------------------------------------------------------------------------- #

# faster-whisper ships the Silero VAD model as a data file and looks for it
# next to its own modules, so it has to end up in faster_whisper/assets.
datas = collect_data_files("faster_whisper")

# The window icon, read by simplewhisper.gui at startup so the frozen
# window shows it instead of the Tk default.
_package_assets = SRC / "simplewhisper" / "assets"
if _package_assets.is_dir():
    datas += [(str(_package_assets), "simplewhisper/assets")]

# The native libraries of sherpa-onnx, ONNX Runtime and CTranslate2. Each
# one is bundled next to the module that loads it, which is where the
# Windows loader looks first.
binaries = (
    collect_dynamic_libs("sherpa_onnx")
    + collect_dynamic_libs("onnxruntime")
    + collect_dynamic_libs("ctranslate2")
)

# Modules that are loaded by name at runtime, or that are imported from
# somewhere PyInstaller does not follow.
hiddenimports = [
    # faster-whisper
    "faster_whisper",
    "faster_whisper.audio",
    "faster_whisper.assets",
    "faster_whisper.feature_extractor",
    "faster_whisper.tokenizer",
    "faster_whisper.transcribe",
    "faster_whisper.utils",
    "faster_whisper.vad",
    # The voice activity filter runs an ONNX model.
    "onnxruntime",
    "onnxruntime.capi.onnxruntime_inference_collection",
    # The decoding backend.
    "ctranslate2",
    # The audio decoder and the tokenizer.
    "av",
    "tokenizers",
    # Model downloads.
    "huggingface_hub",
    # The speaker diarization.
    "sherpa_onnx",
    "sherpa_onnx.lib._sherpa_onnx",
    # Drag and drop for the desktop window.
    "tkinterdnd2",
    # Progress bars.
    "tqdm",
    "tqdm.auto",
]

# Tools that are not needed at runtime.
excludes = [
    "IPython",
    "matplotlib",
    "notebook",
    "pytest",
    "ruff",
    "tkinter.test",
]

# An icon is optional; the build works fine without one.
_icon = ROOT / "assets" / "simplewhisper.ico"
icon = str(_icon) if _icon.is_file() else None

# The Windows version resource is written by scripts/make_version_info.py.
_version_file = SCRIPTS / "version_info.txt"
version = str(_version_file) if _version_file.is_file() else None

# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

analysis = Analysis(
    [str(SCRIPTS / "launcher.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(SCRIPTS / "hook_dll_path.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

common = dict(
    debug=DEBUG,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
    version=version,
)

if ONEFILE:
    # Each executable carries everything it needs.
    EXE(
        pyz,
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        [],
        name="SimpleWhisper",
        console=True,
        **common,
    )
    EXE(
        pyz,
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        [],
        name="SimpleWhisper-GUI",
        console=False,
        **common,
    )
else:
    # One folder with both executables and the shared libraries.
    cli = EXE(
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name="SimpleWhisper",
        console=True,
        **common,
    )
    gui = EXE(
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name="SimpleWhisper-GUI",
        console=False,
        **common,
    )
    COLLECT(
        cli,
        gui,
        analysis.binaries,
        analysis.datas,
        strip=False,
        upx=False,
        name="SimpleWhisper",
    )
