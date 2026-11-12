"""Runtime hook: point the Windows loader at the bundled native libraries.

PyInstaller keeps the DLLs of a package in that package's own folder inside
the bundle. Windows finds the dependencies of a DLL next to that DLL only
when the DLL is loaded through the module that needs it, which is not always
the case. ONNX Runtime, CTranslate2 and sherpa-onnx each ship their own
copy of some libraries, so the folders are added to the DLL search path
explicitly. That keeps one package from loading another package' version.

This hook runs before the entry script, so the libraries are in place by the
time anything is imported.
"""

import os
import sys

# The handles have to stay alive for as long as the search path entry is
# needed; the directory is removed again when the handle is collected.
_handles = []


def _bundle_folders():
    """The folders in the bundle that hold native libraries."""
    bundle = getattr(sys, "_MEIPASS", "")
    if not bundle:
        return []
    return [
        bundle,
        os.path.join(bundle, "onnxruntime", "capi"),
        os.path.join(bundle, "ctranslate2"),
        os.path.join(bundle, "sherpa_onnx", "lib"),
        os.path.join(bundle, "tokenizers"),
    ]


if sys.platform == "win32":
    for _folder in _bundle_folders():
        if not os.path.isdir(_folder):
            continue
        try:
            _handles.append(os.add_dll_directory(_folder))
        except (AttributeError, OSError):
            pass
        # Keep the folder reachable for the tools that only look at PATH.
        os.environ["PATH"] = _folder + os.pathsep + os.environ.get("PATH", "")
