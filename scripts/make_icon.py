"""Render assets/simplewhisper.svg into the Windows icon assets.

Run it after editing the SVG:

    python scripts/make_icon.py

Needs Pillow and resvg-py:

    pip install pillow resvg-py
"""

from __future__ import annotations

import io
import shutil
import struct
from pathlib import Path
from typing import List, Tuple

import resvg_py
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "assets" / "simplewhisper.svg"
PNG = ROOT / "assets" / "simplewhisper.png"
ICO = ROOT / "assets" / "simplewhisper.ico"
# The window looks for the icon inside its own package, so the rendered
# files are copied there as well.
PACKAGE = ROOT / "src" / "simplewhisper" / "assets"

# Windows picks the closest size; 16 and 32 are the ones Explorer and the
# taskbar use most, 256 is what the installer and the Start menu show.
SIZES = [16, 24, 32, 48, 64, 128, 256]


def render(size: int) -> Image.Image:
    """One size, rendered straight from the SVG so every entry is crisp."""
    png = resvg_py.svg_to_bytes(svg_path=str(SVG), width=size, height=size)
    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


def ico_bytes(images: List[Tuple[int, Image.Image]]) -> bytes:
    """Pack PNG-compressed images into an .ico container."""
    header = struct.pack("<HHH", 0, 1, len(images))
    entries = b""
    blobs = b""
    offset = len(header) + 16 * len(images)

    for size, image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        blob = buffer.getvalue()
        entries += struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,  # 0 means 256 in the directory entry
            0 if size >= 256 else size,
            0,  # palette size: none
            0,  # reserved
            1,  # colour planes
            32,  # bits per pixel
            len(blob),
            offset,
        )
        blobs += blob
        offset += len(blob)

    return header + entries + blobs


def main() -> int:
    if not SVG.is_file():
        print(f"missing {SVG}")
        return 1

    images = [(size, render(size)) for size in SIZES]

    images[-1][1].save(PNG)
    ICO.write_bytes(ico_bytes(images))

    PACKAGE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PNG, PACKAGE / PNG.name)
    shutil.copyfile(ICO, PACKAGE / ICO.name)

    print(f"png: {PNG}")
    print(f"ico: {ICO} ({', '.join(str(size) for size in SIZES)})")
    print(f"package copy: {PACKAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
