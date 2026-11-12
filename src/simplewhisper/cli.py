"""Interactive and flag-driven command line interface."""

from __future__ import annotations

import argparse
import glob as globlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import __version__
from .audio import prepare_media
from .config import (
    APP_NAME,
    DEFAULT_FORMAT,
    DEFAULT_MODEL,
    DISPLAY_NAME,
    MEDIA_EXTENSIONS,
    OUTPUT_FORMATS,
    default_model_dir,
    format_bytes,
    language_label,
    resolve_language,
)
from .cuda import (
    CUBLAS_SIZE,
    GpuUnavailable,
    gpu_report,
    install_cublas,
    plan_gpu,
    prepare_gpu,
)
from .diarize import DEFAULT_THRESHOLD as DEFAULT_SPEAKER_THRESHOLD
from .diarize import (
    assign_speakers,
    diarization_models_cached,
)
from .models import (
    MODEL_CATALOG,
    MODEL_NAMES,
    cached_model_names,
    download_diarization_models,
    download_model_files,
    model_cache_path,
    model_table_rows,
)
from .output import expand_formats, write_outputs
from .speakers import open_speaker_service
from .transcribe import load_model, transcribe_media

log = logging.getLogger(APP_NAME)

# Name used with "simplewhisper models download" for the speaker models.
SPEAKERS_MODEL_NAME = "speakers"


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simplewhisper",
        description=(
            "Transcribe audio and video locally with Whisper. "
            "Everything runs on your machine; models are downloaded on first use."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  simplewhisper\n"
            "      Interactive wizard. Drag & drop a file when prompted.\n"
            "  simplewhisper lecture.mp4 -m small -l en -f srt\n"
            "      Transcribe one file non-interactively.\n"
            "  simplewhisper talk.mkv -m large-v3-turbo -f all -o transcripts\n"
            "      Write SRT, VTT, TXT, JSON and TSV into ./transcripts.\n"
            "  simplewhisper podcast.mp3 -l auto -f txt\n"
            "      Auto-detect the language and write plain text.\n"
            "  simplewhisper interview.mp4 -t translate -f srt\n"
            "      Translate speech to English subtitles.\n"
            "  simplewhisper panel.mp4 -S -f txt\n"
            "      Break the transcript down by speaker.\n"
            "  simplewhisper models download small\n"
            "      Download a model now so the first run works offline.\n"
            "  simplewhisper cuda\n"
            "      Show whether the GPU can be used.\n"
            "  simplewhisper gui\n"
            "      Open the desktop window instead of using the terminal wizard.\n"
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="audio or video files; omit to be prompted",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=MODEL_NAMES,
        help=f"model size (default: {DEFAULT_MODEL}; prompted when omitted)",
    )
    parser.add_argument(
        "-l",
        "--language",
        metavar="LANG",
        help="language code or name, e.g. en or spanish; empty or 'auto' to detect",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="formats",
        metavar="FMT",
        help="srt, vtt, txt, json, tsv, or all; comma-separate for several",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="folder for transcripts (default: next to each input file)",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute device (default: auto)",
    )
    parser.add_argument(
        "-c",
        "--compute-type",
        default="default",
        choices=("default", "int8", "int8_float16", "float16", "float32"),
        help="numeric precision (default: int8 on CPU, float16 on GPU)",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="translate translates speech to English",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help=f"model cache folder (default: {default_model_dir()})",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        metavar="N",
        help="decoding beam size; higher is slower and usually more accurate (default: 5)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="disable voice activity detection (keeps silence)",
    )
    parser.add_argument(
        "-S",
        "--speakers",
        action="store_true",
        help="break the transcript down by speaker (Speaker 1, Speaker 2, ...)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=0,
        metavar="N",
        help="how many speakers to expect; 0 lets the model decide (default: 0)",
    )
    parser.add_argument(
        "--speaker-threshold",
        type=float,
        default=DEFAULT_SPEAKER_THRESHOLD,
        metavar="T",
        help=(
            "speaker clustering threshold; lower values split into more speakers "
            f"(default: {DEFAULT_SPEAKER_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="skip ffmpeg and use the built-in PyAV decoder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing transcript files",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="never prompt; use the flags and defaults as given",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _parse_formats(value: str) -> List[str]:
    items = [
        part.strip().lower()
        for part in value.replace(";", ",").split(",")
        if part.strip()
    ]
    if not items:
        raise ValueError(
            f"no output format given. Choose from: {', '.join(OUTPUT_FORMATS)}, all"
        )
    for item in items:
        if item not in OUTPUT_FORMATS and item != "all":
            raise ValueError(
                f"unknown format '{item}'. Choose from: {', '.join(OUTPUT_FORMATS)}, all"
            )
    return expand_formats(items)


# --------------------------------------------------------------------------- #
# Terminal input helpers
# --------------------------------------------------------------------------- #


def _split_paths(raw: str, posix: Optional[bool] = None) -> List[str]:
    """Split drag & drop input while respecting quotes and shell escapes.

    Windows paths keep their backslashes; on POSIX systems a backslash escapes
    the next character, which is how macOS Terminal inserts dragged paths.
    """
    if posix is None:
        posix = os.name != "nt"

    tokens: List[str] = []
    buffer: List[str] = []
    quote: Optional[str] = None
    escaped = False

    for character in raw:
        if escaped:
            buffer.append(character)
            escaped = False
        elif quote:
            if character == quote:
                quote = None
            else:
                buffer.append(character)
        elif posix and character == "\\":
            escaped = True
        elif character in {'"', "'"}:
            quote = character
        elif character.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
        else:
            buffer.append(character)

    if escaped:
        buffer.append("\\")
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _expand_inputs(raw_values: Sequence[str]) -> Tuple[List[Path], List[str]]:
    """Resolve CLI/wizard inputs, expanding globs and dropping duplicates."""
    paths: List[Path] = []
    missing: List[str] = []

    for raw in raw_values:
        cleaned = raw.strip().strip('"').strip("'")
        if not cleaned:
            continue
        has_wildcard = any(character in cleaned for character in "*?[")
        matches = globlib.glob(cleaned) if has_wildcard else [cleaned]
        found = False
        for match in matches:
            candidate = Path(match).expanduser()
            if candidate.is_file():
                if candidate not in paths:
                    paths.append(candidate)
                found = True
        if not found:
            missing.append(cleaned)

    return paths, missing


def _ask(prompt: str) -> str:
    return input(prompt).strip()


def _choose(prompt: str, options: List[Tuple[str, str]], default_index: int = 0) -> str:
    print(prompt)
    for index, (value, label) in enumerate(options, start=1):
        marker = "  (default)" if index - 1 == default_index else ""
        print(f"  {index}) {label}{marker}")
    while True:
        raw = _ask(f"Choice [1-{len(options)}] ({default_index + 1}): ")
        if not raw:
            return options[default_index][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        for value, _ in options:
            if raw.lower() == value:
                return value
        print("  Invalid choice; enter a number or the value shown.")


def _prompt_files() -> List[Path]:
    print()
    while True:
        raw = _ask(
            "Drag & drop audio/video file(s) here, then press Enter:\n> "
        )
        if not raw:
            print("  Nothing entered. Try again.")
            continue
        paths, missing = _expand_inputs(_split_paths(raw))
        for item in missing:
            print(f"  Not found (or not a file): {item}")
        if paths:
            return paths
        print("  No readable files. Try again.")


def _prompt_model() -> str:
    print()
    options: List[Tuple[str, str]] = []
    for info in MODEL_CATALOG:
        label = (
            f"{info.name:<17} {info.params:>6} params  {info.download:>8}  "
            f"{info.memory:>6} RAM  {info.note}"
        )
        options.append((info.name, label))
    default_index = MODEL_NAMES.index(DEFAULT_MODEL)
    return _choose("Choose a model:", options, default_index)


def _prompt_language() -> Optional[str]:
    print()
    print("Language: enter a code (en, es, ja, ...) or a name (english, spanish, ...).")
    while True:
        raw = _ask("Language [auto]: ")
        try:
            return resolve_language(raw)
        except ValueError as exc:
            print(f"  {exc}")


def _prompt_formats() -> List[str]:
    print()
    options = [
        ("srt", "srt  - SubRip subtitles, works almost everywhere"),
        ("vtt", "vtt  - WebVTT subtitles for browsers and web players"),
        ("txt", "txt  - plain text, one segment per line"),
        ("json", "json - segments plus language and duration metadata"),
        ("tsv", "tsv  - tab-separated start/end/text"),
        ("all", "all  - write every format above"),
    ]
    format_name = _choose("Choose an output format:", options, 0)
    return expand_formats([format_name])


def _prompt_output_dir() -> Optional[Path]:
    print()
    raw = _ask("Output folder (empty = next to each input): ")
    if not raw:
        return None
    return Path(raw.strip().strip('"').strip("'")).expanduser()


def _prompt_speakers() -> Tuple[bool, int]:
    """Ask whether to break the transcript down by speaker."""
    print()
    while True:
        raw = _ask("Break the transcript down by speaker? [y/N]: ").lower()
        if not raw or raw in {"n", "no"}:
            return False, 0
        if raw in {"y", "yes"}:
            break
        print("  Please answer y or n.")

    while True:
        raw_count = _ask("How many speakers? (empty = let the model decide): ")
        raw_count = raw_count.strip()
        if not raw_count:
            return True, 0
        if raw_count.isdigit() and int(raw_count) > 0:
            return True, int(raw_count)
        print("  Enter a whole number above zero, or leave it empty.")


def _confirm() -> bool:
    while True:
        raw = _ask("Proceed? [Y/n]: ").lower()
        if not raw or raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  Please answer y or n.")


def _confirm(question: str) -> bool:
    """Ask a yes or no question on the terminal."""
    print(f"\n{question}")
    while True:
        raw = input("Download them now? [y/N]: ").strip().lower()
        if not raw or raw in {"n", "no"}:
            return False
        if raw in {"y", "yes"}:
            return True
        print("  Please answer y or no.")


def _print_summary(
    files: Sequence[Path],
    model_name: str,
    language: Optional[str],
    formats: Sequence[str],
    output_dir: Optional[Path],
    speakers: bool = False,
    num_speakers: int = 0,
) -> None:
    print("\nSummary:")
    print(f"  Files ({len(files)}):")
    for path in files:
        print(f"    {path}")
    print(f"  Model:    {model_name}")
    print(f"  Language: {language_label(language)}")
    print(f"  Format:   {', '.join(formats)}")
    if speakers:
        expected = f"{num_speakers}" if num_speakers else "auto"
        print(f"  Speakers: yes ({expected})")
    print(f"  Output:   {output_dir if output_dir else 'same folder as each input'}")


# --------------------------------------------------------------------------- #
# Transcription run
# --------------------------------------------------------------------------- #


class _ProgressPrinter:
    """Single-line progress display on stderr."""

    def __init__(self) -> None:
        self._last_update = 0.0
        self._active = False

    def update(self, segment, info) -> None:
        now = time.monotonic()
        if now - self._last_update < 0.1:
            return
        self._last_update = now

        total = float(getattr(info, "duration", 0.0) or 0.0)
        percentage = (
            f"{min(segment.end / total, 1.0) * 100:5.1f}% " if total else "  ... "
        )
        width = max(20, shutil.get_terminal_size((100, 24)).columns - 24)
        text = segment.text.replace("\n", " ").strip()
        sys.stderr.write(f"\r  {percentage}{text[:width]:<{width}}")
        sys.stderr.flush()
        self._active = True

    def finish(self) -> None:
        if not self._active:
            return
        width = shutil.get_terminal_size((100, 24)).columns
        sys.stderr.write("\r" + " " * max(0, width - 1) + "\r")
        sys.stderr.flush()
        self._active = False


class _SpeakerPrinter:
    """Single-line progress display for the speaker detection on stderr."""

    def __init__(self) -> None:
        self._last_update = 0.0
        self._active = False

    def update(self, fraction: float) -> None:
        now = time.monotonic()
        if now - self._last_update < 0.15 and fraction < 1.0:
            return
        self._last_update = now
        sys.stderr.write(f"\r  detecting speakers: {fraction * 100:5.1f}%")
        sys.stderr.flush()
        self._active = True

    def finish(self) -> None:
        if not self._active:
            return
        width = shutil.get_terminal_size((100, 24)).columns
        sys.stderr.write("\r" + " " * max(0, width - 1) + "\r")
        sys.stderr.flush()
        self._active = False


def _short_duration(seconds: float) -> str:
    if seconds >= 60:
        minutes, remainder = divmod(int(round(seconds)), 60)
        return f"{minutes}m {remainder}s"
    return f"{seconds:.1f}s"


class _DownloadPrinter:
    """Single-line model download progress on stderr."""

    def __init__(self) -> None:
        self._last_update = 0.0
        self._active = False
        self._finished = False

    def update(self, done: int, total: int) -> None:
        if self._finished or total <= 0:
            return
        fraction = min(done / total, 1.0)
        now = time.monotonic()
        # Always show the final state, even if it arrives within the
        # throttle window of the previous update.
        if fraction < 1.0 and now - self._last_update < 0.1:
            return
        self._last_update = now
        sys.stderr.write(
            f"\r  downloading: {fraction * 100:5.1f}% "
            f"({format_bytes(done)} / {format_bytes(total)})"
        )
        sys.stderr.flush()
        self._active = True

    def finish(self) -> None:
        self._finished = True
        if not self._active:
            return
        width = shutil.get_terminal_size((100, 24)).columns
        sys.stderr.write("\r" + " " * max(0, width - 1) + "\r")
        sys.stderr.flush()
        self._active = False


def run(args: argparse.Namespace) -> int:
    files, missing = _expand_inputs(args.files)
    for item in missing:
        print(f"error: input not found: {item}", file=sys.stderr)

    interactive = not args.yes and sys.stdin.isatty()
    needs_wizard = interactive and (
        not files
        or args.model is None
        or args.language is None
        or args.formats is None
    )

    if not files and not interactive:
        print("error: no input files.", file=sys.stderr)
        print("usage: simplewhisper FILE [FILE ...]   (see 'simplewhisper --help')", file=sys.stderr)
        return 2

    try:
        language = resolve_language(args.language)
        formats = _parse_formats(args.formats) if args.formats else [DEFAULT_FORMAT]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model_name = args.model or DEFAULT_MODEL
    output_dir = args.output_dir
    speakers = args.speakers
    num_speakers = args.num_speakers

    if needs_wizard:
        print(f"\n{DISPLAY_NAME} - local speech-to-text")
        print("-" * 42)
        if not files:
            files = _prompt_files()
        if args.model is None:
            model_name = _prompt_model()
        if args.language is None:
            language = _prompt_language()
        if args.formats is None:
            formats = _prompt_formats()
        if output_dir is None:
            output_dir = _prompt_output_dir()
        if not speakers:
            speakers, num_speakers = _prompt_speakers()

        _print_summary(files, model_name, language, formats, output_dir, speakers, num_speakers)
        if not _confirm():
            print("Cancelled.")
            return 1

    for source in files:
        if source.suffix.lower() not in MEDIA_EXTENSIONS:
            print(
                f"warning: '{source.name}' has an unrecognized extension; trying anyway",
                file=sys.stderr,
            )

    model_dir = (args.model_dir or default_model_dir()).expanduser()

    # Work out whether the GPU can be used. This may ask the user whether
    # the CUDA libraries may be downloaded; without an answer the CPU is
    # used instead.
    device = args.device
    if device != "cpu":
        downloader = _DownloadPrinter()
        try:
            device, note = prepare_gpu(
                device,
                ask=_confirm if interactive else None,
                progress=downloader.update,
            )
        except GpuUnavailable as exc:
            downloader.finish()
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            downloader.finish()
        if note:
            print(f"  {note}", file=sys.stderr)

    if model_cache_path(model_name, model_dir) is None:
        print(
            f"\nDownloading model '{model_name}' (this happens only once)...",
            file=sys.stderr,
        )
        downloader = _DownloadPrinter()
        try:
            download_model_files(model_name, model_dir, progress=downloader.update)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"error: could not download model '{model_name}': {exc}", file=sys.stderr)
            return 1
        finally:
            downloader.finish()

    print(f"\nLoading model '{model_name}'...", file=sys.stderr)
    started = time.monotonic()
    try:
        model = load_model(
            model_name,
            device=device,
            compute_type=args.compute_type,
            model_dir=model_dir,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"error: could not load model '{model_name}': {exc}", file=sys.stderr)
        if "cuda" in str(exc).lower():
            print(
                "hint: run 'simplewhisper cuda' to see what the GPU support found, "
                "or use -d cpu.",
                file=sys.stderr,
            )
        else:
            print(
                f"hint: run 'simplewhisper models download {model_name}' to fetch it "
                "first, or check your GPU drivers.",
                file=sys.stderr,
            )
        return 1
    print(f"Model ready in {_short_duration(time.monotonic() - started)}.", file=sys.stderr)

    failures = 0

    diarizer = None
    if speakers:
        if not diarization_models_cached(model_dir):
            print(
                "\nDownloading speaker models (this happens only once)...",
                file=sys.stderr,
            )
            downloader = _DownloadPrinter()
            try:
                download_diarization_models(model_dir, progress=downloader.update)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"error: could not download the speaker models: {exc}", file=sys.stderr)
                return 1
            finally:
                downloader.finish()

        try:
            diarizer = open_speaker_service(
                model_dir,
                threshold=args.speaker_threshold,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"error: could not load the speaker models: {exc}", file=sys.stderr)
            return 1
        print("  Speaker models ready.", file=sys.stderr)

    try:
        for index, source in enumerate(files, start=1):
            print(f"\n[{index}/{len(files)}] {source.name}", file=sys.stderr)
            media = source
            temporary = False
            printer = _ProgressPrinter()
            started = time.monotonic()

            try:
                media, temporary = prepare_media(source, use_ffmpeg=not args.no_ffmpeg)
                result = transcribe_media(
                    model,
                    media,
                    language=language,
                    task=args.task,
                    beam_size=args.beam_size,
                    vad_filter=not args.no_vad,
                    word_timestamps=diarizer is not None,
                    progress=printer.update,
                )
                if diarizer is not None:
                    speaker_printer = _SpeakerPrinter()
                    speaker_started = time.monotonic()
                    try:
                        turns = diarizer.diarize(
                            media,
                            num_speakers=num_speakers,
                            progress=speaker_printer.update,
                        )
                    finally:
                        speaker_printer.finish()
                    result.segments = assign_speakers(result.segments, turns)
                    print(
                        f"  speakers: {len(turns)} turn(s) in "
                        f"{_short_duration(time.monotonic() - speaker_started)}",
                        file=sys.stderr,
                    )
                # Keep the user's file as the source so output names and JSON
                # metadata refer to the original, not the temporary WAV.
                result.source = source
                result.model = model_name
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failures += 1
                print(f"  error: {exc}", file=sys.stderr)
                continue
            finally:
                printer.finish()
                if temporary and media != source:
                    media.unlink(missing_ok=True)

            if language is None and result.language:
                print(
                    f"  language: {language_label(result.language)} "
                    f"({result.language_probability:.0%})",
                    file=sys.stderr,
                )
            if not result.segments:
                print("  warning: no speech detected", file=sys.stderr)

            try:
                written, skipped = write_outputs(
                    result, formats, output_dir, overwrite=args.overwrite
                )
            except OSError as exc:
                failures += 1
                print(f"  error: could not write transcripts: {exc}", file=sys.stderr)
                continue

            for path in written:
                print(f"  wrote {path}")
            for path in skipped:
                print(f"  skipped existing {path} (use --overwrite)")
            print(
                f"  done in {_short_duration(time.monotonic() - started)}",
                file=sys.stderr,
            )
    finally:
        if diarizer is not None:
            diarizer.close()

    if failures:
        print(f"\n{failures} file(s) failed.", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Model management
# --------------------------------------------------------------------------- #


def models_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="simplewhisper models",
        description="List models and the local cache, or download models ahead of time.",
        epilog=(
            "examples:\n"
            "  simplewhisper models\n"
            "  simplewhisper models download small\n"
            "  simplewhisper models download large-v3-turbo --model-dir D:\\models\n"
            "  simplewhisper models download speakers\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("list", "download"),
        default="list",
        help="list cached models (default) or download them",
    )
    parser.add_argument("names", nargs="*", metavar="MODEL", help="model name(s) for download")
    parser.add_argument("--model-dir", type=Path, help="model cache folder")
    args = parser.parse_args(list(argv))

    model_dir = (args.model_dir or default_model_dir()).expanduser()

    if args.action == "download":
        if not args.names:
            parser.error("name at least one model, e.g. simplewhisper models download small")
        unknown = [
            name
            for name in args.names
            if name not in MODEL_NAMES and name != SPEAKERS_MODEL_NAME
        ]
        if unknown:
            parser.error(
                f"unknown model(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(MODEL_NAMES)}, {SPEAKERS_MODEL_NAME}"
            )

        failures = 0
        for name in dict.fromkeys(args.names):
            # The progress line goes to stderr, so flush stdout first to keep
            # the two streams in order when the output is piped.
            print(f"Downloading '{name}' to {model_dir} ...", flush=True)
            downloader = _DownloadPrinter()
            try:
                if name == SPEAKERS_MODEL_NAME:
                    path = download_diarization_models(
                        model_dir, progress=downloader.update
                    )
                else:
                    path = download_model_files(
                        name, model_dir, progress=downloader.update
                    )
                downloader.finish()
                print(f"  ready: {path}", flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                downloader.finish()
                failures += 1
                print(f"  error: {exc}", file=sys.stderr)
        return 1 if failures else 0

    cached = cached_model_names(model_dir)
    rows = model_table_rows(cached)
    headers = ("MODEL", "PARAMS", "DOWNLOAD", "RAM", "NOTES")
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    print(f"Model cache: {model_dir}\n")
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))

    speakers_cached = diarization_models_cached(model_dir)
    speaker_row = (
        "speakers",
        "-",
        "~33 MB",
        "~200 MB",
        "speaker diarization" + (" *" if speakers_cached else ""),
    )
    print("  ".join(value.ljust(width) for value, width in zip(speaker_row, widths)))

    print("\n* = already cached locally.")
    print("Pre-download with: simplewhisper models download <model> [<model> ...]")
    return 0


def cuda_command(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="simplewhisper cuda",
        description="Show what the GPU support found, and install the CUDA libraries.",
        epilog=(
            "examples:\n"
            "  simplewhisper cuda\n"
            "      Show whether the GPU can be used.\n"
            "  simplewhisper cuda --download\n"
            "      Download the CUDA libraries (548 MB) and check again.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the CUDA libraries from NVIDIA",
    )
    args = parser.parse_args(list(argv))

    if args.download:
        print(f"Downloading the CUDA libraries ({CUBLAS_SIZE // (1024 * 1024)} MB)...")
        printer = _DownloadPrinter()
        try:
            folder = install_cublas(progress=printer.update)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            printer.finish()
            print(f"  error: {exc}", file=sys.stderr)
            return 1
        printer.finish()
        print(f"  ready: {folder}")

    print("\nGPU support:\n")
    for label, value in gpu_report():
        print(f"  {label:<17} {value}")

    plan = plan_gpu("auto")
    print()
    if plan.device == "cuda":
        print("The GPU will be used.")
    else:
        print(f"The CPU will be used ({plan.reason}).")
    if plan.needs_download:
        print(
            "Run 'simplewhisper cuda --download' to fetch the CUDA libraries, "
            "or let SimpleWhisper ask for them on the next run."
        )
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if not verbose:
        # Download progress bars are enough; hide hub-internal chatter.
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    arguments = list(sys.argv[1:] if argv is None else argv)

    # Keep first-run downloads quiet on Windows, where symlink caching is
    # often unavailable; files are still cached correctly.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if arguments and arguments[0] == "models":
        return models_command(arguments[1:])

    if arguments and arguments[0] == "cuda":
        return cuda_command(arguments[1:])

    if arguments and arguments[0] == "gui":
        from .gui import main as gui_main

        return gui_main()

    parser = build_parser()
    args = parser.parse_args(arguments)
    _configure_logging(args.verbose)

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
