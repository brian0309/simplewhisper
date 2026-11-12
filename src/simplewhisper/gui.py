"""Tkinter desktop UI for SimpleWhisper.

Launch with ``simplewhisper gui`` or the ``SimpleWhisper-GUI.cmd`` launcher.
Everything runs locally; models are downloaded on first use just like the CLI.

The window is drawn with the hand-written light/dark theme from ``theme.py``,
which follows the operating system colour scheme and can be switched at any
time from the title bar.
"""

from __future__ import annotations

import ctypes
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

try:  # Optional: native drag & drop into the window.
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except Exception:  # pragma: no cover - depends on the environment
    DND_FILES = None
    TkinterDnD = None
    HAS_DND = False

from .audio import find_ffmpeg, prepare_media
from .config import (
    DEFAULT_MODEL,
    LANGUAGES,
    MEDIA_EXTENSIONS,
    default_model_dir,
    format_bytes,
    format_clock,
    language_label,
    resolve_language,
)
from .cuda import activate_cublas, install_cublas, plan_gpu
from .diarize import assign_speakers, diarization_models_cached
from .errors import JobCancelled
from .models import (
    MODEL_NAMES,
    cached_model_names,
    download_diarization_models,
    download_model_files,
    model_cache_path,
)
from .output import expand_formats, write_outputs
from .playback import SPEEDS, AudioPlayer, PlaybackError
from .speakers import open_speaker_service
from .theme import (
    Card,
    FileList,
    Label,
    Surface,
    Theme,
    Transcript,
    enable_dpi_awareness,
    system_prefers_dark,
)
from .transcribe import load_model, transcribe_media

AUTO_LABEL = "Auto-detect"
LANGUAGE_CHOICES = [AUTO_LABEL] + [
    f"{name.title()} ({code})" for code, name in sorted(LANGUAGES.items(), key=lambda item: item[1])
]
FORMAT_CHOICES = ["srt", "vtt", "txt", "json", "tsv", "all"]

# Upper bound on the lines kept in the log pane, so a long recording does
# not grow the widget without limit.
MAX_LOG_LINES = 5000

# How long the log keeps scrolling on its own after the reader scrolled away.
AUTO_SCROLL_DELAY = 3.0


@dataclass
class SegmentRef:
    """One transcript line and where it sits in the log."""

    start: float
    end: float
    start_mark: str
    end_mark: str
    media: "MediaEntry" = None


@dataclass
class MediaEntry:
    """One transcribed file and the lines that belong to it."""

    path: Path
    name: str
    label: str = ""
    duration: float = 0.0
    segments: List[SegmentRef] = field(default_factory=list)


def _media_filetypes():
    patterns = " ".join("*" + extension for extension in sorted(MEDIA_EXTENSIONS))
    return [("Audio and video", patterns), ("All files", "*.*")]


def _parse_language_choice(value: str) -> Optional[str]:
    text = value.strip()
    if not text or text.lower() == AUTO_LABEL.lower():
        return None
    if text.endswith(")") and "(" in text:
        text = text[text.rfind("(") + 1 : -1]
    return resolve_language(text)


def _format_bytes(count: float) -> str:
    return format_bytes(count)


def _format_clock(seconds: float) -> str:
    return format_clock(seconds)


def _format_speed(speed: float) -> str:
    """Label a playback speed, e.g. ``1x`` or ``1.25x``."""
    if float(speed).is_integer():
        return f"{int(speed)}x"
    return f"{speed:g}x"


def _parse_speed(value: str) -> Optional[float]:
    """Read a speed label back, so ``2x`` becomes ``2.0``."""
    text = value.strip().lower().rstrip("x")
    try:
        return float(text)
    except ValueError:
        return None


def _speaker_summary(segments) -> str:
    """Summarize how much speaking time each speaker got."""
    totals: dict = {}
    for segment in segments:
        if not segment.speaker:
            continue
        totals[segment.speaker] = totals.get(segment.speaker, 0.0) + (
            segment.end - segment.start
        )
    if not totals:
        return "No speakers were detected."
    parts = [
        f"{name} {_format_clock(seconds)}" for name, seconds in totals.items()
    ]
    return f"Speakers detected: {len(totals)} ({', '.join(parts)})"


class App:
    def __init__(self, root: tk.Tk, theme: Theme) -> None:
        self.root = root
        self.theme = theme
        self.files: List[Path] = []
        self.messages: "queue.Queue[tuple]" = queue.Queue()
        self.running = False
        self._model = None
        self._model_name = ""
        self._output_folder: Optional[Path] = None
        self._placeholder = True
        self._cancel_event = threading.Event()
        self._cancelled = False

        # The mini player: the audio of the transcribed files and the lines
        # of the transcript, so the log can follow along.
        self._player = AudioPlayer()
        self._media: List[MediaEntry] = []
        self._active: Optional[MediaEntry] = None
        self._segments: List[SegmentRef] = []
        self._highlight: Optional[SegmentRef] = None
        self._segment_count = 0
        self._log_scrolled_at = 0.0
        self._slider_syncing = False
        self._seeking = False
        self._was_playing = False
        self._player_error = ""

        self._build()
        self._poll_messages()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        theme = self.theme
        palette = theme.palette

        self.root.title("SimpleWhisper")
        self.root.geometry("1120x780")
        self.root.minsize(940, 640)
        self.root.configure(background=palette.bg)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        shell = Surface(self.root, theme)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        # ---------------------------------------------------------- header
        header = Surface(shell, theme)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 14))
        header.columnconfigure(0, weight=1)

        titles = Surface(header, theme)
        titles.grid(row=0, column=0, sticky="w")

        Label(
            titles,
            theme,
            "SimpleWhisper",
            role="text",
            size="title",
        ).pack(anchor="w")
        Label(
            titles,
            theme,
            "Local speech-to-text — audio never leaves this computer",
            role="muted",
            size="small",
        ).pack(anchor="w", pady=(4, 0))

        self.theme_button = ttk.Button(
            header,
            text="Light" if theme.dark else "Dark",
            style="Ghost.TButton",
            width=7,
            command=self._toggle_theme,
        )
        self.theme_button.grid(row=0, column=1, sticky="e")

        # ------------------------------------------------------------ body
        body = Surface(shell, theme)
        body.grid(row=1, column=0, sticky="nsew", padx=24)
        body.columnconfigure(0, weight=0, minsize=400)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = Surface(body, theme)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        right = Surface(body, theme)
        right.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_files_card(left)
        self._build_settings_card(left)
        self._build_transcript_card(right)
        self._build_actions(right)
        self._build_status_bar(shell)

        self._bind_shortcuts()

        # Paint every widget with the current palette, so nothing keeps a
        # system default colour (a white border, for instance) on startup.
        self._apply_theme()

    def _build_files_card(self, parent: tk.Misc) -> None:
        theme = self.theme
        card = Card(parent, theme, "Source files")
        card.grid(row=0, column=0, sticky="nsew")

        ttk.Button(
            card.actions, text="Add files…", style="TButton", command=self._add_files
        ).pack(side="right")

        self.file_list = FileList(card.body, theme)
        self.file_list.grid(row=0, column=0, sticky="nsew")
        file_scroll = ttk.Scrollbar(
            card.body,
            orient="vertical",
            command=self.file_list.yview,
            style="Surface.Vertical.TScrollbar",
        )
        file_scroll.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=file_scroll.set)

        buttons = Surface(card.body, theme, role="surface")
        buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(
            buttons,
            text="Remove",
            style="SurfaceGhost.TButton",
            command=self._remove_selected,
        ).pack(side="left")
        ttk.Button(
            buttons, text="Clear", style="SurfaceGhost.TButton", command=self._clear_files
        ).pack(side="left", padx=(8, 0))

        hint = (
            "Drag & drop files onto the list"
            if HAS_DND
            else "Drag & drop needs the extra: pip install -e \".[gui]\""
        )
        Label(buttons, theme, hint, role="faint", size="small").pack(side="right")

        if HAS_DND and hasattr(self.file_list, "drop_target_register"):
            try:
                self.file_list.drop_target_register(DND_FILES)
                self.file_list.dnd_bind("<<Drop>>", self._on_drop)
                self.file_list.dnd_bind("<<DragEnter>>", self._on_drag_over)
                self.file_list.dnd_bind("<<DragLeave>>", self._on_drag_leave)
            except Exception:  # pragma: no cover - depends on the environment
                pass

    def _build_settings_card(self, parent: tk.Misc) -> None:
        theme = self.theme
        card = Card(parent, theme, "Settings")
        card.grid(row=1, column=0, sticky="ew", pady=(16, 0))

        form = Surface(card.body, theme, role="surface")
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(0, weight=0, minsize=92)
        form.columnconfigure(1, weight=1)

        Label(form, theme, "Model", role="muted", size="small").grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 10)
        )
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        model_row = Surface(form, theme, role="surface")
        model_row.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        model_row.columnconfigure(0, weight=1)
        self.model_box = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=list(MODEL_NAMES),
            state="readonly",
            style="Surface.TCombobox",
        )
        self.model_box.grid(row=0, column=0, sticky="ew")
        ttk.Button(
            model_row,
            text="Model files…",
            style="SurfaceGhost.TButton",
            command=self._open_models,
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        Label(form, theme, "Language", role="muted", size="small").grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=(0, 10)
        )
        self.language_var = tk.StringVar(value=AUTO_LABEL)
        self.language_box = ttk.Combobox(
            form,
            textvariable=self.language_var,
            values=LANGUAGE_CHOICES,
            style="Surface.TCombobox",
        )
        self.language_box.grid(row=1, column=1, sticky="ew", pady=(0, 8))

        Label(form, theme, "Format", role="muted", size="small").grid(
            row=2, column=0, sticky="w", padx=(0, 12), pady=(0, 8)
        )
        self.format_var = tk.StringVar(value="srt")
        ttk.Combobox(
            form,
            textvariable=self.format_var,
            values=FORMAT_CHOICES,
            state="readonly",
            style="Surface.TCombobox",
        ).grid(row=2, column=1, sticky="ew", pady=(0, 8))

        Label(form, theme, "Output", role="muted", size="small").grid(
            row=3, column=0, sticky="w", padx=(0, 12), pady=(0, 8)
        )
        output_row = Surface(form, theme, role="surface")
        output_row.grid(row=3, column=1, sticky="ew", pady=(0, 8))
        output_row.columnconfigure(0, weight=1)
        self.output_var = tk.StringVar()
        ttk.Entry(
            output_row, textvariable=self.output_var, style="Surface.TEntry"
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            output_row,
            text="Browse…",
            style="SurfaceGhost.TButton",
            command=self._browse_output,
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        Label(form, theme, "", role="muted", size="small").grid(
            row=4, column=0, sticky="w", padx=(0, 12)
        )
        speakers_row = Surface(form, theme, role="surface")
        speakers_row.grid(row=4, column=1, sticky="ew", pady=(2, 0))
        self.speakers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            speakers_row,
            text="Break down by speaker",
            variable=self.speakers_var,
            style="Surface.TCheckbutton",
            command=self._toggle_speakers,
        ).pack(side="left")
        self.speaker_count_var = tk.StringVar(value="")
        try:
            self.speaker_count_entry = ttk.Spinbox(
                speakers_row,
                from_=1,
                to=20,
                width=4,
                textvariable=self.speaker_count_var,
                state="disabled",
                style="Surface.TSpinbox",
            )
        except tk.TclError:  # pragma: no cover - very old Tk
            self.speaker_count_entry = ttk.Entry(
                speakers_row,
                textvariable=self.speaker_count_var,
                width=4,
                state="disabled",
                style="Surface.TEntry",
            )
        self.speaker_count_entry.pack(side="left", padx=(10, 6))
        Label(speakers_row, theme, "speakers (optional)", role="faint", size="small").pack(
            side="left"
        )

        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            form,
            text="Overwrite existing transcripts",
            variable=self.overwrite_var,
            style="Surface.TCheckbutton",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

        Label(
            form,
            theme,
            "Leave the output folder empty to write next to each input.",
            role="faint",
            size="small",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # The speaker count only matters when the breakdown is enabled.
        self._toggle_speakers()

    def _build_transcript_card(self, parent: tk.Misc) -> None:
        theme = self.theme
        card = Card(parent, theme, "Transcript")
        card.grid(row=0, column=0, sticky="nsew")

        ttk.Button(
            card.actions,
            text="Copy",
            style="SurfaceGhost.TButton",
            command=self._copy_log,
        ).pack(side="right")
        ttk.Button(
            card.actions,
            text="Clear",
            style="SurfaceGhost.TButton",
            command=self._clear_log,
        ).pack(side="right", padx=(0, 8))

        log_body = Surface(card.body, theme, role="surface")
        log_body.grid(row=0, column=0, sticky="nsew")
        log_body.columnconfigure(0, weight=1)
        log_body.rowconfigure(0, weight=1)

        self.log_text = Transcript(log_body, theme)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(
            log_body,
            orient="vertical",
            command=self.log_text.yview,
            style="Surface.Vertical.TScrollbar",
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_scroll = log_scroll

        # The log follows the audio: click a line to jump to it.
        self.log_text.bind("<Button-1>", self._on_log_click)
        self.log_text.bind("<space>", self._on_log_space)
        self.log_text.bind("<MouseWheel>", self._on_log_scroll)
        self.log_text.bind("<Button-4>", self._on_log_scroll)
        self.log_text.bind("<Button-5>", self._on_log_scroll)
        log_scroll.bind("<Button-1>", self._on_log_scroll)

        self._build_player_bar(card.body)
        self._clear_log()

    def _build_player_bar(self, parent: tk.Misc) -> None:
        """The mini player under the log.

        It stays out of the way until there is a transcript to review.
        """
        theme = self.theme
        bar = Surface(parent, theme, role="surface")
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        bar.columnconfigure(4, weight=1)

        self.media_var = tk.StringVar()
        self.media_box = ttk.Combobox(
            bar,
            textvariable=self.media_var,
            values=(),
            state="readonly",
            width=10,
            style="Surface.TCombobox",
        )
        self.media_box.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.media_box.bind("<<ComboboxSelected>>", self._on_media_selected)

        self.play_button = ttk.Button(
            bar, text="Play", style="Player.TButton", command=self._toggle_play
        )
        self.play_button.grid(row=0, column=1, sticky="w")

        self.stop_button = ttk.Button(
            bar,
            text="Stop",
            style="Player.TButton",
            command=self._stop_playback,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=2, sticky="w", padx=(8, 12))

        self.speed_var = tk.StringVar(value=_format_speed(1.0))
        self.speed_box = ttk.Combobox(
            bar,
            textvariable=self.speed_var,
            values=[_format_speed(speed) for speed in SPEEDS],
            state="readonly",
            width=5,
            style="Surface.TCombobox",
        )
        self.speed_box.grid(row=0, column=3, sticky="w", padx=(0, 14))
        self.speed_box.bind("<<ComboboxSelected>>", self._on_speed_selected)

        self.seek_scale = ttk.Scale(
            bar,
            from_=0,
            to=1,
            orient="horizontal",
            style="Player.Horizontal.TScale",
            command=self._on_seek_drag,
        )
        self.seek_scale.grid(row=0, column=4, sticky="ew")
        self.seek_scale.bind("<Button-1>", self._on_seek_press)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)

        self.time_var = tk.StringVar(value="00:00 / 00:00")
        Label(bar, theme, textvariable=self.time_var, role="muted", size="small").grid(
            row=0, column=5, sticky="e", padx=(14, 0)
        )

        self.player_bar = bar
        bar.grid_remove()

    def _build_actions(self, parent: tk.Misc) -> None:
        theme = self.theme
        action = Surface(parent, theme)
        action.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        action.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(
            action, text="Start transcription", style="Accent.TButton", command=self._start
        )
        self.start_button.grid(row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(
            action, mode="determinate", maximum=1.0, style="Accent.Horizontal.TProgressbar"
        )
        self.progress.grid(row=0, column=1, sticky="ew", padx=(18, 18))

        ttk.Button(
            action, text="Open output folder", style="Ghost.TButton", command=self._open_output
        ).grid(row=0, column=2, sticky="e")

        self.cancel_button = ttk.Button(
            action,
            text="Cancel",
            style="Ghost.TButton",
            command=self._cancel,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=3, sticky="e", padx=(10, 0))

    def _build_status_bar(self, parent: tk.Misc) -> None:
        theme = self.theme
        bar = Surface(parent, theme)
        bar.grid(row=2, column=0, sticky="ew", padx=24, pady=(12, 18))
        bar.columnconfigure(1, weight=1)

        self.status_dot = Label(bar, theme, "●", role="faint", size="small")
        self.status_dot.grid(row=0, column=0, sticky="w", padx=(0, 9))

        self.status_var = tk.StringVar(value="Ready")
        Label(bar, theme, textvariable=self.status_var, role="muted", size="small").grid(
            row=0, column=1, sticky="w"
        )

        Label(
            bar,
            theme,
            "Ctrl+O add files   ·   Ctrl+Shift+M model files   ·   "
            "Ctrl+Enter start   ·   Del remove",
            role="faint",
            size="small",
        ).grid(row=0, column=2, sticky="e")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _event: self._add_files())
        self.root.bind("<Control-Shift-M>", lambda _event: self._open_models())
        self.root.bind("<Control-Return>", lambda _event: self._start())
        self.root.bind("<Control-KP_Enter>", lambda _event: self._start())
        self.file_list.bind("<Delete>", lambda _event: self._remove_selected())
        self.file_list.bind("<BackSpace>", lambda _event: self._remove_selected())
        # A drag that ends outside the scrubber still has to finish the seek.
        self.root.bind("<ButtonRelease-1>", self._on_seek_release)

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #

    def _toggle_theme(self) -> None:
        """Switch between the light and the dark colour scheme."""
        self.theme = Theme(self.root, dark=not self.theme.dark)
        self.root.configure(background=self.theme.palette.bg)
        self._apply_theme()
        self.theme_button.configure(text="Light" if self.theme.dark else "Dark")
        self._refresh_status_dot()

    def _apply_theme(self) -> None:
        """Push the current theme onto every themed widget of the window."""
        for child in self.root.winfo_children():
            if hasattr(child, "refresh"):
                child.refresh(self.theme)

    def _refresh_status_dot(self) -> None:
        self.status_dot.apply_theme(self.theme)

    def _set_status(self, text: str, role: str = "muted") -> None:
        self.status_var.set(text)
        self.status_dot.role = role
        self.status_dot.apply_theme(self.theme)

    # ------------------------------------------------------------------ #
    # File handling
    # ------------------------------------------------------------------ #

    def _add_files(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Choose audio or video files",
            filetypes=_media_filetypes(),
        )
        if selected:
            self._add_paths([Path(path) for path in selected])

    def _add_paths(self, paths: List[Path]) -> None:
        added = 0
        for path in paths:
            path = path.expanduser()
            if path.is_file() and path not in self.files:
                self.files.append(path)
                self.file_list.insert("end", str(path))
                added += 1
        if added:
            self.file_list.set_empty(False)
            self._set_status(f"{len(self.files)} file(s) selected")

    def _on_drop(self, event) -> None:
        try:
            entries = self.root.tk.splitlist(event.data)
        except tk.TclError:
            entries = [event.data]
        self._add_paths([Path(entry) for entry in entries])

    def _on_drag_over(self, _event=None) -> None:
        self.file_list.configure(highlightbackground=self.theme.palette.accent)

    def _on_drag_leave(self, _event=None) -> None:
        self.file_list.configure(highlightbackground=self.theme.palette.border)

    def _remove_selected(self) -> None:
        for index in reversed(self.file_list.curselection()):
            self.file_list.delete(index)
            del self.files[index]
        self.file_list.set_empty(not self.files)
        self._set_status(f"{len(self.files)} file(s) selected")

    def _clear_files(self) -> None:
        self.file_list.delete(0, "end")
        self.files.clear()
        self.file_list.set_empty(True)
        self._set_status("Ready")

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose an output folder")
        if folder:
            self.output_var.set(folder)

    def _toggle_speakers(self) -> None:
        """Enable the speaker count box only when diarization is on."""
        state = "normal" if self.speakers_var.get() else "disabled"
        self.speaker_count_entry.configure(state=state)

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #

    def _start(self) -> None:
        if self.running:
            return
        if not self.files:
            messagebox.showwarning("No files", "Add at least one audio or video file.")
            return

        try:
            language = _parse_language_choice(self.language_var.get())
            formats = expand_formats([self.format_var.get()])
        except ValueError as exc:
            messagebox.showerror("Invalid option", str(exc))
            return

        output_text = self.output_var.get().strip().strip('"').strip("'")
        output_dir = Path(output_text).expanduser() if output_text else None
        if output_dir is not None:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Invalid output folder", str(exc))
                return

        files = list(self.files)
        model_name = self.model_var.get()
        overwrite = self.overwrite_var.get()
        speakers = self.speakers_var.get()

        # Work out whether the GPU can be used. The question about the
        # download is asked here, on the main thread, so the window stays
        # responsive while the libraries are fetched.
        device = "cpu"
        cuda_download = False
        plan = plan_gpu("auto")
        if plan.needs_download:
            answer = messagebox.askyesno(
                "Download the CUDA libraries?",
                "An NVIDIA GPU was found, but the CUDA libraries it needs are "
                "missing.\n\n"
                "They are 548 MB and are downloaded once from NVIDIA.\n"
                "Without them the transcription runs on the CPU.",
            )
            cuda_download = bool(answer)
            if cuda_download:
                device = "cuda"
            else:
                self._log("CUDA libraries not downloaded; using the CPU.", tag="muted")
        elif plan.device == "cuda":
            device = "cuda"
            if plan.folder is not None:
                activate_cublas(plan.folder)

        num_speakers = 0
        if speakers:
            raw_count = self.speaker_count_var.get().strip()
            if raw_count:
                if not raw_count.isdigit() or int(raw_count) <= 0:
                    messagebox.showerror(
                        "Invalid speaker count", "Enter a whole number above zero."
                    )
                    return
                num_speakers = int(raw_count)

        self._output_folder = output_dir or files[0].parent

        self.running = True
        self.start_button.configure(state="disabled", text="Transcribing…")
        self.cancel_button.configure(state="normal")
        self._cancel_event.clear()
        self._cancelled = False
        self.progress.configure(value=0)
        self._set_status("Starting…", role="accent")
        self._clear_log(placeholder=False)
        self._log(
            f"Model: {model_name}  |  Language: {language_label(language)}  |  "
            f"Format: {', '.join(formats)}"
        )
        if speakers:
            expected = str(num_speakers) if num_speakers else "auto"
            self._log(f"Speaker breakdown: on ({expected} speakers)")
        if not find_ffmpeg():
            self._log("ffmpeg not found; using the built-in decoder.")

        threading.Thread(
            target=self._worker,
            args=(
                files,
                model_name,
                language,
                formats,
                output_dir,
                overwrite,
                speakers,
                num_speakers,
                device,
                cuda_download,
            ),
            daemon=True,
        ).start()

    def _worker(
        self,
        files: List[Path],
        model_name: str,
        language: Optional[str],
        formats: List[str],
        output_dir: Optional[Path],
        overwrite: bool,
        speakers: bool = False,
        num_speakers: int = 0,
        device: str = "cpu",
        cuda_download: bool = False,
    ) -> None:
        failures = 0
        diarizer = None
        try:
            if cuda_download:
                self.messages.put(("status", "Downloading the CUDA libraries..."))
                self.messages.put(
                    (
                        "log",
                        "Downloading the CUDA libraries (548 MB, first run only)...",
                    )
                )
                install_cublas(
                    progress=lambda done, total: self.messages.put(
                        ("cuda_download", done, total)
                    )
                )
                self.messages.put(("cuda_download_done",))
            if self._model is None or self._model_name != model_name:
                model_dir = default_model_dir()
                if model_cache_path(model_name, model_dir) is None:
                    self.messages.put(("status", f"Downloading model '{model_name}'..."))
                    self.messages.put(
                        ("log", f"Downloading model '{model_name}' (first run only)...")
                    )
                    download_model_files(
                        model_name,
                        model_dir,
                        progress=lambda done, total: self.messages.put(
                            ("download", model_name, done, total)
                        ),
                    )
                    self.messages.put(("download_done", model_name))
                self.messages.put(("status", f"Loading model '{model_name}'..."))
                self._model = load_model(
                    model_name, device=device, model_dir=model_dir
                )
                self._model_name = model_name
            self.messages.put(("status", "Model ready."))

            if speakers:
                model_dir = default_model_dir()
                if not diarization_models_cached(model_dir):
                    self.messages.put(("status", "Downloading speaker models..."))
                    self.messages.put(
                        ("log", "Downloading speaker models (first run only)...")
                    )
                    download_diarization_models(model_dir)
                    self.messages.put(("log", "Speaker models downloaded."))
                self.messages.put(("status", "Loading speaker models..."))
                diarizer = open_speaker_service(model_dir)
                if diarizer.runs_in_process:
                    self.messages.put(
                        ("log", "Speaker models ready (separate process).")
                    )
                else:
                    self.messages.put(
                        (
                            "log",
                            "Speaker models ready (this process; the window may "
                            "pause while the speakers are detected).",
                        )
                    )
                self.messages.put(("status", "Speaker models ready."))

            for index, source in enumerate(files, start=1):
                self.messages.put(
                    ("file", index, len(files), source.name, str(source))
                )
                media = source
                temporary = False
                try:
                    try:
                        media, temporary = prepare_media(source)
                        started = time.monotonic()
                        result = transcribe_media(
                            self._model,
                            media,
                            language=language,
                            word_timestamps=diarizer is not None,
                            progress=self._progress_callback(),
                        )
                        elapsed = time.monotonic() - started
                        self.messages.put(
                            (
                                "log",
                                f"Transcribed {source.name} in {elapsed:.1f}s.",
                            )
                        )
                        self.messages.put(
                            ("media_done", str(source), float(result.duration))
                        )
                        if diarizer is not None:
                            turns = self._detect_speakers(
                                diarizer, media, num_speakers
                            )
                            result.segments = assign_speakers(result.segments, turns)
                            self.messages.put(("speakers", _speaker_summary(result.segments)))
                        result.source = source
                        result.model = model_name
                    finally:
                        if temporary and media != source:
                            media.unlink(missing_ok=True)

                    if language is None and result.language:
                        self.messages.put(
                            (
                                "log",
                                f"Detected language: {language_label(result.language)} "
                                f"({result.language_probability:.0%})",
                            )
                        )
                    written, skipped = write_outputs(
                        result, formats, output_dir, overwrite=overwrite
                    )
                    self.messages.put(("written", written, skipped))
                except JobCancelled:
                    self.messages.put(("log", "Cancelled.", "muted"))
                    break
                except Exception as exc:  # keep going with the other files
                    failures += 1
                    self.messages.put(("error", f"{source.name}: {exc}"))
        except JobCancelled:
            self.messages.put(("log", "Cancelled.", "muted"))
        except Exception as exc:
            failures = max(failures, 1)
            self.messages.put(("error", str(exc)))
        finally:
            if diarizer is not None:
                diarizer.close()
            self.messages.put(("done", failures, len(files)))

    def _detect_speakers(self, diarizer, media, num_speakers):
        """Run the speaker detection and report how it is going."""
        self.messages.put(("status", "Detecting speakers…"))
        self.messages.put(("progress", 0.0))
        started = time.monotonic()
        turns = diarizer.diarize(
            media,
            num_speakers=num_speakers,
            progress=lambda fraction: self.messages.put(
                ("speaker_progress", fraction)
            ),
            cancel=self._cancel_event.is_set,
        )
        elapsed = time.monotonic() - started
        speakers = len({turn.speaker for turn in turns})
        self.messages.put(
            (
                "log",
                f"Speaker detection finished in {elapsed:.1f}s "
                f"({len(turns)} turns, {speakers} speaker(s)).",
            )
        )
        return turns

    def _progress_callback(self):
        last_update = [0.0]

        def callback(segment, info) -> None:
            if self._cancel_event.is_set():
                raise JobCancelled("the transcription was cancelled")

            # Transcript lines go to the log immediately so the text is visible
            # while the file is still being processed.
            text = segment.text.strip()
            if text:
                self.messages.put(
                    ("segment", float(segment.start), float(segment.end), text)
                )

            now = time.monotonic()
            if now - last_update[0] < 0.12:
                return
            last_update[0] = now
            total = float(getattr(info, "duration", 0.0) or 0.0)
            fraction = min(segment.end / total, 1.0) if total else 0.0
            self.messages.put(("progress", fraction))

        return callback

    def _poll_messages(self) -> None:
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message[0]

                if kind == "status":
                    self._set_status(message[1], role="accent")
                elif kind == "file":
                    _, index, total, name, path = message
                    self._set_status(f"Transcribing {index}/{total}: {name}", role="accent")
                    self._log(f"[{index}/{total}] {name}", tag="file")
                    self.progress.configure(value=0)
                    self._begin_media(Path(path), name)
                elif kind == "download":
                    _, name, done, total = message
                    fraction = min(done / total, 1.0) if total else 0.0
                    self.progress.configure(value=fraction)
                    text = f"Downloading model '{name}': {fraction:.0%}"
                    if total:
                        text += f" ({_format_bytes(done)} / {_format_bytes(total)})"
                    self._set_status(text, role="accent")
                elif kind == "download_done":
                    _, name = message
                    self.progress.configure(value=0)
                    self._set_status(f"Model '{name}' downloaded.", role="accent")
                    self._log(f"Model '{name}' downloaded.")
                elif kind == "cuda_download":
                    _, done, total = message
                    fraction = min(done / total, 1.0) if total else 0.0
                    self.progress.configure(value=fraction)
                    text = f"Downloading the CUDA libraries: {fraction:.0%}"
                    if total:
                        text += f" ({_format_bytes(done)} / {_format_bytes(total)})"
                    self._set_status(text, role="accent")
                elif kind == "cuda_download_done":
                    self.progress.configure(value=0)
                    self._set_status("CUDA libraries downloaded.", role="accent")
                    self._log("CUDA libraries downloaded.")
                elif kind == "progress":
                    self.progress.configure(value=message[1])
                elif kind == "speaker_progress":
                    fraction = message[1]
                    self.progress.configure(value=fraction)
                    self._set_status(f"Detecting speakers: {fraction:.0%}", role="accent")
                elif kind == "segment":
                    _, start, end, text = message
                    self._log_segment(start, end, text)
                elif kind == "media_done":
                    _, path, duration = message
                    self._set_media_duration(path, duration)
                elif kind == "speakers":
                    self._log(message[1], tag="file")
                elif kind == "written":
                    _, written, skipped = message
                    for path in written:
                        self._log(f"Wrote {path}", tag="ok")
                    for path in skipped:
                        self._log(f"Skipped existing {path} (enable Overwrite)", tag="muted")
                elif kind == "log":
                    self._log(message[1], message[2] if len(message) > 2 else "")
                elif kind == "error":
                    self._log(message[1], tag="error")
                elif kind == "done":
                    _, failures, total = message
                    self._finish(failures, total)
        except queue.Empty:
            pass

        self._update_player()
        self.root.after(100, self._poll_messages)

    def _finish(self, failures: int, total: int) -> None:
        self.running = False
        self.start_button.configure(state="normal", text="Start transcription")
        self.cancel_button.configure(state="disabled")
        if self._cancelled:
            self.progress.configure(value=0)
            self._set_status("Cancelled.", role="muted")
        elif failures:
            self.progress.configure(value=0)
            self._set_status(f"Finished with {failures} error(s).", role="danger")
        else:
            self.progress.configure(value=1.0)
            self._set_status(f"Done. {total} file(s) transcribed.", role="success")

    def _cancel(self) -> None:
        """Ask the running job to stop."""
        if not self.running:
            return
        self._cancelled = True
        self._cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self._set_status("Cancelling…", role="accent")
        self._log("Cancelling…", tag="muted")

    # ------------------------------------------------------------------ #
    # Log
    # ------------------------------------------------------------------ #

    def _clear_log(self, placeholder: bool = True) -> None:
        self._reset_player()
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self._placeholder = placeholder
        if placeholder:
            self.log_text.insert(
                "end", "Transcript and progress will appear here.\n", "muted"
            )
        self.log_text.configure(state="disabled")

    def _copy_log(self) -> None:
        text = self.log_text.get("1.0", "end-1c").strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Transcript copied to the clipboard.", role="success")

    def _log(self, text: str, tag: str = "") -> None:
        self.log_text.configure(state="normal")
        self._drop_placeholder()
        if tag:
            self.log_text.insert("end", text.rstrip() + "\n", tag)
        else:
            self.log_text.insert("end", text.rstrip() + "\n")
        self._trim_log()
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log_segment(self, start: float, end: float, text: str) -> None:
        """Write one transcript line and remember where it sits in the log."""
        self.log_text.configure(state="normal")
        self._drop_placeholder()

        index = self._segment_count
        self._segment_count += 1
        # A throwaway tag marks the text that is being written, so the exact
        # position of the line is known without counting characters.
        tag = f"segment{index}"
        self.log_text.insert("end", f"[{_format_clock(start)}]", ("time", tag))
        self.log_text.insert("end", f" {text.strip()}\n", ("segment", tag))
        ranges = self.log_text.tag_ranges(tag)
        if len(ranges) >= 2:
            start_mark, end_mark = f"seg{index}s", f"seg{index}e"
            self.log_text.mark_set(start_mark, ranges[0])
            self.log_text.mark_gravity(start_mark, "left")
            self.log_text.mark_set(end_mark, ranges[1])
            self.log_text.mark_gravity(end_mark, "left")
            self.log_text.tag_remove(tag, ranges[0], ranges[1])
        else:  # pragma: no cover - the text widget always reports a range
            start_mark = end_mark = ""

        self._trim_log()
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

        segment = SegmentRef(
            start=start,
            end=end,
            start_mark=start_mark,
            end_mark=end_mark,
            media=self._active,
        )
        self._segments.append(segment)
        if self._active is not None:
            self._active.segments.append(segment)
        self._show_player()
        self._refresh_player_controls()

    def _drop_placeholder(self) -> None:
        if not self._placeholder:
            return
        self.log_text.delete("1.0", "end")
        self._placeholder = False

    def _trim_log(self) -> None:
        """Drop the oldest lines so a long transcript cannot grow forever."""
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{lines - MAX_LOG_LINES + 1}.0")

    # ------------------------------------------------------------------ #
    # Mini player
    # ------------------------------------------------------------------ #

    def _reset_player(self) -> None:
        """Forget the transcript and stop the playback."""
        self._player.stop()
        self._media.clear()
        self._active = None
        self._segments.clear()
        self._highlight = None
        self._segment_count = 0
        self._player_error = ""
        self.log_text.set_highlight(None, None)
        self.media_box.configure(values=())
        self.media_var.set("")
        self.time_var.set("00:00 / 00:00")
        self.player_bar.grid_remove()

    def _show_player(self) -> None:
        """The player appears once there is a transcript to review."""
        if not self.player_bar.winfo_ismapped():
            self.player_bar.grid()

    def _begin_media(self, path: Path, name: str) -> None:
        """Point the player at the file that is about to be transcribed."""
        self._player.stop()
        entry = MediaEntry(path=path, name=name)
        entry.label = self._unique_label(name)
        self._media.append(entry)
        self._active = entry
        self._highlight = None
        self.log_text.set_highlight(None, None)
        self._player.load(path)
        self._player.prepare()
        self._refresh_media_box()
        self._refresh_player_controls()

    def _set_media_duration(self, path: str, duration: float) -> None:
        """Remember how long a transcribed file is."""
        entry = next((item for item in self._media if str(item.path) == path), None)
        if entry is not None:
            entry.duration = float(duration)
        if self._active is not None and str(self._active.path) == path:
            self._player.set_duration(duration)
            self._refresh_player_controls()

    def _unique_label(self, name: str) -> str:
        """A label for the file picker that is not used twice."""
        label = name
        number = 2
        while any(entry.label == label for entry in self._media):
            label = f"{name} ({number})"
            number += 1
        return label

    def _refresh_media_box(self) -> None:
        self.media_box.configure(values=[entry.label for entry in self._media])
        if self._active is not None:
            self.media_var.set(self._active.label)

    def _refresh_player_controls(self) -> None:
        """Match the buttons and the scrubber to the state of the player."""
        playing = self._player.is_playing
        self.play_button.configure(text="Pause" if playing else "Play")

        entry = self._active
        ready = bool(entry and entry.segments)
        self.play_button.configure(state="normal" if ready else "disabled")
        self.stop_button.configure(
            state="normal" if (playing or self._player.position > 0) else "disabled"
        )
        self.seek_scale.configure(to=max(self._player.duration, 0.001))

    def _set_player_status(self, text: str, role: str = "muted") -> None:
        """Show a message, unless a transcription is reporting its own."""
        if self.running:
            return
        self._set_status(text, role)

    def _toggle_play(self) -> None:
        """Start or pause the playback."""
        if self._player.is_playing:
            self._player.pause()
            self._refresh_player_controls()
            self._set_player_status("Paused.", role="muted")
            return
        self._start_playback()

    def _start_playback(self) -> None:
        try:
            self._player.play()
        except PlaybackError as exc:
            self._log(str(exc), tag="error")
            self._set_player_status("Playback failed.", role="danger")
            return
        self._refresh_player_controls()
        self._set_player_status("Playing.", role="accent")

    def _stop_playback(self) -> None:
        self._player.stop()
        self._set_highlight(None)
        self._refresh_player_controls()
        self._update_player()
        self._set_player_status("Stopped.", role="muted")

    def _seek_to(self, seconds: float) -> None:
        """Move the playback to a point in the audio."""
        entry = self._active
        if entry is None or not entry.segments:
            return
        try:
            self._player.seek(seconds)
        except PlaybackError as exc:
            self._log(str(exc), tag="error")
            self._set_player_status("Playback failed.", role="danger")
            return
        self._set_highlight(self._find_segment(self._player.position))

    def _on_seek_press(self, _event=None) -> None:
        """Hold the audio still while the reader drags the scrubber."""
        self._seeking = True
        self._was_playing = self._player.is_playing
        if self._was_playing:
            self._player.pause()
            self._refresh_player_controls()

    def _on_seek_drag(self, value: str) -> None:
        if self._slider_syncing or self._seeking:
            return
        try:
            self._seek_to(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    def _on_seek_release(self, _event=None) -> None:
        if not self._seeking:
            return
        self._seeking = False
        try:
            target = float(self.seek_scale.get())
        except (TypeError, ValueError):  # pragma: no cover - defensive
            target = 0.0
        self._seek_to(target)
        if self._was_playing:
            self._start_playback()

    def _on_speed_selected(self, _event=None) -> None:
        """Change the playback speed, keeping the place in the audio."""
        speed = _parse_speed(self.speed_var.get())
        if speed is None:
            return
        try:
            self._player.set_speed(speed)
        except PlaybackError as exc:
            self._log(str(exc), tag="error")
            self._set_player_status("Playback failed.", role="danger")
            return
        self._refresh_player_controls()

    def _on_media_selected(self, _event=None) -> None:
        """Review another file that was transcribed earlier."""
        label = self.media_var.get()
        entry = next((item for item in self._media if item.label == label), None)
        if entry is None or entry is self._active:
            return
        self._player.stop()
        self._active = entry
        self._highlight = None
        self.log_text.set_highlight(None, None)
        self._player.load(entry.path, entry.duration)
        self._player.prepare()
        self._refresh_player_controls()
        self._update_player()

    def _update_player(self) -> None:
        """Follow the audio and highlight the line that is being spoken."""
        error = self._player.error
        if error and error != self._player_error:
            self._player_error = error
            self._log(error, tag="error")
            self._set_player_status("Playback is not available.", role="danger")

        status = self._player.poll()
        if status == "failed":
            self._refresh_player_controls()
            self._set_player_status("Playback failed.", role="danger")
            self._log("The player stopped with an error.", tag="error")
        elif status == "finished":
            self._refresh_player_controls()
            self._set_player_status("Playback finished.", role="muted")

        position = self._player.position
        duration = self._player.duration
        if not self._seeking:
            # The scrubber is left alone while the reader drags it.
            self._slider_syncing = True
            try:
                self.seek_scale.set(position)
            finally:
                self._slider_syncing = False
        self.time_var.set(f"{_format_clock(position)} / {_format_clock(duration)}")

        segment = self._find_segment(position)
        if segment is not self._highlight:
            self._set_highlight(segment)

    def _find_segment(self, position: float) -> Optional[SegmentRef]:
        """The transcript line that is being spoken at ``position``."""
        entry = self._active
        if entry is None or not entry.segments:
            return None

        segments = entry.segments
        low, high = 0, len(segments) - 1
        found = None
        while low <= high:
            middle = (low + high) // 2
            segment = segments[middle]
            if segment.start <= position:
                found = segment
                low = middle + 1
            else:
                high = middle - 1
        return found

    def _set_highlight(self, segment: Optional[SegmentRef]) -> None:
        """Move the highlight to another line of the transcript."""
        self._highlight = segment
        if segment is None or not segment.start_mark:
            self.log_text.set_highlight(None, None)
            return

        self.log_text.set_highlight(segment.start_mark, segment.end_mark)
        # The log only scrolls by itself while the reader does not scroll.
        if time.monotonic() - self._log_scrolled_at > AUTO_SCROLL_DELAY:
            self.log_text.see(segment.start_mark)

    def _segment_at_index(self, index: str) -> Optional[SegmentRef]:
        """Find the transcript line that covers a spot in the log."""
        segments = self._segments
        low, high = 0, len(segments) - 1
        candidate = None
        while low <= high:
            middle = (low + high) // 2
            segment = segments[middle]
            if not segment.start_mark:
                return None
            try:
                if self.log_text.compare(segment.start_mark, "<=", index):
                    candidate = segment
                    low = middle + 1
                else:
                    high = middle - 1
            except tk.TclError:  # pragma: no cover - the marks are gone
                return None

        if candidate is None:
            return None
        try:
            if self.log_text.compare(index, "<=", candidate.end_mark):
                return candidate
        except tk.TclError:  # pragma: no cover - the marks are gone
            return None
        return None

    def _on_log_click(self, event) -> None:
        """Jump to the line that was clicked."""
        index = self.log_text.index(f"@{event.x},{event.y}")
        segment = self._segment_at_index(index)
        if segment is None:
            return

        media = segment.media
        if media is not None and media is not self._active:
            self._player.stop()
            self._active = media
            self._highlight = None
            self.log_text.set_highlight(None, None)
            self._player.load(media.path, media.duration)
            self._player.prepare()
            self.media_var.set(media.label)
            self._refresh_player_controls()

        self._seek_to(segment.start)

    def _on_log_space(self, _event=None) -> None:
        """Play or pause with the space bar."""
        self._toggle_play()
        return "break"

    def _on_log_scroll(self, _event=None) -> None:
        """Remember that the reader took over the scrolling."""
        self._log_scrolled_at = time.monotonic()

    def _on_close(self) -> None:
        """Stop the audio and close the window."""
        self._player.stop()
        self.root.destroy()

    def _open_output(self) -> None:
        folder = self._output_folder
        if folder is None or not folder.is_dir():
            messagebox.showinfo("No output folder", "Run a transcription first.")
            return
        self._reveal_folder(folder)

    def _open_models(self) -> None:
        """Show the folder that holds the downloaded model files.

        The folder is created when it is not there yet, so the path can be
        copied into a file manager before the first model is downloaded.
        """
        folder = default_model_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Could not open model files", str(exc))
            return

        cached = len(cached_model_names(folder))
        note = (
            f"{cached} model(s) downloaded"
            if cached
            else "nothing downloaded yet"
        )
        self._set_status(f"Model files: {folder}  ({note})")
        self._reveal_folder(folder)

    def _reveal_folder(self, folder: Path) -> None:
        """Open a folder in the file manager of the operating system."""
        try:
            if os.name == "nt":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            messagebox.showerror("Could not open folder", str(exc))


def _icon_candidates() -> List[Path]:
    """Places the window icon may live, most specific first.

    The copy inside the package is the one an installed build or the
    frozen executable ships; the one at the repository root is what a
    source checkout has before anything is installed.
    """
    package = Path(__file__).resolve().parent
    return [
        package / "assets" / "simplewhisper.ico",
        package.parent.parent / "assets" / "simplewhisper.ico",
    ]


def _use_native_windows_icon(root: tk.Tk, ico: Path) -> None:
    """Give Windows native-size icons read straight from the .ico file.

    Tk's own ``iconbitmap`` call answers the system with one 32x32 icon
    built from the 16 pixel artwork stretched up, so the task bar and the
    title bar both show a resample of a small picture, which looks
    blurry. Loading the file at the sizes Windows asks for (16 and 32,
    doubled on high DPI screens) keeps every size pixel crisp.

    A toplevel only gets its final window handle once it is mapped, and
    Tk writes its own icons at that moment as well, so this runs again
    on ``<Map>`` and once more shortly after.
    """
    WM_SETICON = 0x80
    GCLP_HICON, GCLP_HICONSM = -14, -34
    ICON_SMALL, ICON_BIG = 0, 1
    IMAGE_ICON, LR_LOADFROMFILE = 1, 0x10

    user32 = ctypes.windll.user32
    user32.LoadImageW.argtypes = (
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    )
    user32.LoadImageW.restype = ctypes.c_void_p
    user32.IsWindow.argtypes = (ctypes.c_void_p,)
    user32.IsWindow.restype = ctypes.c_bool
    user32.SendMessageW.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_size_t,
    )
    user32.SendMessageW.restype = ctypes.c_void_p
    user32.SetClassLongPtrW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
    user32.SetClassLongPtrW.restype = ctypes.c_size_t

    def load(edge: int) -> int:
        handle = user32.LoadImageW(0, str(ico), IMAGE_ICON, edge, edge, LR_LOADFROMFILE)
        return int(handle or 0)

    def apply(*_event: object) -> None:
        try:
            hwnd = int(root.wm_frame(), 16)
            if not hwnd or not user32.IsWindow(hwnd):
                return

            icons = getattr(root, "_simplewhisper_icons", None)
            if icons is None:
                dpi = int(round(float(root.winfo_fpixels("1i"))))
                icons = (
                    load(max(16, int(round(16 * dpi / 96)))),
                    load(max(32, int(round(32 * dpi / 96)))),
                )
                if not all(icons):
                    return  # keep the fallback from iconbitmap
                root._simplewhisper_icons = icons

            small, big = icons
            # The window's own icons are what the title bar and the task
            # bar show; the class icons are the fallback dialogs use.
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
            user32.SetClassLongPtrW(hwnd, GCLP_HICON, big)
            user32.SetClassLongPtrW(hwnd, GCLP_HICONSM, small)
        except (ctypes.ArgumentError, OSError, tk.TclError, ValueError):
            pass  # better the default icon than a launch that fails

    root.bind(
        "<Map>", lambda event: apply() if event.widget is root else None, add="+"
    )
    apply()
    root.after(100, apply)


def set_window_icon(root: tk.Tk) -> None:
    """Use the SimpleWhisper icon instead of the Tk default.

    Windows gets the .ico loaded at native sizes (see above), other
    platforms the .png through ``iconphoto``. A missing file only means
    the default icon, so a failure here never stops the window opening.
    """
    ico = next((path for path in _icon_candidates() if path.is_file()), None)
    if ico is None:
        return

    if os.name == "nt":
        try:
            root.iconbitmap(str(ico))  # fallback if the native path fails
        except (OSError, tk.TclError):
            pass
        _use_native_windows_icon(root, ico)
        return

    png = ico.with_suffix(".png")
    if png.is_file():
        try:
            # Kept on the window so the photo is not garbage collected
            # while Tk still needs it for the title bar icon.
            root._simplewhisper_icon = tk.PhotoImage(file=str(png))
            root.iconphoto(True, root._simplewhisper_icon)
        except (OSError, tk.TclError):
            pass


def main() -> int:
    enable_dpi_awareness()

    try:
        root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    except Exception:
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            print(f"error: could not start the GUI: {exc}", file=sys.stderr)
            return 1

    set_window_icon(root)

    try:
        App(root, Theme(root, dark=system_prefers_dark()))
        root.mainloop()
    except tk.TclError as exc:  # e.g. no display on Linux
        print(f"error: could not start the GUI: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
