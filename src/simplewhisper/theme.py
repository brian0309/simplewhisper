"""Light and dark theme for the SimpleWhisper desktop UI.

Tkinter only ships a handful of dated platform themes, so the look of the
window is built by hand here: one palette per theme, a set of ``ttk`` styles
built on top of the ``clam`` theme, and a few composite widgets (cards, the
file list and the transcript log).

The theme follows the operating system colour scheme by default and can be
switched by the user at runtime.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import font as tkfont
from tkinter import ttk
from typing import Dict, Optional, Sequence, Tuple

# Fonts that look native on the platforms we support, best first.
UI_FONT_CANDIDATES = (
    "Segoe UI Variable Text",
    "Segoe UI",
    "SF Pro Text",
    "SF Pro Display",
    "Helvetica Neue",
    "Inter",
    "Ubuntu",
    "Cantarell",
    "Noto Sans",
    "DejaVu Sans",
)

MONO_FONT_CANDIDATES = (
    "Cascadia Mono",
    "Cascadia Code",
    "SF Mono",
    "Consolas",
    "Menlo",
    "JetBrains Mono",
    "Ubuntu Mono",
    "DejaVu Sans Mono",
)


def system_prefers_dark() -> bool:
    """Return ``True`` when the operating system is set to a dark scheme."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return not int(value)
        except Exception:
            return False

    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return False
        return result.stdout.strip().lower() == "dark"

    if "dark" in os.environ.get("GTK_THEME", "").lower():
        return True

    try:
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return False
    return "dark" in result.stdout.lower()


def _pick_family(root: tk.Misc, candidates: Sequence[str]) -> str:
    """Return the first available font family from ``candidates``."""
    try:
        available = {name.lower() for name in tkfont.families(root)}
    except tk.TclError:  # pragma: no cover - no usable font server
        available = set()

    for candidate in candidates:
        if candidate.lower() in available:
            return candidate

    try:
        return tkfont.nametofont("TkDefaultFont").actual("family")
    except tk.TclError:  # pragma: no cover - defensive
        return "TkDefaultFont"


@dataclass(frozen=True)
class Palette:
    """Every colour used by the application."""

    name: str
    bg: str
    surface: str
    surface_alt: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_faint: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_muted: str
    on_accent: str
    danger: str
    success: str
    warning: str
    selection: str
    selection_text: str
    hover: str
    entry_bg: str
    log_bg: str
    scrollbar: str
    scrollbar_hover: str
    track: str
    highlight: str


LIGHT = Palette(
    name="light",
    bg="#f4f5f8",
    surface="#ffffff",
    surface_alt="#eceef3",
    border="#e1e4ec",
    border_strong="#c8cdd9",
    text="#121722",
    text_muted="#5b6473",
    text_faint="#8a92a1",
    accent="#4f46e5",
    accent_hover="#4338ca",
    accent_pressed="#3730a3",
    accent_muted="#c9cbf7",
    on_accent="#ffffff",
    danger="#dc2626",
    success="#15803d",
    warning="#b45309",
    selection="#e0e7ff",
    selection_text="#1e1b4b",
    hover="#eef0f5",
    entry_bg="#ffffff",
    log_bg="#fbfbfd",
    scrollbar="#c9cfdb",
    scrollbar_hover="#a9b1c2",
    track="#e6e9ef",
    highlight="#d7d9fb",
)

DARK = Palette(
    name="dark",
    bg="#0d1017",
    surface="#161b24",
    surface_alt="#1e242f",
    border="#272e3b",
    border_strong="#3a4351",
    text="#e8ebf1",
    text_muted="#9aa5b5",
    text_faint="#6b7686",
    accent="#6366f1",
    accent_hover="#7c7ff5",
    accent_pressed="#4f46e5",
    accent_muted="#33365c",
    on_accent="#ffffff",
    danger="#f87171",
    success="#4ade80",
    warning="#fbbf24",
    selection="#2b3350",
    selection_text="#e8ebf1",
    hover="#1f2530",
    entry_bg="#11151d",
    log_bg="#10141b",
    scrollbar="#39424f",
    scrollbar_hover="#4b5666",
    track="#232a36",
    highlight="#3b4068",
)


class Theme:
    """Colours, fonts and ``ttk`` styles for one colour scheme."""

    def __init__(self, root: tk.Misc, dark: bool = False) -> None:
        self.root = root
        self.dark = dark
        self.palette = DARK if dark else LIGHT
        self.style = ttk.Style(root)

        family = _pick_family(root, UI_FONT_CANDIDATES)
        mono = _pick_family(root, MONO_FONT_CANDIDATES)
        self.fonts: Dict[str, tuple] = {
            "title": (family, 19, "bold"),
            "section": (family, 10, "bold"),
            "body": (family, 10),
            "body_bold": (family, 10, "bold"),
            "small": (family, 9),
            "small_bold": (family, 9, "bold"),
            "mono": (mono, 10),
            "mono_small": (mono, 9),
            "mono_bold": (mono, 10, "bold"),
        }
        self._thumb = None

        self.apply()

    # ------------------------------------------------------------------ #
    # Style helpers
    # ------------------------------------------------------------------ #

    def _build_scale_thumb(self, palette: Palette) -> None:
        """Give the scrubber a solid thumb instead of the striped default."""
        element = f"Player.{palette.name}.Scale.slider"
        try:
            image = tk.PhotoImage(master=self.root, width=12, height=14)
            image.put(palette.accent, to=(0, 0, 12, 14))
            self.style.element_create(element, "image", image, border=0, padding=0)
            self._thumb = image
        except tk.TclError:  # pragma: no cover - keep the default thumb
            return

        try:
            self.style.layout(
                "Player.Horizontal.TScale",
                [
                    (
                        "Horizontal.Scale.trough",
                        {
                            "sticky": "nswe",
                            "children": [(element, {"side": "left", "sticky": ""})],
                        },
                    ),
                    ("Horizontal.Scale.label", {"sticky": "nswe"}),
                ],
            )
        except tk.TclError:  # pragma: no cover - layout differs between Tk builds
            pass

    def _configure(self, style: str, **options) -> None:
        """Configure a style, ignoring options this Tk build does not know."""
        try:
            self.style.configure(style, **options)
        except tk.TclError:
            for key in list(options):
                try:
                    self.style.configure(style, **{key: options[key]})
                except tk.TclError:
                    options.pop(key, None)
            if options:
                self.style.configure(style, **options)

    def _map(self, style: str, **options) -> None:
        """Map a style, ignoring options this Tk build does not know."""
        try:
            self.style.map(style, **options)
        except tk.TclError:
            for key in list(options):
                try:
                    self.style.map(style, **{key: options[key]})
                except tk.TclError:
                    options.pop(key, None)
            if options:
                self.style.map(style, **options)

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def apply(self) -> None:
        """(Re)build every style for the current palette."""
        p = self.palette
        style = self.style

        try:
            style.theme_use("clam")
        except tk.TclError:  # pragma: no cover - clam ships with Tk
            pass

        self._configure(
            ".",
            background=p.bg,
            foreground=p.text,
            font=self.fonts["body"],
            borderwidth=0,
        )

        # Frames and labels -------------------------------------------------
        self._configure("TFrame", background=p.bg)
        self._configure("Surface.TFrame", background=p.surface)
        self._configure("TLabel", background=p.bg, foreground=p.text)
        self._configure("Surface.TLabel", background=p.surface, foreground=p.text)
        self._configure("Muted.TLabel", background=p.bg, foreground=p.text_muted)
        self._configure("SurfaceMuted.TLabel", background=p.surface, foreground=p.text_muted)
        self._configure("Faint.TLabel", background=p.bg, foreground=p.text_faint)
        self._configure("SurfaceFaint.TLabel", background=p.surface, foreground=p.text_faint)
        self._configure("Accent.TLabel", background=p.bg, foreground=p.accent)
        self._configure("SurfaceAccent.TLabel", background=p.surface, foreground=p.accent)

        # Buttons ------------------------------------------------------------
        self._configure(
            "TButton",
            background=p.surface_alt,
            foreground=p.text,
            borderwidth=0,
            padding=(12, 7),
            relief="flat",
            font=self.fonts["body"],
        )
        self._map(
            "TButton",
            background=[("pressed", p.border), ("active", p.hover)],
            foreground=[("disabled", p.text_faint)],
        )

        self._configure(
            "Ghost.TButton",
            background=p.bg,
            foreground=p.text_muted,
            borderwidth=1,
            relief="solid",
            padding=(11, 6),
            bordercolor=p.border_strong,
            font=self.fonts["body"],
        )
        self._map(
            "Ghost.TButton",
            background=[("pressed", p.surface_alt), ("active", p.hover)],
            foreground=[("active", p.text), ("disabled", p.text_faint)],
            bordercolor=[("active", p.accent)],
        )

        self._configure(
            "SurfaceGhost.TButton",
            background=p.surface,
            foreground=p.text_muted,
            borderwidth=1,
            relief="solid",
            padding=(11, 6),
            bordercolor=p.border_strong,
            font=self.fonts["body"],
        )
        self._map(
            "SurfaceGhost.TButton",
            background=[("pressed", p.surface_alt), ("active", p.hover)],
            foreground=[("active", p.text), ("disabled", p.text_faint)],
            bordercolor=[("active", p.accent)],
        )

        self._configure(
            "Accent.TButton",
            background=p.accent,
            foreground=p.on_accent,
            borderwidth=0,
            padding=(18, 10),
            relief="flat",
            font=self.fonts["body_bold"],
        )
        self._map(
            "Accent.TButton",
            background=[
                ("pressed", p.accent_pressed),
                ("active", p.accent_hover),
                ("disabled", p.accent_muted),
            ],
            foreground=[("disabled", p.on_accent)],
        )

        # The buttons of the mini player hug their text, so the scrubber
        # keeps as much room as possible.
        self._configure(
            "Player.TButton",
            background=p.surface,
            foreground=p.text_muted,
            borderwidth=1,
            relief="solid",
            padding=(11, 5),
            width=0,
            bordercolor=p.border_strong,
            font=self.fonts["body"],
        )
        self._map(
            "Player.TButton",
            background=[("pressed", p.surface_alt), ("active", p.hover)],
            foreground=[("active", p.text), ("disabled", p.text_faint)],
            bordercolor=[("active", p.accent)],
        )

        # Entries and combo boxes -------------------------------------------
        for name, background in (("TEntry", p.bg), ("Surface.TEntry", p.surface)):
            self._configure(
                name,
                fieldbackground=p.entry_bg,
                foreground=p.text,
                bordercolor=p.border,
                lightcolor=p.border,
                darkcolor=p.border,
                insertcolor=p.text,
                padding=(9, 7),
            )
            self._map(name, bordercolor=[("focus", p.accent), ("disabled", p.border)])

        for name, background in (("TCombobox", p.bg), ("Surface.TCombobox", p.surface)):
            self._configure(
                name,
                fieldbackground=p.entry_bg,
                background=p.surface_alt,
                foreground=p.text,
                arrowcolor=p.text_muted,
                bordercolor=p.border,
                lightcolor=p.border,
                darkcolor=p.border,
                insertcolor=p.text,
                arrowsize=14,
                padding=(9, 6),
            )
            self._map(
                name,
                fieldbackground=[("readonly", p.entry_bg), ("focus", p.entry_bg)],
                bordercolor=[("focus", p.accent)],
                arrowcolor=[("disabled", p.text_faint)],
            )

        for name in ("TSpinbox", "Surface.TSpinbox"):
            self._configure(
                name,
                fieldbackground=p.entry_bg,
                background=p.surface_alt,
                foreground=p.text,
                arrowcolor=p.text_muted,
                bordercolor=p.border,
                lightcolor=p.border,
                darkcolor=p.border,
                arrowsize=14,
                padding=(6, 5),
            )
            self._map(name, bordercolor=[("focus", p.accent)])

        # Check buttons ------------------------------------------------------
        for name, background in (("TCheckbutton", p.bg), ("Surface.TCheckbutton", p.surface)):
            self._configure(
                name,
                background=background,
                foreground=p.text,
                indicatorbackground=p.entry_bg,
                indicatorforeground=p.on_accent,
                bordercolor=p.border,
                padding=(2, 2),
            )
            self._map(
                name,
                indicatorbackground=[("selected", p.accent), ("active", p.hover)],
                foreground=[("selected", p.on_accent), ("disabled", p.text_faint)],
            )

        # Progress and scroll bars -------------------------------------------
        self._configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=p.track,
            background=p.accent,
            bordercolor=p.track,
            lightcolor=p.accent,
            darkcolor=p.accent,
            thickness=8,
        )

        # The mini player scrubber -------------------------------------------
        self._configure(
            "Player.Horizontal.TScale",
            background=p.accent,
            foreground=p.accent,
            troughcolor=p.track,
            bordercolor=p.track,
            lightcolor=p.accent,
            darkcolor=p.accent,
            sliderthickness=14,
        )
        self._map(
            "Player.Horizontal.TScale",
            background=[("active", p.accent_hover), ("pressed", p.accent_pressed)],
        )
        self._build_scale_thumb(p)

        scrollbar_styles = (
            ("Vertical.TScrollbar", p.bg),
            ("Surface.Vertical.TScrollbar", p.surface),
        )
        for name, trough in scrollbar_styles:
            self._configure(
                name,
                background=p.scrollbar,
                troughcolor=trough,
                bordercolor=trough,
                lightcolor=trough,
                darkcolor=trough,
                arrowcolor=trough,
                arrowsize=11,
                gripcount=0,
            )
            self._map(
                name,
                background=[("active", p.scrollbar_hover), ("pressed", p.scrollbar_hover)],
            )
            # Drop the up/down arrow buttons so only a slim thumb is left.
            try:
                self.style.layout(
                    name,
                    [
                        (
                            "Vertical.Scrollbar.trough",
                            {
                                "sticky": "ns",
                                "children": [
                                    (
                                        "Vertical.Scrollbar.thumb",
                                        {"expand": "1", "sticky": "nswe"},
                                    )
                                ],
                            },
                        )
                    ],
                )
            except tk.TclError:  # pragma: no cover - layout differs between Tk builds
                pass

        # Combo box drop-down lists use plain list boxes; colour them too.
        root = self.root
        root.option_add("*TCombobox*Listbox.background", p.surface)
        root.option_add("*TCombobox*Listbox.foreground", p.text)
        root.option_add("*TCombobox*Listbox.selectBackground", p.selection)
        root.option_add("*TCombobox*Listbox.selectForeground", p.selection_text)
        root.option_add("*TCombobox*Listbox.font", self.fonts["body"])


class Themed:
    """Mixin for widgets that have to be recoloured when the theme changes."""

    def apply_theme(self, theme: Theme) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def refresh(self, theme: Theme) -> None:
        """Apply the theme to this widget and, recursively, to its children."""
        self.apply_theme(theme)
        for child in self.winfo_children():  # type: ignore[attr-defined]
            if isinstance(child, Themed):
                child.refresh(theme)


class Surface(tk.Frame, Themed):
    """A plain frame that paints itself with one of the theme surfaces."""

    def __init__(self, master: tk.Misc, theme: Theme, role: str = "bg", **kwargs) -> None:
        background = theme.palette.surface if role == "surface" else theme.palette.bg
        super().__init__(master, bg=background, **kwargs)
        self.role = role

    def apply_theme(self, theme: Theme) -> None:
        palette = theme.palette
        self.configure(bg=palette.surface if self.role == "surface" else palette.bg)


class Label(tk.Label, Themed):
    """A label that follows the theme, including its foreground colour."""

    ROLES = ("text", "muted", "faint", "accent", "danger", "success")

    def __init__(
        self,
        master: tk.Misc,
        theme: Theme,
        text: str = "",
        role: str = "text",
        size: str = "body",
        **kwargs,
    ) -> None:
        if "background" not in kwargs and "bg" not in kwargs:
            try:
                kwargs["background"] = master.cget("background")
            except tk.TclError:
                kwargs["background"] = theme.palette.bg
        kwargs.setdefault("anchor", "w")
        super().__init__(master, text=text, font=theme.fonts[size], **kwargs)
        self.role = role
        self.size = size
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        palette = theme.palette
        colors = {
            "text": palette.text,
            "muted": palette.text_muted,
            "faint": palette.text_faint,
            "accent": palette.accent,
            "danger": palette.danger,
            "success": palette.success,
        }
        try:
            background = self.master.cget("background")
        except tk.TclError:
            background = palette.bg
        self.configure(
            background=background,
            foreground=colors.get(self.role, palette.text),
            font=theme.fonts[self.size],
        )


class Card(Surface):
    """A titled panel with a title row and a content area."""

    def __init__(
        self,
        master: tk.Misc,
        theme: Theme,
        title: str = "",
        subtitle: str = "",
        padding: int = 14,
    ) -> None:
        super().__init__(master, theme, role="surface", bd=0, highlightthickness=1)
        self.padding = padding
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.header = Surface(self, theme, role="surface")
        self.header.grid(row=0, column=0, sticky="ew", padx=padding, pady=(padding, 0))
        self.header.columnconfigure(0, weight=1)

        self.title_label = Label(self.header, theme, title, role="text", size="section")
        self.title_label.grid(row=0, column=0, sticky="w")
        if subtitle:
            self.subtitle_label = Label(self.header, theme, subtitle, role="faint", size="small")
            self.subtitle_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        else:
            self.subtitle_label = None

        self.actions = Surface(self.header, theme, role="surface")
        self.actions.grid(row=0, column=1, sticky="e")

        self.body = Surface(self, theme, role="surface")
        self.body.grid(row=1, column=0, sticky="nsew", padx=padding, pady=(10, padding))
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)

        # Paint the border right away. Without this the frame keeps the
        # system default highlight colour (white on Windows) until the
        # theme is toggled for the first time.
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        palette = theme.palette
        super().apply_theme(theme)
        self.configure(
            highlightbackground=palette.border,
            highlightcolor=palette.border,
        )


class FileList(tk.Listbox, Themed):
    """The list of files to transcribe, styled as a drop target."""

    def __init__(self, master: tk.Misc, theme: Theme, **kwargs) -> None:
        super().__init__(
            master,
            height=4,
            selectmode=tk.EXTENDED,
            exportselection=False,
            activestyle="none",
            borderwidth=0,
            relief="flat",
            highlightthickness=1,
            **kwargs,
        )
        self.theme = theme
        self.hint = tk.Label(
            self,
            text="Drop audio or video files here",
            font=theme.fonts["small"],
            bg=theme.palette.entry_bg,
            fg=theme.palette.text_faint,
        )
        self.hint.place(relx=0.5, rely=0.5, anchor="center")
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        palette = theme.palette
        self.configure(
            background=palette.entry_bg,
            foreground=palette.text,
            selectbackground=palette.selection,
            selectforeground=palette.selection_text,
            highlightbackground=palette.border,
            highlightcolor=palette.accent,
            font=theme.fonts["body"],
        )
        self.hint.configure(
            background=palette.entry_bg,
            foreground=palette.text_faint,
            font=theme.fonts["small"],
        )

    def set_empty(self, empty: bool) -> None:
        """Show or hide the placeholder shown while the list is empty."""
        if empty:
            self.hint.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.hint.place_forget()


class Transcript(tk.Text, Themed):
    """The scrollable log that doubles as the live transcript."""

    TAGS = {
        "error": "danger",
        "segment": "text",
        "time": "faint",
        "file": "muted",
        "muted": "faint",
        "ok": "success",
        "accent": "accent",
    }

    def __init__(self, master: tk.Misc, theme: Theme, **kwargs) -> None:
        super().__init__(
            master,
            width=1,
            height=8,
            wrap="word",
            state="disabled",
            borderwidth=0,
            relief="flat",
            highlightthickness=0,
            padx=14,
            pady=12,
            spacing1=2,
            spacing3=2,
            cursor="arrow",
            **kwargs,
        )
        self._highlight: Optional[Tuple[str, str]] = None
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        palette = theme.palette
        self.configure(
            background=palette.log_bg,
            foreground=palette.text,
            insertbackground=palette.text,
            selectbackground=palette.selection,
            selectforeground=palette.selection_text,
            font=theme.fonts["mono"],
        )
        colors = {
            "danger": palette.danger,
            "text": palette.text,
            "faint": palette.text_faint,
            "muted": palette.text_muted,
            "success": palette.success,
            "accent": palette.accent,
        }
        for tag, role in self.TAGS.items():
            self.tag_configure(tag, foreground=colors.get(role, palette.text))

        # The line that is being spoken right now.
        self.tag_configure(
            "highlight",
            background=palette.highlight,
            foreground=palette.text,
            font=theme.fonts["mono_bold"],
        )
        try:
            self.tag_raise("highlight")
        except tk.TclError:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------ #
    # Highlighting
    # ------------------------------------------------------------------ #

    def set_highlight(self, start_mark: Optional[str], end_mark: Optional[str]) -> None:
        """Move the highlight to another range of the transcript.

        The ranges are given as text marks, so they survive the log being
        trimmed while a long recording is transcribed.
        """
        if self._highlight is not None:
            try:
                self.tag_remove("highlight", self._highlight[0], self._highlight[1])
            except tk.TclError:  # pragma: no cover - the marks are gone
                pass
            self._highlight = None

        if start_mark is None or end_mark is None:
            return

        try:
            self.tag_add("highlight", start_mark, end_mark)
        except tk.TclError:  # pragma: no cover - the marks are gone
            return
        self._highlight = (start_mark, end_mark)


def enable_dpi_awareness() -> None:
    """Make the window crisp on high DPI displays (Windows)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # pragma: no cover - best effort only
        pass
