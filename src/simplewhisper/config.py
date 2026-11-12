"""Shared constants and small helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

APP_NAME = "simplewhisper"
DISPLAY_NAME = "SimpleWhisper"
DEFAULT_MODEL = "small"
DEFAULT_FORMAT = "srt"

OUTPUT_FORMATS = ("srt", "vtt", "txt", "json", "tsv")

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".aac",
    ".wma",
    ".amr",
    ".aiff",
    ".aif",
    ".caf",
    ".webm",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
    ".webm",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".3gp",
    ".ogv",
}

MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Whisper's language codes. The interactive prompt accepts either the code
# ("en") or the English name ("english"). Empty input means auto-detect.
LANGUAGES: Dict[str, str] = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

_NAME_TO_CODE = {name: code for code, name in LANGUAGES.items()}


def default_model_dir() -> Path:
    """Return the directory where Whisper model files are cached."""
    override = os.environ.get("SIMPLEWHISPER_HOME")
    if override:
        return Path(override).expanduser()

    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / APP_NAME / "models"

    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / APP_NAME / "models"

    return Path.home() / ".cache" / APP_NAME / "models"


def default_cuda_dir() -> Path:
    """Return the directory where the CUDA libraries are cached."""
    override = os.environ.get("SIMPLEWHISPER_CUDA_HOME")
    if override:
        return Path(override).expanduser()

    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / APP_NAME / "cuda"

    return default_model_dir().parent / "cuda"


def resolve_language(value: Optional[str]) -> Optional[str]:
    """Turn user input into a Whisper language code.

    Empty input, ``auto`` and ``detect`` mean auto-detection (``None``).
    Accepts codes ("en"), English names ("english") and regional codes
    ("pt-BR" -> "pt").
    """
    if value is None:
        return None

    cleaned = value.strip().lower().replace("_", "-")
    if not cleaned or cleaned in {"auto", "detect", "automatic"}:
        return None

    if cleaned in LANGUAGES:
        return cleaned

    if cleaned in _NAME_TO_CODE:
        return _NAME_TO_CODE[cleaned]

    base = cleaned.split("-", 1)[0]
    if base in LANGUAGES:
        return base

    raise ValueError(
        f"unknown language '{value.strip()}'. Use a code such as 'en', 'es', 'zh', "
        "or a name such as 'english'; leave empty for auto-detect."
    )


def language_label(code: Optional[str]) -> str:
    """Human-readable label for a language code."""
    if not code:
        return "auto-detect"
    name = LANGUAGES.get(code)
    return f"{name} ({code})" if name else code


def format_bytes(count: float) -> str:
    """Format a byte count for display, e.g. ``484.0 MB``."""
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def format_clock(seconds: float) -> str:
    """Format a timestamp as ``mm:ss`` (or ``h:mm:ss`` past an hour)."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
