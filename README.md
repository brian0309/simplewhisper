# SimpleWhisper

**Local, offline speech-to-text in your terminal.** Drag in an audio or video file, choose a model, and get subtitles. No API keys, no uploads, no cloud.

SimpleWhisper is a small Python CLI built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a fast CTranslate2 implementation of OpenAI's Whisper. Model weights are **not bundled** with this project. They are downloaded on first use and cached on your machine, so every run after that works offline.

```
$ simplewhisper

SimpleWhisper - local speech-to-text
------------------------------------------
Drag & drop audio/video file(s) here, then press Enter:
> "C:\Users\you\Videos\lecture.mp4"

Choose a model:
  1) tiny                39 M params    ~75 MB   ~1 GB RAM  fastest; rough drafts and smoke tests
  2) base                74 M params   ~145 MB   ~1 GB RAM  fast drafts, light on CPU
  3) small              244 M params   ~484 MB   ~2 GB RAM  best speed/quality balance on CPU  (default)
  ...
Choice [1-7] (3):
Language [auto]:
Choice [1-6] (1):
```

## Features

- **Fully local** - audio never leaves your computer. The only network traffic is downloading models.
- **Interactive wizard** - drag & drop files straight into the terminal, then pick options from menus.
- **Desktop GUI** - an optional Tkinter window for point-and-click transcription, with drag & drop.
- **Scriptable** - every option is also a flag, so it works in batch files and pipelines.
- **Audio and video** - MP4, MKV, MOV, MP3, WAV, M4A, FLAC, and more. Video soundtracks are extracted automatically.
- **Five output formats** - SRT, WebVTT, plain text, JSON, and TSV, or all at once.
- **Speaker breakdown** - label every line with who said it (Speaker 1, Speaker 2, ...) on request.
- **100 languages** - auto-detect, force a language, or translate speech to English.
- **Batch friendly** - pass multiple files or a wildcard; each one gets its own transcript.
- **Download manager** - `simplewhisper models` shows what is cached and pre-downloads models for offline use.

## Requirements

- **Python 3.9 - 3.12** (3.11 or 3.12 recommended)
- **Disk space** for the model you choose: ~150 MB for `tiny`/`base`, ~500 MB for `small`, up to ~3 GB for `large-v3`
- **ffmpeg** (optional, recommended) for the widest media-format support
- **NVIDIA GPU** (optional) for much faster transcription of large models

The desktop GUI uses Tkinter, which is included with the official Windows and macOS Python installers. On Linux, install it with `sudo apt install python3-tk` if it is missing.

Without ffmpeg, the bundled PyAV decoder still handles most common audio and video files.

## Installation

### Windows (PowerShell)

```powershell
git clone <your-repo-url> simplewhisper
cd simplewhisper

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

If PowerShell refuses to run the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS / Linux

```bash
git clone <your-repo-url> simplewhisper
cd simplewhisper

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

### Optional: install ffmpeg

| Platform | Command |
| --- | --- |
| Windows | `winget install --id Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |

Confirm it is visible:

```bash
ffmpeg -version
```

### Verify the install

```bash
simplewhisper --version
simplewhisper --help
```

`python -m simplewhisper` works too, if the console script is not on your `PATH`.

## Quick start

Run the wizard with no arguments:

```bash
simplewhisper
```

1. **Drop files in.** Drag one or more files from Explorer/Finder into the terminal, or paste paths separated by spaces. Quoted paths are handled, so paths with spaces are fine.
2. **Pick a model.** `small` is the default and the best starting point on CPU.
3. **Pick a language.** Press Enter to auto-detect, or type a code such as `en` or a name such as `spanish`.
4. **Pick a format.** SRT, VTT, TXT, JSON, TSV, or all of them.
5. **Choose an output folder.** Leave empty to write next to each input file.
6. **Confirm.** Press Enter.

The first run downloads the selected model (a few hundred MB for `small`) into your user cache. Later runs use the cached copy. The download shows a progress bar with the percentage and the number of megabytes downloaded.

## Non-interactive usage

If a file is passed on the command line, only the options you left out are asked interactively. Add `-y` to skip all prompts and use defaults (`small`, auto-detect, SRT, output next to the input).

```bash
# One file, fully specified
simplewhisper "C:\Videos\lecture.mp4" -m small -l en -f srt

# Auto-detect, write every format into ./transcripts
simplewhisper interview.mkv -m large-v3-turbo -l auto -f all -o transcripts

# Plain text only
simplewhisper podcast.mp3 -f txt

# Translate speech to English subtitles
simplewhisper talk.mp4 -t translate -f srt

# Multiple files in one go
simplewhisper intro.mp4 chapter1.mp4 chapter2.mp4 -m base -f srt -o out

# Accept defaults, no prompts
simplewhisper -y voice-note.m4a
```

More examples:

| Goal | Command |
| --- | --- |
| Fast rough draft | `simplewhisper clip.mp4 -m tiny -f txt` |
| Good CPU balance | `simplewhisper lecture.mp4 -m small -l en` |
| Best accuracy on GPU | `simplewhisper interview.mp4 -m large-v3 -d cuda -c float16` |
| Fast large-class model | `simplewhisper meeting.mkv -m large-v3-turbo` |
| Burn-ready subtitles | `simplewhisper film.mkv -m medium -f srt -o subs` |
| Data for your own tool | `simplewhisper call.mp3 -f json` |
| English translation | `simplewhisper foreign.mp4 -t translate -f vtt` |
| Who said what | `simplewhisper panel.mp4 -S -f txt` |

## Speaker breakdown

Interviews, panels and podcasts are hard to read as one flat transcript.
With `-S` (or **Break down by speaker** in the GUI) every line is labelled
with the voice that said it:

```
Speaker 1: So, tell us how the project got started.
Speaker 2: It began with a single customer asking for a report.
Speaker 1: And how did that grow?
```

The speaker names are numbered by order of first appearance, so the first
voice you hear is always `Speaker 1`. JSON output carries the same
information in a `speaker` field per segment.

If you already know how many people are talking, say so and the labelling is
forced to that number:

```bash
simplewhisper interview.mp4 -S --num-speakers 2
```

Otherwise the number of speakers is detected automatically. Diarization runs
on the same machine as the transcription, using ONNX Runtime - there is no
PyTorch, no Hugging Face account and no API key. The two speaker model files
(~33 MB) are downloaded once into the model cache and then work offline.

Diarization is a separate pass over the audio and adds roughly a few percent
to the processing time on top of transcription.

### How the speaker detection is run

sherpa-onnx does the whole speaker detection inside one call, and that call
holds the Python GIL from start to finish. Running it in a thread would
therefore freeze every other thread in the program, which is why the desktop
window used to stop responding while the speakers were being detected.

SimpleWhisper therefore runs the speaker detection in a **separate process**. The
window stays responsive, keeps showing the transcript as it arrives, and
reports how far the speaker detection has got:

```
[1/1] interview.mp4
Transcribed interview.mp4 in 42.1s.
Detecting speakers: 63%
Speaker detection finished in 8.4s (57 turns, 3 speaker(s)).
```

The same progress is shown on the command line:

```
  detecting speakers:  63.3%
  speakers: 57 turn(s) in 8.4s
```

A long job can be stopped with the **Cancel** button in the GUI, which also
kills the speaker process.

## Desktop GUI (Tkinter)

Prefer windows and buttons? SimpleWhisper includes a local Tkinter app:

```bash
simplewhisper gui
```

On Windows you can also double-click **`SimpleWhisper-GUI.cmd`**, which opens the window without a console.

The GUI supports:

- Add files with a picker, or drag & drop them onto the file list
- A **Model files…** button next to the model dropdown, which opens the model
  cache folder so the downloaded weights can be inspected or deleted
- Model, language, output format, and output folder selectors
- **Break down by speaker** toggle, with an optional speaker count
- Overwrite toggle
- Live progress bar, including the model download status on the first run
- Live transcript lines in the log while each file is processed
- Timing lines in the log for the transcription and the speaker detection
- A **mini player** under the transcript, so a recording can be reviewed
  (play, pause, scrub, and a speed from `0.5x` to `4x`)
- A **Cancel** button that stops a run that is taking too long
- **Copy** and **Clear** buttons for the transcript
- An **Open output folder** button
- A light and a dark theme: the window follows the operating system colour
  scheme, and the button in the title bar switches between the two by hand

Keyboard shortcuts: `Ctrl+O` adds files, `Ctrl+Shift+M` opens the model files
folder, `Ctrl+Enter` starts a run, and `Delete` removes the selected files.

### Reviewing a transcript

As soon as a file has a transcript, a small player appears under the log. It
plays the audio of that file while the transcript follows along: the line that
is being spoken is highlighted, and the log scrolls to it.

- **Play** and **Stop** run the audio, the scrubber moves through it
- A **speed** box plays the recording slower or faster, from `0.5x` to `4x`
- Click any line of the transcript to jump straight to that moment
- The space bar plays and pauses, as long as the log has the focus
- When several files were transcribed, the box in front of the player picks
  the file to review

The player uses `ffplay` from ffmpeg when ffmpeg is installed, so every
container ffmpeg understands can be played. Without ffmpeg the audio is
decoded with the built-in decoder and handed to the system player.

Drag & drop needs the optional extra:

```bash
pip install -e ".[gui]"
```

Without it, everything else works; use **Add files...** instead.

The GUI uses the same local model cache, ffmpeg extraction, and writers as the CLI. Nothing is uploaded, and no extra server or browser is involved.

## Options

| Flag | Description |
| --- | --- |
| `FILE ...` | One or more audio/video files. Omit to use the wizard. |
| `-m, --model NAME` | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, `distil-large-v3` |
| `-l, --language LANG` | Code (`en`, `es`, `ja`) or name (`english`). Empty or `auto` = detect. |
| `-f, --format FMT` | `srt`, `vtt`, `txt`, `json`, `tsv`, `all`; comma-separate for several (`srt,txt`) |
| `-o, --output-dir DIR` | Write transcripts here instead of next to each input |
| `-d, --device auto\|cpu\|cuda` | Compute device (default: `auto`) |
| `-c, --compute-type TYPE` | `default`, `int8`, `int8_float16`, `float16`, `float32` |
| `-t, --task transcribe\|translate` | `translate` writes English from any detected language |
| `--model-dir DIR` | Model cache location |
| `--beam-size N` | Decoding beam size (default: `5`) |
| `-S, --speakers` | Break the transcript down by speaker |
| `--num-speakers N` | How many speakers to expect; `0` lets the model decide (default: `0`) |
| `--speaker-threshold T` | Speaker clustering threshold; lower splits into more speakers (default: `0.7`) |
| `--no-vad` | Disable voice activity detection |
| `--no-ffmpeg` | Use the built-in decoder instead of calling ffmpeg |
| `--overwrite` | Replace existing transcript files |
| `-y, --yes` | Never prompt; use flags and defaults |
| `-v, --verbose` | Verbose logging |
| `--version` | Print the version |

Run `simplewhisper --help` for the same list in your terminal.

Three subcommands sit outside the flags: `simplewhisper gui` opens the desktop window, `simplewhisper cuda` shows what the GPU support found, and `simplewhisper models` manages the model cache (see below).

## Models

No model is included in the repository. The first run downloads the model you choose and caches it. Sizes and memory use are approximate and vary with quantization and device.

| Model | Params | Download | RAM | Notes |
| --- | ---: | ---: | ---: | --- |
| `tiny` | 39 M | ~75 MB | ~1 GB | Fastest. Rough drafts, smoke tests. |
| `base` | 74 M | ~145 MB | ~1 GB | Fast drafts; light on CPU. |
| `small` | 244 M | ~484 MB | ~2 GB | Best speed/quality balance on CPU. **Default.** |
| `medium` | 769 M | ~1.5 GB | ~4 GB | More accurate; slow on CPU. |
| `large-v3` | 1.55 B | ~3.1 GB | ~6 GB | Most accurate; GPU recommended. |
| `large-v3-turbo` | 809 M | ~1.6 GB | ~4 GB | Near `large-v3` quality, much faster. GPU recommended. |
| `distil-large-v3` | 756 M | ~1.5 GB | ~4 GB | English-only; large-class accuracy for less compute. |

### Model cache

| Platform | Default location |
| --- | --- |
| Windows | `%LOCALAPPDATA%\simplewhisper\models` |
| macOS / Linux | `~/.cache/simplewhisper/models` (or `$XDG_CACHE_HOME/simplewhisper/models`) |

Override it with `--model-dir DIR` or by setting `SIMPLEWHISPER_HOME` to a folder of your choice.

### Manage models

```bash
# List models, sizes, and which ones are already cached (marked with *)
simplewhisper models

# Download now so the first transcription works offline
simplewhisper models download small

# Download several at once, into a custom folder
simplewhisper models download small large-v3-turbo --model-dir D:\ai\models
```

Once a model is cached, you can stay offline. To force offline mode for Hugging Face downloads:

```powershell
$env:HF_HUB_OFFLINE = "1"   # PowerShell
```

```bash
export HF_HUB_OFFLINE=1     # macOS / Linux
```

If `simplewhisper models` does not mark a model as cached even though you downloaded it, check that you are using the same `--model-dir` / `SIMPLEWHISPER_HOME`.

## Languages

- **Auto-detect:** press Enter at the language prompt, or pass `-l auto` (an empty `-l ""` also works).
- **Force a language:** use the two-letter code (`en`, `es`, `pt`) or the English name (`english`, `spanish`, `portuguese`).
- Regional codes fall back to the base language: `pt-BR` becomes `pt`.
- If you already know the language, set it. It skips detection and is often faster and more accurate on short or noisy clips.
- `-t translate` translates any language to **English** (Whisper's translation mode).

## Output formats

| Format | File | Description |
| --- | --- | --- |
| `srt` | `.srt` | SubRip subtitles. Works in VLC, MPC, most editors. **Default.** |
| `vtt` | `.vtt` | WebVTT subtitles for browsers and web players. |
| `txt` | `.txt` | Plain transcript, one segment per line. |
| `json` | `.json` | Segments plus source, model, language, probability, and duration. |
| `tsv` | `.tsv` | `start<TAB>end<TAB>text`, timestamps with dots. |
| `all` | all of the above | Writes every format in one run. |

Combine formats with commas: `-f srt,txt`.

### SRT

```srt
1
00:00:00,000 --> 00:00:02,480
Welcome back to the show.

2
00:00:02,480 --> 00:00:05,120
Today we are talking about local speech recognition.
```

### JSON

```json
{
  "source": "C:\\Videos\\lecture.mp4",
  "model": "small",
  "task": "transcribe",
  "language": "en",
  "language_probability": 0.9971,
  "duration": 5.12,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.48,
      "text": "Welcome back to the show."
    }
  ]
}
```

Output naming: the input `lecture.mp4` produces `lecture.srt` (and `lecture.vtt`, etc.). Files are UTF-8 with LF newlines.

## Video and audio extraction

You do not need to extract audio yourself.

1. If **ffmpeg** is installed, SimpleWhisper extracts a temporary mono 16 kHz WAV from the input, transcribes it, and deletes it automatically.
2. If ffmpeg is missing, the original file is passed to faster-whisper's built-in PyAV decoder, which can read audio from most containers directly.
3. If ffmpeg fails on a file, SimpleWhisper warns you and falls back to the built-in decoder.

Common supported containers: `.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`, `.webm`, `.flv`, `.wmv`, `.mpg`, `.mpeg`, `.ts`, `.3gp`, plus audio formats such as `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac`, and `.wma`.

If a file has no audio track, or the codec is unusual, install ffmpeg and try again. Use `--no-ffmpeg` to bypass extraction entirely.

## Drag & drop tips

- **Windows Terminal / PowerShell / cmd:** drag a file from Explorer into the window. The path is inserted automatically and quoted if it contains spaces.
- **macOS Terminal:** drag from Finder; Terminal inserts an escaped path.
- **Linux terminals:** most support drag & drop the same way.
- **Multiple files:** drop them all at once. The wizard accepts several quoted paths separated by spaces.
- **Wildcards:** the wizard also expands patterns such as `C:\audio\*.mp3` and `./podcasts/*.m4a`.
- **Typing manually:** wrap paths containing spaces in double quotes.

## Where transcripts go

By default, each transcript is written **next to its input file**:

```
C:\Videos\lecture.mp4  ->  C:\Videos\lecture.srt
```

Use `-o` to collect everything in one folder:

```bash
simplewhisper a.mp4 b.mp4 -o transcripts
```

Existing files are skipped unless you pass `--overwrite`. If two inputs share the same filename and you write them to one output folder, they will collide - use `--overwrite` deliberately or transcribe them separately.

## GPU and performance

- `-d auto` (the default) uses an NVIDIA GPU if CTranslate2 can load one; otherwise it runs on CPU.
- On CPU, `int8` is used by default (fast, low memory). On GPU, `float16` is used by default.
- For CUDA, install a recent NVIDIA driver plus **CUDA 12.x** and **cuDNN 9** compatible with your CTranslate2 version. See the [faster-whisper README](https://github.com/SYSTRAN/faster-whisper#gpu) for the current matrix.
- If CUDA fails with a missing `cublas64_12.dll` or `cudnn_*.dll`, fall back with `-d cpu` while you fix the driver setup.
- Rough guidance: `tiny`/`base` are comfortable on CPU, `small` is the CPU sweet spot, `medium` and `large-*` are much better on a GPU. For CPU-only large-model work, try `large-v3-turbo` first.
- `--beam-size 1` is faster and less accurate; `--beam-size 5` is the default. `--no-vad` disables silence skipping and can help if quiet speech is being cut off, but usually increases runtime.

## Building a standalone executable

`build.bat` turns the project into a self-contained Windows program with
[PyInstaller](https://pyinstaller.org), then wraps the result in an installer
built with [Inno Setup](https://jrsoftware.org/isinfo.php). No Python
installation is needed on the machine that runs the build or the program.

```bat
build.bat
```

The script creates the virtual environment if it is missing, installs the
dependencies and writes two executables into `dist\SimpleWhisper\`:

| File | What it does |
| --- | --- |
| `SimpleWhisper.exe` | The command line interface. |
| `SimpleWhisper-GUI.exe` | The desktop window, without a console window. |

Both take the same arguments as the `simplewhisper` command, so
`SimpleWhisper.exe lecture.mp4 -m small -f srt` works exactly like the
installed version.

The installer lands in `build\installer\simplewhisper-setup.exe`. It installs both
executables and the shared `_internal` folder into `Program Files\SimpleWhisper`,
adds Start menu shortcuts, offers a desktop icon, and registers an uninstall
entry. The MIT license is copied next to the program.

Options:

| Option | Effect |
| --- | --- |
| `--onefile` | Two single file executables instead of a folder. Bigger files, slower start, no separate `_internal` folder. |
| `--clean` | Throw away the PyInstaller cache and build everything from scratch. |
| `--no-install` | Skip the dependency install and use the virtual environment as it is. |
| `--no-pause` | Do not wait for a key press when the build ends. |

Environment variables:

| Variable | Values | Effect |
| --- | --- | --- |
| `BUILD_MODE` | `standalone` (default), `onefile` | Which PyInstaller mode to build. Only the folder build gets an installer. |
| `BUILD_INSTALLER` | `1` (default), `0` | Whether to run Inno Setup after PyInstaller. |
| `AUTO_INSTALL_INNO` | `1` (default), `0` | Whether to install Inno Setup with winget or Chocolatey when `ISCC.exe` is not found. |
| `SIGN_MODE` | `self`, `none` | `none` on CI, `self` on a desktop: sign the executables and the installer with a local self-signed certificate. |

Notes:

- The models are **not** part of the build. They are downloaded on the first
  run into `%LOCALAPPDATA%\simplewhisper\models`, exactly as with the installed
  version.
- ffmpeg is optional. Without it the bundled PyAV decoder is used.
- The build carries no distribution signature, so Windows SmartScreen may ask
  before the first run. Locally, `SIGN_MODE=self` signs the executables and the
  installer with a self-signed certificate; that only silences warnings on the
  machine that trusts the certificate.
- CUDA is **not** bundled, and the build works fine without it: `-d auto` (the
  default) picks the CPU when no GPU is present, and the CPU path (int8 by
  default) needs no NVIDIA libraries at all.
- To use a GPU, the machine needs a **NVIDIA driver** (525.60.13 or newer).
  The CUDA libraries themselves (`cublas64_12.dll` / `cublasLt64_12.dll`,
  548 MB) are **not** shipped with the build. On the first GPU run
  SimpleWhisper finds them where they are usually installed (the CUDA toolkit,
  PyTorch's `torch\lib`, or NVIDIA's pip packages). If they are nowhere to be
  found, it asks before downloading them once from NVIDIA into
  `%LOCALAPPDATA%\simplewhisper\cuda`; say no and it uses the CPU. The download
  is checked against a SHA-256, and the files are only ever taken from
  NVIDIA's redistributable archive.
- `simplewhisper cuda` shows what the GPU support found, and
  `simplewhisper cuda --download` fetches the libraries up front.
- With `-d cuda` on a machine without a usable GPU, the run fails with a
  clear error instead of crashing.
- The build is reproducible from the checked-in files: `SimpleWhisper.spec`
  holds the PyInstaller settings, `scripts\launcher.py` the entry point, and
  `scripts\hook_dll_path.py` the runtime hook that points Windows at the
  bundled native libraries.
- The icon is `assets\simplewhisper.ico`. Edit `assets\simplewhisper.svg`, then run
  `python scripts\make_icon.py` (needs `pillow` and `resvg-py`) to regenerate
  it. The script also refreshes the copy in `src\simplewhisper\assets\`, which
  is the one the window shows while it runs.

## Automated builds

`.github\workflows\manual-build-publish.yml` runs the same `build.bat` on a
`windows-latest` runner, uploads the installer as a workflow artifact, and can
publish it as a GitHub release.

1. Open **Actions**, pick **Manual Build and Publish**, then **Run workflow**.
2. Enter the version. It has to match `version` in `pyproject.toml`, or the run
   stops before building.
3. Tick **Publish GitHub Release** to create tag `vX.Y.Z` with the installer
   attached; leave it off to only keep the artifact.

The workflow installs Inno Setup with Chocolatey, runs the build with
`SIGN_MODE=none`, and expects `build\installer\simplewhisper-setup.exe`.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Warning that ffmpeg could not extract audio | Install ffmpeg, then run again. Most files still work through PyAV without it. |
| Model download stalls or fails | Run `simplewhisper models download small` to see the error directly. Check proxy/firewall rules, retry, or pre-download on another machine and copy the cache folder. |
| `CUDA driver version is insufficient for CUDA runtime version` | The NVIDIA driver is missing or older than 525.60.13. Update it, or use `-d cpu` (or let the CPU fallback happen). Run `simplewhisper cuda` to see what is present. |
| `could not load model ... cublas64_12.dll` | The CUDA libraries are missing. Run `simplewhisper cuda --download` (548 MB), or install the CUDA toolkit. Without them, use `-d cpu`. |
| `Could not load library cublas64_12.dll` / `cudnn_*.dll` | CUDA/cuDNN mismatch. Install the versions required by faster-whisper, update the GPU driver, or use `-d cpu`. |
| Very slow on CPU | Use `tiny`, `base`, or `small`; keep `-c int8`; reduce `--beam-size`; close other CPU-heavy apps. |
| Garbled or missing characters in the terminal | Use Windows Terminal and UTF-8: `chcp 65001`, or set `$env:PYTHONUTF8=1` in PowerShell. Transcript files are always UTF-8. |
| "No speech detected" | Check that the file actually has an audible audio track. If it does, try `--no-vad`. |
| The window stops responding | The speaker detection now runs in its own process, so it should not happen. If it does, the log pane shows which step was running; press **Cancel** to stop the job. |
| Existing transcript is skipped | Pass `--overwrite`. |
| `unknown language` | Use a two-letter code such as `en`, `es`, `zh`, or the English language name. Empty means auto-detect. |
| Python 3.13 install fails building a dependency | Create the virtualenv with Python 3.11 or 3.12. |
| `simplewhisper` command not found | Activate the virtualenv, reinstall with `pip install -e .`, or run `python -m simplewhisper`. |

## Project layout

```
simplewhisper/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ build.bat            # build the standalone executables + installer
├─ SimpleWhisper.cmd        # double-click: terminal wizard
├─ SimpleWhisper-GUI.cmd    # double-click: desktop window
├─ SimpleWhisper.spec       # PyInstaller settings
├─ installer/
│  └─ simplewhisper.iss          # Inno Setup script
├─ assets/
│  ├─ simplewhisper.svg          # icon source
│  ├─ simplewhisper.png          # rendered preview
│  └─ simplewhisper.ico          # icon used by the executables and the installer
├─ .github/
│  └─ workflows/
│     └─ manual-build-publish.yml  # CI: build, artifact, optional release
├─ scripts/
│  ├─ launcher.py           # entry point of the frozen builds
│  ├─ hook_dll_path.py      # runtime hook for the bundled native libraries
│  ├─ make_icon.py          # renders the SVG into the .ico
│  └─ make_version_info.py  # writes the Windows version resource
├─ src/
│  └─ simplewhisper/
│     ├─ __init__.py
│     ├─ __main__.py      # python -m simplewhisper
│     ├─ assets/          # icon shown by the window (copied by make_icon.py)
│     │  ├─ simplewhisper.ico
│     │  └─ simplewhisper.png
│     ├─ cli.py           # wizard, flags, model commands
│     ├─ gui.py           # Tkinter desktop UI
│     ├─ theme.py         # light/dark theme for the desktop UI
│     ├─ playback.py      # audio playback for the mini player
│     ├─ audio.py         # ffmpeg extraction + PyAV fallback
│     ├─ config.py        # languages, defaults, cache paths
│     ├─ diarize.py       # speaker diarization (sherpa-onnx)
│     ├─ speakers.py      # runs the speaker detection in its own process
│     ├─ errors.py        # shared exception types
│     ├─ cuda.py          # GPU support: find or fetch the CUDA libraries
│     ├─ formatters.py    # SRT / VTT / TXT / JSON / TSV
│     ├─ models.py        # model catalog and downloads
│     ├─ output.py        # output naming and file writing
│     └─ transcribe.py    # faster-whisper wrapper
└─ tests/
   └─ test_core.py
```

## Development

```bash
pip install -e ".[dev]"

pytest
ruff check .
```

The tests cover timestamp formatting, all output formats, language resolution, drag & drop path parsing, file expansion, and the speaker labelling logic.

## Roadmap ideas

- Folder scanning / watch mode
- Transcript preview and editing in the desktop app
- Word-level timestamps and karaoke subtitles
- Naming speakers (map "Speaker 1" to a real name per recording)
- A TUI with a queue and live preview
- Custom vocabulary / prompt hints per file

## Privacy

Transcription runs entirely on your machine. SimpleWhisper does not upload audio or transcripts. The only network access happens when faster-whisper downloads model files from Hugging Face, and when the speaker models are fetched on the first run with `-S`; after that, everything works offline.

## License and attribution

SimpleWhisper is released under the MIT license. It is an unofficial project and is not affiliated with OpenAI.

Whisper model weights are downloaded separately from Hugging Face and carry their own licenses; check the model repository before redistributing them.

Speaker diarization uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with the pyannote segmentation model and 3D-Speaker embedding models. Both are downloaded from public Hugging Face repositories under the Apache 2.0 and MIT licenses respectively; see the model repositories for details.
