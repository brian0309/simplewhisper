"""Small, dependency-free checks for the core helpers."""

import wave

import pytest

from simplewhisper.cli import _expand_inputs, _parse_formats, _split_paths
from simplewhisper.config import format_bytes, format_clock, resolve_language
from simplewhisper.diarize import SpeakerTurn, _progress_hook, assign_speakers
from simplewhisper.errors import JobCancelled
from simplewhisper.formatters import format_srt, format_tsv, format_vtt, timestamp
from simplewhisper.models import model_cache_path
from simplewhisper.output import expand_formats
from simplewhisper.playback import (
    SPEEDS,
    AudioPlayer,
    PlaybackError,
    _atempo_chain,
    _slice_wav,
    decode_to_wav,
)
from simplewhisper.speakers import InlineSpeakerService, SpeakerService
from simplewhisper.transcribe import Segment, Transcription, Word


def test_timestamp_rounding_and_rollover():
    assert timestamp(0) == "00:00:00,000"
    assert timestamp(61.5) == "00:01:01,500"
    assert timestamp(3661.25, ".") == "01:01:01.250"
    assert timestamp(59.9996) == "00:01:00,000"


def _result():
    return Transcription(
        source=None,  # not used by the formatters under test
        model="small",
        language="en",
        language_probability=0.99,
        duration=3.0,
        segments=[
            Segment(0.0, 1.5, " hello"),
            Segment(1.5, 3.0, "world "),
        ],
    )


def test_srt_structure():
    text = format_srt(_result().segments)
    assert text == (
        "1\n00:00:00,000 --> 00:00:01,500\nhello\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\nworld\n"
    )


def test_vtt_structure():
    text = format_vtt(_result().segments)
    assert text.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.500" in text


def test_tsv_escapes_tabs():
    text = format_tsv([Segment(0.0, 1.0, "a\tb")])
    assert text == "00:00:00.000\t00:00:01.000\ta b\n"


def test_expand_formats():
    assert expand_formats(["all"]) == ["srt", "vtt", "txt", "json", "tsv"]
    assert expand_formats(["txt", "txt", "srt"]) == ["txt", "srt"]


def test_parse_formats_rejects_unknown():
    assert _parse_formats("srt, txt") == ["srt", "txt"]
    with pytest.raises(ValueError):
        _parse_formats("docx")


def test_resolve_language():
    assert resolve_language("") is None
    assert resolve_language("auto") is None
    assert resolve_language("English") == "en"
    assert resolve_language("es") == "es"
    assert resolve_language("pt-BR") == "pt"
    with pytest.raises(ValueError):
        resolve_language("klingon")


def test_split_paths_handles_quotes():
    raw = '"C:\\Users\\me\\My Video.mp4" C:\\audio.mp3'
    assert _split_paths(raw, posix=False) == ["C:\\Users\\me\\My Video.mp4", "C:\\audio.mp3"]


def test_split_paths_handles_posix_escapes():
    raw = r"/Users/me/My\ Video.mp4 /tmp/audio.mp3"
    assert _split_paths(raw, posix=True) == ["/Users/me/My Video.mp4", "/tmp/audio.mp3"]


def test_expand_inputs_reports_missing(tmp_path):
    real = tmp_path / "clip with spaces.mp3"
    real.write_bytes(b"not really audio")
    paths, missing = _expand_inputs([f'"{real}"', "definitely-missing.wav"])
    assert paths == [real]
    assert missing == ["definitely-missing.wav"]


def test_format_bytes():
    assert format_bytes(0) == "0.0 B"
    assert format_bytes(512) == "512.0 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(484 * 1024**2) == "484.0 MB"
    assert format_bytes(3.1 * 1024**3) == "3.1 GB"


def test_format_clock():
    assert format_clock(0) == "00:00"
    assert format_clock(5) == "00:05"
    assert format_clock(65) == "01:05"
    assert format_clock(3600) == "1:00:00"
    assert format_clock(3725.4) == "1:02:05"


def test_model_cache_path_finds_snapshot(tmp_path):
    repo = tmp_path / "models--Systran--faster-whisper-tiny"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"weights")

    assert model_cache_path("tiny", tmp_path) == snapshot


def test_model_cache_path_missing_model(tmp_path):
    assert model_cache_path("tiny", tmp_path) is None


def test_model_cache_path_ignores_incomplete_download(tmp_path):
    """A half-finished download must not look like a cached model."""
    repo = tmp_path / "models--Systran--faster-whisper-tiny"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}")

    assert model_cache_path("tiny", tmp_path) is None


# --------------------------------------------------------------------------- #
# Speaker breakdown
# --------------------------------------------------------------------------- #


def _turns():
    return [
        SpeakerTurn(start=0.0, end=2.0, speaker=0),
        SpeakerTurn(start=2.0, end=4.0, speaker=1),
    ]


def test_assign_speakers_labels_segments():
    segments = [
        Segment(start=0.0, end=2.0, text="hello"),
        Segment(start=2.0, end=4.0, text="world"),
    ]
    labelled = assign_speakers(segments, _turns())

    assert [segment.speaker for segment in labelled] == ["Speaker 1", "Speaker 2"]


def test_assign_speakers_numbers_by_first_appearance():
    """A speaker heard later still gets the next free number."""
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker=7),
        SpeakerTurn(start=2.0, end=4.0, speaker=3),
    ]
    labelled = assign_speakers(
        [Segment(start=0.0, end=2.0, text="a"), Segment(start=2.0, end=4.0, text="b")],
        turns,
    )

    assert [segment.speaker for segment in labelled] == ["Speaker 1", "Speaker 2"]


def test_assign_speakers_splits_a_segment_at_the_speaker_change():
    """A sentence spanning two speakers becomes two segments."""
    segment = Segment(
        start=0.0,
        end=4.0,
        text="hello world",
        words=[
            Word(start=0.0, end=1.0, text="hello"),
            Word(start=1.0, end=2.0, text=" "),
            Word(start=2.0, end=3.0, text="world"),
            Word(start=3.0, end=4.0, text="!"),
        ],
    )
    labelled = assign_speakers([segment], _turns())

    assert len(labelled) == 2
    assert labelled[0].speaker == "Speaker 1"
    assert labelled[0].text == "hello"
    assert labelled[1].speaker == "Speaker 2"
    assert labelled[1].text == "world!"


def test_assign_speakers_without_turns_leaves_segments_alone():
    segments = [Segment(start=0.0, end=1.0, text="hello")]
    labelled = assign_speakers(segments, [])

    assert len(labelled) == 1
    assert labelled[0].speaker is None


def test_formats_include_the_speaker_name():
    segment = Segment(start=0.0, end=1.0, text="hello", speaker="Speaker 1")

    assert format_srt([segment]) == "1\n00:00:00,000 --> 00:00:01,000\nSpeaker 1: hello\n"
    assert format_vtt([segment]).endswith("Speaker 1: hello\n")
    assert format_tsv([segment]).endswith("\tSpeaker 1: hello\n")


def test_formats_omit_the_speaker_name_when_unknown():
    segment = Segment(start=0.0, end=1.0, text="hello")

    assert "Speaker" not in format_srt([segment])


# --------------------------------------------------------------------------- #
# Speaker service
# --------------------------------------------------------------------------- #


def test_progress_hook_reports_a_fraction():
    """The sherpa-onnx hook is turned into a 0 to 1 fraction."""
    seen = []
    hook = _progress_hook(seen.append)

    assert hook(1, 4) == 0
    assert hook(4, 4) == 0
    assert seen == [0.25, 1.0]


def test_progress_hook_without_a_reporter():
    assert _progress_hook(None) is None


def test_progress_hook_ignores_an_empty_run():
    seen = []
    hook = _progress_hook(seen.append)

    assert hook(0, 0) == 0
    assert seen == []


def test_the_speaker_service_uses_another_process():
    """The heavy work must not run in the process that owns the window."""
    assert SpeakerService.runs_in_process is True
    assert InlineSpeakerService.runs_in_process is False


def test_the_inline_service_honours_a_cancel_request():
    service = InlineSpeakerService.__new__(InlineSpeakerService)

    with pytest.raises(JobCancelled):
        service.diarize("missing.wav", cancel=lambda: True)


def _write_tone(path, seconds=2.0, rate=16000):
    """Write a short noisy wav file, so the diarizer has something to chew."""
    import numpy as np

    samples = (np.random.default_rng(0).normal(0, 0.1, int(rate * seconds)) * 32767)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.astype("<i2").tobytes())
    return path


def test_the_speaker_service_answers_in_another_process(tmp_path):
    """A full round trip through the child process."""
    from simplewhisper.config import default_model_dir
    from simplewhisper.diarize import diarization_models_cached

    model_dir = default_model_dir()
    if not diarization_models_cached(model_dir):
        pytest.skip("the speaker models are not downloaded")

    media = _write_tone(tmp_path / "tone.wav")
    service = SpeakerService(model_dir)
    try:
        service.wait_ready()
        seen = []
        turns = service.diarize(media, progress=seen.append)

        assert isinstance(turns, list)
        assert all(isinstance(turn, SpeakerTurn) for turn in turns)
        # The progress only ever moves forward.
        assert seen == sorted(seen)
    finally:
        service.close()


def test_the_speaker_service_can_be_cancelled(tmp_path):
    """A cancel request must stop the run instead of waiting it out."""
    import threading

    from simplewhisper.config import default_model_dir
    from simplewhisper.diarize import diarization_models_cached

    model_dir = default_model_dir()
    if not diarization_models_cached(model_dir):
        pytest.skip("the speaker models are not downloaded")

    media = _write_tone(tmp_path / "tone.wav")
    service = SpeakerService(model_dir)
    service.wait_ready()

    failed = []

    def run():
        try:
            service.diarize(media, cancel=lambda: True)
        except JobCancelled:
            failed.append("cancelled")
        except Exception as exc:  # noqa: BLE001
            failed.append(repr(exc))

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=60)

    assert failed == ["cancelled"]
    service.close()


# --------------------------------------------------------------------------- #
# Playback
# --------------------------------------------------------------------------- #


def _write_speech_like_wav(path, seconds=3.0, rate=16000):
    """Write a short stereo wav file, so the decoder has something to read."""
    import numpy as np

    time = np.arange(int(rate * seconds)) / rate
    left = (np.sin(2 * np.pi * 440 * time) * 8000).astype("<i2")
    right = (np.sin(2 * np.pi * 220 * time) * 8000).astype("<i2")
    stereo = np.stack([left, right], axis=1).reshape(-1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(stereo.tobytes())
    return path


def test_decode_to_wav_keeps_the_duration(tmp_path):
    """The decoded file plays for as long as the source does."""
    source = _write_speech_like_wav(tmp_path / "clip.wav", seconds=2.0)
    decoded = tmp_path / "decoded.wav"

    duration = decode_to_wav(source, decoded)

    assert duration == pytest.approx(2.0, abs=0.05)
    with wave.open(str(decoded), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000


def test_decode_to_wav_reports_a_file_without_audio(tmp_path):
    """A file without an audio track cannot be played."""
    source = tmp_path / "silent.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"")

    with pytest.raises(PlaybackError):
        decode_to_wav(source, tmp_path / "decoded.wav")


def test_slice_wav_starts_at_the_given_moment(tmp_path):
    """The tail of a file is what is played after a seek."""
    source = _write_speech_like_wav(tmp_path / "clip.wav", seconds=4.0)
    tail = tmp_path / "tail.wav"

    _slice_wav(source, tail, 2.0)

    with wave.open(str(tail), "rb") as handle:
        assert handle.getnframes() == pytest.approx(2.0 * 16000, abs=2)


def test_the_player_moves_the_position_without_playing(tmp_path):
    """Seeking before the playback starts is remembered."""
    player = AudioPlayer()
    source = _write_speech_like_wav(tmp_path / "clip.wav")

    player.load(source, duration=10.0)
    player.seek(4.0)

    assert player.position == 4.0
    assert player.duration == 10.0
    assert player.is_playing is False


def test_the_player_keeps_the_seek_inside_the_recording(tmp_path):
    """A seek past the end stops at the end."""
    player = AudioPlayer()
    source = _write_speech_like_wav(tmp_path / "clip.wav")

    player.load(source, duration=10.0)
    player.seek(99.0)

    assert player.position == 10.0


def test_the_player_needs_a_file():
    """Playing without a file is an error, not a crash."""
    player = AudioPlayer()

    with pytest.raises(PlaybackError):
        player.play()


def test_the_speed_becomes_a_filter_ffmpeg_understands():
    """Every speed turns into a chain of tempo filters."""
    assert _atempo_chain(1.0) is None
    assert _atempo_chain(2.0) == "atempo=2"
    assert _atempo_chain(4.0) == "atempo=2,atempo=2"
    assert _atempo_chain(0.5) == "atempo=0.5"

    for speed in SPEEDS:
        chain = _atempo_chain(speed)
        if chain is None:
            continue
        for part in chain.split(","):
            factor = float(part.split("=")[1])
            assert 0.5 <= factor <= 2.0


def test_the_player_remembers_the_speed():
    """The speed is a setting of the player, not of the file."""
    player = AudioPlayer()

    player.set_speed(2.0)

    assert player.speed == 2.0


def test_slice_wav_can_change_the_speed(tmp_path):
    """The fallback plays the same samples with another frame rate."""
    source = _write_speech_like_wav(tmp_path / "clip.wav", seconds=4.0)
    fast = tmp_path / "fast.wav"

    _slice_wav(source, fast, 0.0, rate_scale=2.0)

    with wave.open(str(fast), "rb") as handle:
        assert handle.getframerate() == 32000
        assert handle.getnframes() == pytest.approx(4.0 * 16000, abs=2)
