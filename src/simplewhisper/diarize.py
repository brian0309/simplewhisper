"""Speaker diarization: work out who spoke when.

The engine is sherpa-onnx, which runs on ONNX Runtime and needs neither
PyTorch nor a Hugging Face account. Two public model files are involved: a
pyannote segmentation model that finds the speech regions, and a 3D-Speaker
embedding model that recognises the voice in each region.

Both files are downloaded on first use into the same model cache the Whisper
models use, so they are fetched once and then work offline.

The pipeline is used in two stages. First it segments the audio and labels
the turns, which gives good turn boundaries. Those labels are then refined:
one speaker embedding is averaged per detected speaker and the averages are
clustered again, which merges voices the first pass split apart. A fixed
speaker count can be requested instead, which groups the turns into exactly
that many speakers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .transcribe import Segment

log = logging.getLogger("simplewhisper")

# Called with a fraction between 0 and 1 while the audio is worked through.
ProgressCallback = Callable[[float], None]

# Public Hugging Face repos, so no token or account is needed.
SEGMENTATION_REPO = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0"
SEGMENTATION_FILE = "model.onnx"
EMBEDDING_REPO = "csukuangfj/speaker-embedding-models"
EMBEDDING_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"

# Threshold used for the first pass, which finds the turn boundaries.
DEFAULT_THRESHOLD = 0.7

# Threshold used to merge the detected speakers in the second pass. This one
# is deliberately lower than the first: the first pass tends to split a
# single voice into several labels, and merging those averages back together
# recovers the real speaker count.
MERGE_THRESHOLD = 0.5


def _progress_hook(progress: Optional[ProgressCallback]):
    """Wrap a fraction reporter for the sherpa-onnx progress hook.

    sherpa-onnx calls the hook with the number of processed chunks and
    the total number of chunks. Returning zero means "carry on"; any
    other value aborts the run.
    """
    if progress is None:
        return None

    def hook(processed: int, total: int) -> int:
        if total > 0:
            progress(min(max(processed / total, 0.0), 1.0))
        return 0

    return hook


@dataclass
class SpeakerTurn:
    """A stretch of audio attributed to one speaker."""

    start: float
    end: float
    speaker: int


class Diarizer:
    """Runs speaker diarization over media files."""

    def __init__(self, pipeline, model_dir: Path, num_threads: int) -> None:
        self._pipeline = pipeline
        self._model_dir = Path(model_dir)
        self._num_threads = num_threads

    @property
    def sample_rate(self) -> int:
        return int(self._pipeline.sample_rate)

    def diarize(
        self,
        media: Path,
        num_speakers: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> List[SpeakerTurn]:
        """Return the speaker turns found in ``media``.

        When ``num_speakers`` is given, the turns are grouped into exactly
        that many speakers. Otherwise the number of speakers is detected.

        ``progress`` is called with a fraction between 0 and 1 as the audio
        is worked through. Note that sherpa-onnx holds the GIL for the
        whole run, so this callback is only useful for a progress readout;
        it does not keep other threads alive.
        """
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(media), sampling_rate=self.sample_rate)
        result = (
            self._pipeline.process(audio, _progress_hook(progress))
            .sort_by_start_time()
        )
        turns = [
            SpeakerTurn(
                start=float(turn.start),
                end=float(turn.end),
                speaker=int(turn.speaker),
            )
            for turn in result
        ]
        if not turns:
            return turns

        embeddings = self._embed_turns(audio, turns)
        if not embeddings:
            return turns

        if num_speakers and num_speakers > 0:
            labels = self._group_into(embeddings, num_speakers)
        else:
            labels = self._merge_speakers(embeddings)

        regrouped = [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=label)
            for (turn, _), label in zip(embeddings, labels)
        ]
        return _merge_neighbours(regrouped)

    def _embed_turns(self, audio, turns: Sequence[SpeakerTurn]) -> List[Tuple[SpeakerTurn, object]]:
        """Compute one speaker embedding per turn.

        Turns too short to embed are dropped, so the result can be shorter
        than ``turns``.
        """
        import numpy as np
        import sherpa_onnx

        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self._model_dir / EMBEDDING_FILE),
                num_threads=self._num_threads,
            )
        )

        embedded: List[Tuple[SpeakerTurn, object]] = []
        sample_rate = self.sample_rate
        for turn in turns:
            start = max(0, int(turn.start * sample_rate))
            end = min(len(audio), int(turn.end * sample_rate))
            if end - start <= 0:
                continue
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate, audio[start:end].tolist())
            stream.input_finished()
            if not extractor.is_ready(stream):
                continue
            embedding = np.array(extractor.compute(stream), dtype=np.float32)
            embedded.append((turn, embedding))
        return embedded

    def _group_into(self, embeddings, num_speakers: int) -> List[int]:
        """Group the turns into exactly ``num_speakers`` speakers."""
        if len(embeddings) < num_speakers:
            return [turn.speaker for turn, _ in embeddings]

        import numpy as np
        import sherpa_onnx

        features = np.stack([embedding for _, embedding in embeddings])
        clustering = sherpa_onnx.FastClustering(
            sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers, threshold=MERGE_THRESHOLD
            )
        )
        raw = clustering(features)
        return _renumber(raw)

    def _merge_speakers(self, embeddings) -> List[int]:
        """Merge the detected speakers that actually sound the same.

        The first pass labels every turn, but it often gives one voice
        several labels. Averaging the embeddings per label and clustering the
        averages puts those back together.
        """
        import numpy as np
        import sherpa_onnx

        by_speaker: Dict[int, List] = {}
        for turn, embedding in embeddings:
            by_speaker.setdefault(turn.speaker, []).append(embedding)

        speakers = sorted(by_speaker)
        if len(speakers) < 2:
            return [turn.speaker for turn, _ in embeddings]

        # Keep the speaker label alongside its centroid so the mapping back
        # to the turns stays correct.
        centroids = []
        centroid_speakers = []
        for speaker in speakers:
            centroid = np.mean(by_speaker[speaker], axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                continue
            centroids.append(centroid / norm)
            centroid_speakers.append(speaker)

        if len(centroids) < 2:
            return [turn.speaker for turn, _ in embeddings]

        clustering = sherpa_onnx.FastClustering(
            sherpa_onnx.FastClusteringConfig(
                num_clusters=-1, threshold=MERGE_THRESHOLD
            )
        )
        raw = clustering(np.stack(centroids))

        # Map each original speaker label onto its merged group.
        merged: Dict[int, int] = {}
        for index, speaker in enumerate(centroid_speakers):
            if index < len(raw):
                merged[speaker] = raw[index]
        return [merged.get(turn.speaker, turn.speaker) for turn, _ in embeddings]


def _renumber(labels: Sequence[int]) -> List[int]:
    """Turn sparse cluster labels into 0, 1, 2, ... without gaps."""
    mapping: Dict[int, int] = {}
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
    return [mapping[label] for label in labels]


def _merge_neighbours(turns: Sequence[SpeakerTurn]) -> List[SpeakerTurn]:
    """Merge consecutive turns that belong to the same speaker."""
    merged: List[SpeakerTurn] = []
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker:
            previous = merged[-1]
            merged[-1] = SpeakerTurn(
                start=previous.start, end=turn.end, speaker=previous.speaker
            )
        else:
            merged.append(turn)
    return merged


def diarization_model_dir(model_dir: Path) -> Path:
    """Folder holding the diarization model files."""
    return Path(model_dir) / "diarization"


def diarization_models_cached(model_dir: Path) -> bool:
    """True when both diarization model files are already on disk."""
    folder = diarization_model_dir(model_dir)
    return (folder / SEGMENTATION_FILE).is_file() and (folder / EMBEDDING_FILE).is_file()


def load_diarizer(
    model_dir: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    num_threads: Optional[int] = None,
) -> Diarizer:
    """Build the diarization pipeline from the cached model files."""
    import sherpa_onnx

    folder = diarization_model_dir(model_dir)
    segmentation = folder / SEGMENTATION_FILE
    embedding = folder / EMBEDDING_FILE
    for path in (segmentation, embedding):
        if not path.is_file():
            raise FileNotFoundError(
                f"diarization model '{path.name}' is missing from {folder}. "
                "Run the download again to fetch it."
            )

    if num_threads is None:
        num_threads = max(1, min(4, os.cpu_count() or 1))

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation),
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding),
            num_threads=num_threads,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("the diarization configuration was rejected by sherpa-onnx")

    return Diarizer(sherpa_onnx.OfflineSpeakerDiarization(config), folder, num_threads)


def _speaker_for_span(
    start: float, end: float, turns: Sequence[SpeakerTurn]
) -> Optional[int]:
    """Return the speaker who covers ``start``-``end`` best."""
    best_speaker: Optional[int] = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker

    if best_speaker is not None:
        return best_speaker

    # No overlap at all: fall back to the turn whose midpoint is nearest.
    if not turns:
        return None
    middle = (start + end) / 2
    nearest = min(turns, key=lambda turn: abs((turn.start + turn.end) / 2 - middle))
    return nearest.speaker


def assign_speakers(
    segments: Sequence[Segment], turns: Sequence[SpeakerTurn]
) -> List[Segment]:
    """Label every segment with the speaker who said it.

    When the segments carry word-level timestamps, a segment that spans a
    speaker change is split into one segment per speaker. Segments without
    word timestamps keep their shape and simply gain a label.

    Speakers are numbered by order of first appearance, so the first voice
    heard is always "Speaker 1".
    """
    if not turns:
        return list(segments)

    labelled: List[Segment] = []
    labels: dict = {}

    def label_for(speaker: Optional[int]) -> Optional[str]:
        if speaker is None:
            return None
        if speaker not in labels:
            labels[speaker] = f"Speaker {len(labels) + 1}"
        return labels[speaker]

    for segment in segments:
        words = segment.words or []
        if not words:
            speaker = _speaker_for_span(segment.start, segment.end, turns)
            labelled.append(
                Segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=label_for(speaker),
                )
            )
            continue

        # Group the words into runs that share the same speaker.
        runs: List[Tuple[Optional[int], list]] = []
        for word in words:
            speaker = _speaker_for_span(word.start, word.end, turns)
            if runs and runs[-1][0] == speaker:
                runs[-1][1].append(word)
            else:
                runs.append((speaker, [word]))

        if len(runs) == 1:
            speaker = runs[0][0]
            labelled.append(
                Segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=label_for(speaker),
                    words=list(words),
                )
            )
            continue

        for speaker, run in runs:
            text = "".join(word.text for word in run).strip()
            if not text:
                continue
            labelled.append(
                Segment(
                    start=run[0].start,
                    end=run[-1].end,
                    text=text,
                    speaker=label_for(speaker),
                    words=list(run),
                )
            )

    return labelled
