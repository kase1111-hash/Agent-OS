"""
Audio Processing Module

Provides audio analysis capabilities including:
- Audio feature extraction (spectral, temporal)
- Sound classification
- Music analysis
- Speaker identification
- Audio event detection
"""

import io
import logging
import math
import struct
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class AudioEventType(str, Enum):
    """Types of audio events."""

    SPEECH = "speech"
    MUSIC = "music"
    SILENCE = "silence"
    NOISE = "noise"
    APPLAUSE = "applause"
    LAUGHTER = "laughter"
    ALARM = "alarm"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    UNKNOWN = "unknown"


class AudioModel(str, Enum):
    """Available audio analysis models."""

    YAMNET = "yamnet"
    PANNS = "panns"
    OPENL3 = "openl3"
    WAV2VEC = "wav2vec2"
    WHISPER = "whisper"


@dataclass
class AudioFeatures:
    """Extracted audio features."""

    # Temporal features
    duration: float = 0.0  # Duration in seconds
    sample_rate: int = 16000
    channels: int = 1
    rms_energy: float = 0.0  # Root mean square energy
    zero_crossing_rate: float = 0.0

    # Spectral features
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0
    mfcc: List[float] = field(default_factory=list)  # Mel-frequency cepstral coefficients

    # Rhythm features
    tempo: float = 0.0  # BPM
    beat_times: List[float] = field(default_factory=list)

    # Pitch features
    pitch: float = 0.0  # Fundamental frequency in Hz
    pitch_confidence: float = 0.0

    # Embeddings
    embeddings: Optional[List[float]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_silent(self) -> bool:
        """Check if audio is mostly silent."""
        return self.rms_energy < 0.01

    @property
    def has_embeddings(self) -> bool:
        """Check if embeddings are available."""
        return self.embeddings is not None and len(self.embeddings) > 0


@dataclass
class AudioSegment:
    """A segment of audio with timing."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    event_type: AudioEventType = AudioEventType.UNKNOWN
    label: str = ""
    confidence: float = 1.0
    features: Optional[AudioFeatures] = None

    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end - self.start


@dataclass
class AudioInput:
    """Input audio with metadata."""

    data: bytes  # Raw audio bytes (PCM)
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # Bytes per sample
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AudioInput":
        """Load audio from WAV file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            data = wav.readframes(wav.getnframes())

        return cls(
            data=data,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            source=str(path),
        )

    @property
    def duration(self) -> float:
        """Get audio duration in seconds."""
        bytes_per_sample = self.channels * self.sample_width
        samples = len(self.data) // bytes_per_sample
        return samples / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Get number of samples."""
        bytes_per_sample = self.channels * self.sample_width
        return len(self.data) // bytes_per_sample

    def to_wav(self) -> bytes:
        """Convert to WAV format."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(self.sample_width)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.data)
        return buffer.getvalue()

    def get_samples(self) -> List[int]:
        """Get audio samples as integers."""
        if self.sample_width == 2:
            fmt = f"<{len(self.data) // 2}h"
            return list(struct.unpack(fmt, self.data))
        elif self.sample_width == 1:
            return [b - 128 for b in self.data]
        else:
            raise ValueError(f"Unsupported sample width: {self.sample_width}")


@dataclass
class AudioAnalysisResult:
    """Result from audio analysis."""

    features: AudioFeatures
    segments: List[AudioSegment] = field(default_factory=list)
    events: List[AudioEventType] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    transcription: Optional[str] = None
    model: str = ""
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_event(self) -> AudioEventType:
        """Get the most prominent event type."""
        if self.events:
            return self.events[0]
        return AudioEventType.UNKNOWN


# =============================================================================
# Audio Analyzer Interface
# =============================================================================


class AudioAnalyzer(ABC):
    """
    Abstract base class for audio analysis engines.
    """

    def __init__(self, model: AudioModel = AudioModel.YAMNET):
        self.model = model
        self._callbacks: List[Callable[[AudioAnalysisResult], None]] = []

    @abstractmethod
    def analyze(self, audio: AudioInput) -> AudioAnalysisResult:
        """
        Perform comprehensive audio analysis.

        Args:
            audio: Input audio

        Returns:
            AudioAnalysisResult with features and classifications
        """
        pass

    @abstractmethod
    def extract_features(self, audio: AudioInput) -> AudioFeatures:
        """
        Extract audio features.

        Args:
            audio: Input audio

        Returns:
            AudioFeatures with spectral and temporal features
        """
        pass

    @abstractmethod
    def classify(self, audio: AudioInput) -> AudioAnalysisResult:
        """
        Classify audio content.

        Args:
            audio: Input audio

        Returns:
            AudioAnalysisResult with classification labels
        """
        pass

    @abstractmethod
    def detect_events(self, audio: AudioInput) -> AudioAnalysisResult:
        """
        Detect audio events and segments.

        Args:
            audio: Input audio

        Returns:
            AudioAnalysisResult with detected events
        """
        pass

    @abstractmethod
    def embed(self, audio: AudioInput) -> AudioFeatures:
        """
        Generate audio embeddings.

        Args:
            audio: Input audio

        Returns:
            AudioFeatures with embeddings
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the analyzer is available."""
        pass

    def compare_audio(self, audio1: AudioInput, audio2: AudioInput) -> float:
        """
        Compare two audio clips by embedding similarity.

        Args:
            audio1: First audio
            audio2: Second audio

        Returns:
            Similarity score (0-1)
        """
        feat1 = self.embed(audio1)
        feat2 = self.embed(audio2)

        if not feat1.has_embeddings or not feat2.has_embeddings:
            return 0.0

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(feat1.embeddings, feat2.embeddings))
        norm1 = math.sqrt(sum(a * a for a in feat1.embeddings))
        norm2 = math.sqrt(sum(b * b for b in feat2.embeddings))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def on_result(self, callback: Callable[[AudioAnalysisResult], None]) -> None:
        """Register callback for analysis results."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, result: AudioAnalysisResult) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Audio analyzer callback error: {e}")


# =============================================================================
# Basic Feature Extractor
# =============================================================================


class BasicAudioAnalyzer(AudioAnalyzer):
    """
    Basic audio analyzer using simple signal processing.

    Provides feature extraction without ML models.
    """

    def __init__(self, model: AudioModel = AudioModel.YAMNET):
        super().__init__(model)

    def analyze(self, audio: AudioInput) -> AudioAnalysisResult:
        """Perform comprehensive analysis."""
        start_time = time.time()

        features = self.extract_features(audio)
        events = self._detect_basic_events(audio, features)

        result = AudioAnalysisResult(
            features=features,
            events=events,
            model="basic",
            processing_time=time.time() - start_time,
        )

        self._notify_callbacks(result)
        return result

    def extract_features(self, audio: AudioInput) -> AudioFeatures:
        """Extract basic audio features."""
        samples = audio.get_samples()

        if not samples:
            return AudioFeatures(
                duration=audio.duration,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
            )

        # RMS energy
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        normalized_rms = rms / 32768.0 if audio.sample_width == 2 else rms / 128.0

        # Zero crossing rate
        zcr = sum(
            1 for i in range(1, len(samples))
            if (samples[i] >= 0) != (samples[i - 1] >= 0)
        ) / len(samples)

        return AudioFeatures(
            duration=audio.duration,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            rms_energy=normalized_rms,
            zero_crossing_rate=zcr,
        )

    def classify(self, audio: AudioInput) -> AudioAnalysisResult:
        """Basic classification using heuristics."""
        features = self.extract_features(audio)

        labels = []
        events = []

        if features.is_silent:
            labels.append("silence")
            events.append(AudioEventType.SILENCE)
        elif features.zero_crossing_rate > 0.3:
            labels.append("noise")
            events.append(AudioEventType.NOISE)
        elif features.zero_crossing_rate > 0.1:
            labels.append("speech")
            events.append(AudioEventType.SPEECH)
        else:
            labels.append("music")
            events.append(AudioEventType.MUSIC)

        result = AudioAnalysisResult(
            features=features,
            events=events,
            labels=labels,
            model="basic",
        )

        self._notify_callbacks(result)
        return result

    def detect_events(self, audio: AudioInput) -> AudioAnalysisResult:
        """Detect audio events using energy thresholds."""
        samples = audio.get_samples()
        sample_rate = audio.sample_rate
        window_size = int(sample_rate * 0.1)  # 100ms windows

        segments = []
        current_type = AudioEventType.SILENCE
        segment_start = 0.0

        for i in range(0, len(samples), window_size):
            window = samples[i:i + window_size]
            if not window:
                continue

            rms = math.sqrt(sum(s * s for s in window) / len(window))
            normalized_rms = rms / 32768.0

            if normalized_rms < 0.01:
                new_type = AudioEventType.SILENCE
            elif normalized_rms > 0.1:
                new_type = AudioEventType.SPEECH
            else:
                new_type = AudioEventType.NOISE

            current_time = i / sample_rate

            if new_type != current_type:
                if current_time > segment_start:
                    segments.append(
                        AudioSegment(
                            start=segment_start,
                            end=current_time,
                            event_type=current_type,
                        )
                    )
                segment_start = current_time
                current_type = new_type

        # Add final segment
        if audio.duration > segment_start:
            segments.append(
                AudioSegment(
                    start=segment_start,
                    end=audio.duration,
                    event_type=current_type,
                )
            )

        events = list(set(s.event_type for s in segments))

        result = AudioAnalysisResult(
            features=self.extract_features(audio),
            segments=segments,
            events=events,
            model="basic",
        )

        self._notify_callbacks(result)
        return result

    def embed(self, audio: AudioInput) -> AudioFeatures:
        """Generate simple embeddings from features."""
        features = self.extract_features(audio)

        # Create simple embedding from features
        embeddings = [
            features.rms_energy,
            features.zero_crossing_rate,
            features.duration / 10.0,  # Normalized duration
        ]

        # Normalize
        norm = math.sqrt(sum(e * e for e in embeddings))
        if norm > 0:
            embeddings = [e / norm for e in embeddings]

        features.embeddings = embeddings
        return features

    def _detect_basic_events(
        self, audio: AudioInput, features: AudioFeatures
    ) -> List[AudioEventType]:
        """Detect basic events from features."""
        events = []

        if features.is_silent:
            events.append(AudioEventType.SILENCE)
        elif features.zero_crossing_rate > 0.3:
            events.append(AudioEventType.NOISE)
        else:
            events.append(AudioEventType.SPEECH)

        return events

    def is_available(self) -> bool:
        """Basic analyzer is always available."""
        return True


# =============================================================================
# Mock Audio Analyzer
# =============================================================================


class MockAudioAnalyzer(AudioAnalyzer):
    """
    Mock audio analyzer for testing.
    """

    def __init__(self, model: AudioModel = AudioModel.YAMNET):
        super().__init__(model)
        self._mock_labels: List[str] = ["speech", "music"]
        self._mock_events: List[AudioEventType] = [AudioEventType.SPEECH]
        self._embedding_dim = 128

    def set_mock_labels(self, labels: List[str]) -> None:
        """Set mock labels to return."""
        self._mock_labels = labels

    def set_mock_events(self, events: List[AudioEventType]) -> None:
        """Set mock events to return."""
        self._mock_events = events

    def analyze(self, audio: AudioInput) -> AudioAnalysisResult:
        """Return mock analysis."""
        import random

        features = AudioFeatures(
            duration=audio.duration,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            rms_energy=random.random() * 0.5,
            zero_crossing_rate=random.random() * 0.3,
        )

        result = AudioAnalysisResult(
            features=features,
            events=self._mock_events.copy(),
            labels=self._mock_labels.copy(),
            model=self.model.value,
        )

        self._notify_callbacks(result)
        return result

    def extract_features(self, audio: AudioInput) -> AudioFeatures:
        """Return mock features."""
        import random

        return AudioFeatures(
            duration=audio.duration,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            rms_energy=random.random() * 0.5,
            zero_crossing_rate=random.random() * 0.3,
            spectral_centroid=random.random() * 5000,
            tempo=random.uniform(60, 180),
        )

    def classify(self, audio: AudioInput) -> AudioAnalysisResult:
        """Return mock classification."""
        result = AudioAnalysisResult(
            features=self.extract_features(audio),
            labels=self._mock_labels.copy(),
            model=self.model.value,
        )

        self._notify_callbacks(result)
        return result

    def detect_events(self, audio: AudioInput) -> AudioAnalysisResult:
        """Return mock events."""
        import random

        segments = [
            AudioSegment(
                start=0.0,
                end=audio.duration / 2,
                event_type=self._mock_events[0] if self._mock_events else AudioEventType.SPEECH,
            ),
            AudioSegment(
                start=audio.duration / 2,
                end=audio.duration,
                event_type=AudioEventType.SILENCE,
            ),
        ]

        result = AudioAnalysisResult(
            features=self.extract_features(audio),
            segments=segments,
            events=self._mock_events.copy(),
            model=self.model.value,
        )

        self._notify_callbacks(result)
        return result

    def embed(self, audio: AudioInput) -> AudioFeatures:
        """Return mock embeddings."""
        import random

        features = self.extract_features(audio)

        embeddings = [random.gauss(0, 1) for _ in range(self._embedding_dim)]
        norm = math.sqrt(sum(e * e for e in embeddings))
        if norm > 0:
            embeddings = [e / norm for e in embeddings]

        features.embeddings = embeddings
        return features

    def is_available(self) -> bool:
        """Mock is always available."""
        return True


# =============================================================================
# YAMNet Audio Analyzer
# =============================================================================


class YAMNetAnalyzer(AudioAnalyzer):
    """
    Audio analyzer using YAMNet for sound classification.

    YAMNet is trained on AudioSet and can classify 521 sound classes.

    Install:
        pip install tensorflow tensorflow-hub
    """

    def __init__(self, model: AudioModel = AudioModel.YAMNET):
        super().__init__(model)
        self._model = None
        self._class_names = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize YAMNet model."""
        try:
            import tensorflow_hub as hub

            self._model = hub.load("https://tfhub.dev/google/yamnet/1")

            # Load class names
            import csv
            import urllib.request

            class_map_path = self._model.class_map_path().numpy().decode("utf-8")
            self._class_names = []
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._class_names.append(row["display_name"])

            logger.info("YAMNet initialized")

        except ImportError:
            logger.warning("TensorFlow Hub not installed for YAMNet")
        except Exception as e:
            logger.error(f"Failed to initialize YAMNet: {e}")

    def _prepare_audio(self, audio: AudioInput):
        """Prepare audio for YAMNet (16kHz mono float32)."""
        import numpy as np

        samples = np.array(audio.get_samples(), dtype=np.float32)
        samples = samples / 32768.0  # Normalize to [-1, 1]

        # Resample to 16kHz if needed
        if audio.sample_rate != 16000:
            # Simple resampling (use librosa for better quality)
            ratio = 16000 / audio.sample_rate
            new_length = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples) - 1, new_length)
            samples = np.interp(indices, np.arange(len(samples)), samples)

        return samples

    def analyze(self, audio: AudioInput) -> AudioAnalysisResult:
        """Comprehensive analysis with YAMNet."""
        start_time = time.time()

        # Get classifications
        classification = self.classify(audio)

        # Merge with basic features
        basic = BasicAudioAnalyzer()
        features = basic.extract_features(audio)

        result = AudioAnalysisResult(
            features=features,
            segments=classification.segments,
            events=classification.events,
            labels=classification.labels,
            model=self.model.value,
            processing_time=time.time() - start_time,
        )

        self._notify_callbacks(result)
        return result

    def extract_features(self, audio: AudioInput) -> AudioFeatures:
        """Extract features (uses basic extractor + embeddings)."""
        basic = BasicAudioAnalyzer()
        features = basic.extract_features(audio)

        # Add embeddings
        embedded = self.embed(audio)
        features.embeddings = embedded.embeddings

        return features

    def classify(self, audio: AudioInput) -> AudioAnalysisResult:
        """Classify audio using YAMNet."""
        if not self.is_available():
            return AudioAnalysisResult(
                features=AudioFeatures(),
                metadata={"error": "model_not_loaded"},
            )

        try:
            import numpy as np

            waveform = self._prepare_audio(audio)
            scores, embeddings, spectrogram = self._model(waveform)

            # Get top classes
            scores = scores.numpy()
            mean_scores = np.mean(scores, axis=0)
            top_indices = np.argsort(mean_scores)[::-1][:10]

            labels = [self._class_names[i] for i in top_indices]
            confidences = [float(mean_scores[i]) for i in top_indices]

            # Map to event types
            events = self._labels_to_events(labels[:3])

            result = AudioAnalysisResult(
                features=AudioFeatures(),
                labels=labels,
                events=events,
                model=self.model.value,
                metadata={"confidences": dict(zip(labels, confidences))},
            )

            self._notify_callbacks(result)
            return result

        except Exception as e:
            logger.error(f"YAMNet classification error: {e}")
            return AudioAnalysisResult(
                features=AudioFeatures(),
                metadata={"error": str(e)},
            )

    def detect_events(self, audio: AudioInput) -> AudioAnalysisResult:
        """Detect events with temporal segmentation."""
        if not self.is_available():
            return AudioAnalysisResult(
                features=AudioFeatures(),
                metadata={"error": "model_not_loaded"},
            )

        try:
            import numpy as np

            waveform = self._prepare_audio(audio)
            scores, embeddings, spectrogram = self._model(waveform)

            scores = scores.numpy()
            frame_duration = 0.48  # YAMNet frame duration

            segments = []
            for i, frame_scores in enumerate(scores):
                top_idx = np.argmax(frame_scores)
                label = self._class_names[top_idx]
                confidence = float(frame_scores[top_idx])

                start = i * frame_duration
                end = (i + 1) * frame_duration

                segments.append(
                    AudioSegment(
                        start=start,
                        end=end,
                        label=label,
                        confidence=confidence,
                        event_type=self._label_to_event(label),
                    )
                )

            # Merge consecutive same-label segments
            merged = self._merge_segments(segments)

            events = list(set(s.event_type for s in merged))

            result = AudioAnalysisResult(
                features=AudioFeatures(),
                segments=merged,
                events=events,
                model=self.model.value,
            )

            self._notify_callbacks(result)
            return result

        except Exception as e:
            logger.error(f"YAMNet event detection error: {e}")
            return AudioAnalysisResult(
                features=AudioFeatures(),
                metadata={"error": str(e)},
            )

    def embed(self, audio: AudioInput) -> AudioFeatures:
        """Generate YAMNet embeddings."""
        if not self.is_available():
            return AudioFeatures()

        try:
            import numpy as np

            waveform = self._prepare_audio(audio)
            scores, embeddings, spectrogram = self._model(waveform)

            # Average embeddings across time
            mean_embedding = np.mean(embeddings.numpy(), axis=0).tolist()

            return AudioFeatures(
                duration=audio.duration,
                sample_rate=audio.sample_rate,
                embeddings=mean_embedding,
            )

        except Exception as e:
            logger.error(f"YAMNet embedding error: {e}")
            return AudioFeatures()

    def _labels_to_events(self, labels: List[str]) -> List[AudioEventType]:
        """Convert YAMNet labels to event types."""
        return [self._label_to_event(l) for l in labels]

    def _label_to_event(self, label: str) -> AudioEventType:
        """Convert a single label to event type."""
        label_lower = label.lower()

        if any(kw in label_lower for kw in ["speech", "talk", "voice"]):
            return AudioEventType.SPEECH
        elif any(kw in label_lower for kw in ["music", "song", "instrument"]):
            return AudioEventType.MUSIC
        elif any(kw in label_lower for kw in ["silence", "quiet"]):
            return AudioEventType.SILENCE
        elif any(kw in label_lower for kw in ["applause", "clap"]):
            return AudioEventType.APPLAUSE
        elif any(kw in label_lower for kw in ["laugh"]):
            return AudioEventType.LAUGHTER
        elif any(kw in label_lower for kw in ["alarm", "siren"]):
            return AudioEventType.ALARM
        elif any(kw in label_lower for kw in ["car", "vehicle", "engine"]):
            return AudioEventType.VEHICLE
        elif any(kw in label_lower for kw in ["dog", "cat", "bird", "animal"]):
            return AudioEventType.ANIMAL
        else:
            return AudioEventType.UNKNOWN

    def _merge_segments(self, segments: List[AudioSegment]) -> List[AudioSegment]:
        """Merge consecutive segments with same label."""
        if not segments:
            return []

        merged = [segments[0]]

        for seg in segments[1:]:
            if seg.label == merged[-1].label:
                # Extend previous segment
                merged[-1] = AudioSegment(
                    start=merged[-1].start,
                    end=seg.end,
                    label=seg.label,
                    event_type=seg.event_type,
                    confidence=(merged[-1].confidence + seg.confidence) / 2,
                )
            else:
                merged.append(seg)

        return merged

    def is_available(self) -> bool:
        """Check if YAMNet is loaded."""
        return self._model is not None


# =============================================================================
# Factory Function
# =============================================================================


def create_audio_analyzer(
    analyzer_type: str = "auto",
    model: Optional[AudioModel] = None,
    **kwargs,
) -> AudioAnalyzer:
    """
    Create an audio analyzer.

    Args:
        analyzer_type: Type of analyzer ("yamnet", "basic", "mock", "auto")
        model: Audio model to use
        **kwargs: Additional analyzer-specific arguments

    Returns:
        AudioAnalyzer instance
    """
    if analyzer_type == "mock":
        return MockAudioAnalyzer(model or AudioModel.YAMNET)

    if analyzer_type == "basic":
        return BasicAudioAnalyzer(model or AudioModel.YAMNET)

    if analyzer_type == "yamnet":
        return YAMNetAnalyzer(model or AudioModel.YAMNET)

    # Auto-detect best available analyzer
    if analyzer_type == "auto":
        # Try YAMNet first
        yamnet = YAMNetAnalyzer()
        if yamnet.is_available():
            logger.info("Using YAMNet for audio analysis")
            return yamnet

        # Fall back to basic analyzer
        logger.warning("YAMNet not available, using basic audio analyzer")
        return BasicAudioAnalyzer()

    raise ValueError(f"Unknown audio analyzer type: {analyzer_type}")
