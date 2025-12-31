"""
Wake Word Detection

Provides wake word/hotword detection using Porcupine or compatible engines.
"""

import logging
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .audio import AudioCapture, AudioFormat, create_audio_capture

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class WakeWord(str, Enum):
    """Predefined wake words."""

    AGENT = "agent"
    COMPUTER = "computer"
    HEY_AGENT = "hey_agent"
    JARVIS = "jarvis"
    CUSTOM = "custom"


@dataclass
class WakeWordEvent:
    """Event triggered when wake word is detected."""

    keyword: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    audio_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""

    keywords: List[str] = field(default_factory=lambda: ["agent"])
    sensitivity: float = 0.5  # Detection sensitivity (0.0 - 1.0)
    audio_gain: float = 1.0  # Audio input gain
    sample_rate: int = 16000  # Required sample rate for most engines
    frame_length: int = 512  # Audio frame length in samples


# =============================================================================
# Wake Word Detector Interface
# =============================================================================


class WakeWordDetector(ABC):
    """
    Abstract base class for wake word detection engines.

    Implementations can use Porcupine, Snowboy, Mycroft Precise, or custom models.
    """

    def __init__(self, config: Optional[WakeWordConfig] = None):
        self.config = config or WakeWordConfig()
        self._callbacks: List[Callable[[WakeWordEvent], None]] = []
        self._is_listening = False
        self._listen_thread: Optional[threading.Thread] = None

    @abstractmethod
    def detect(self, audio_data: bytes) -> Optional[WakeWordEvent]:
        """
        Process audio data and detect wake word.

        Args:
            audio_data: Raw PCM audio data (16-bit, mono)

        Returns:
            WakeWordEvent if detected, None otherwise
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the detector is available and configured."""
        pass

    @abstractmethod
    def get_keywords(self) -> List[str]:
        """Get list of supported keywords."""
        pass

    def on_wake_word(self, callback: Callable[[WakeWordEvent], None]) -> None:
        """Register callback for wake word detection."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: WakeWordEvent) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Wake word callback error: {e}")

    def start_listening(self, audio_capture: Optional[AudioCapture] = None) -> bool:
        """
        Start continuous wake word detection.

        Args:
            audio_capture: Audio capture source (creates default if None)

        Returns:
            True if started successfully
        """
        if self._is_listening:
            return True

        if not self.is_available():
            logger.error("Wake word detector not available")
            return False

        self._is_listening = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            args=(audio_capture,),
            daemon=True,
        )
        self._listen_thread.start()
        return True

    def stop_listening(self) -> None:
        """Stop wake word detection."""
        self._is_listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
            self._listen_thread = None

    def _listen_loop(self, audio_capture: Optional[AudioCapture]) -> None:
        """Background listening loop."""
        capture = audio_capture or create_audio_capture(
            sample_rate=self.config.sample_rate,
            channels=1,
        )

        try:
            capture.start()
            frame_bytes = self.config.frame_length * 2  # 16-bit = 2 bytes per sample

            while self._is_listening:
                audio_data = capture.read(frame_bytes)
                if audio_data:
                    event = self.detect(audio_data)
                    if event:
                        self._notify_callbacks(event)
        except Exception as e:
            logger.error(f"Wake word listen loop error: {e}")
        finally:
            capture.stop()

    @property
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._is_listening


# =============================================================================
# Mock Wake Word Detector
# =============================================================================


class MockWakeWordDetector(WakeWordDetector):
    """
    Mock wake word detector for testing.

    Can be triggered programmatically.
    """

    def __init__(self, config: Optional[WakeWordConfig] = None):
        super().__init__(config)
        self._trigger_next = False
        self._trigger_keyword: Optional[str] = None
        self._detection_count = 0

    def detect(self, audio_data: bytes) -> Optional[WakeWordEvent]:
        """Check for triggered detection."""
        if self._trigger_next:
            self._trigger_next = False
            self._detection_count += 1
            keyword = self._trigger_keyword or self.config.keywords[0]
            return WakeWordEvent(
                keyword=keyword,
                confidence=0.95,
                audio_data=audio_data,
            )
        return None

    def is_available(self) -> bool:
        """Mock is always available."""
        return True

    def get_keywords(self) -> List[str]:
        """Return configured keywords."""
        return self.config.keywords

    def trigger_detection(self, keyword: Optional[str] = None) -> None:
        """Trigger a detection on next audio frame."""
        self._trigger_next = True
        self._trigger_keyword = keyword

    def get_detection_count(self) -> int:
        """Get number of detections for testing."""
        return self._detection_count

    def reset(self) -> None:
        """Reset detection state."""
        self._trigger_next = False
        self._trigger_keyword = None
        self._detection_count = 0


# =============================================================================
# Energy-Based Wake Word Detector
# =============================================================================


class EnergyWakeWordDetector(WakeWordDetector):
    """
    Simple energy-based wake word detector.

    Detects audio above a threshold as "activation".
    Useful for testing and as a fallback.
    """

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        energy_threshold: float = 0.1,
        min_duration: float = 0.3,
    ):
        super().__init__(config)
        self.energy_threshold = energy_threshold
        self.min_duration = min_duration
        self._active_frames = 0
        self._frames_required = int(
            min_duration * self.config.sample_rate / self.config.frame_length
        )

    def detect(self, audio_data: bytes) -> Optional[WakeWordEvent]:
        """Detect based on audio energy level."""
        # Calculate RMS energy
        if len(audio_data) < 2:
            return None

        samples = struct.unpack(f"<{len(audio_data) // 2}h", audio_data)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        normalized_rms = rms / 32768.0  # Normalize to 0-1

        if normalized_rms > self.energy_threshold:
            self._active_frames += 1
        else:
            self._active_frames = 0

        if self._active_frames >= self._frames_required:
            self._active_frames = 0
            return WakeWordEvent(
                keyword="activation",
                confidence=min(1.0, normalized_rms / self.energy_threshold),
                audio_data=audio_data,
            )

        return None

    def is_available(self) -> bool:
        """Energy detector is always available."""
        return True

    def get_keywords(self) -> List[str]:
        """Return activation keyword."""
        return ["activation"]


# =============================================================================
# Porcupine Wake Word Detector
# =============================================================================


class PorcupineWakeWordDetector(WakeWordDetector):
    """
    Wake word detector using Picovoice Porcupine.

    Porcupine is a highly accurate, lightweight wake word engine.

    Install:
        pip install pvporcupine

    Requires access key from Picovoice console.
    """

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        access_key: Optional[str] = None,
        keyword_paths: Optional[List[str]] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__(config)
        self.access_key = access_key
        self.keyword_paths = keyword_paths or []
        self.model_path = model_path
        self._porcupine = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Porcupine engine."""
        if not self.access_key:
            logger.warning("Porcupine access key not provided")
            return

        try:
            import pvporcupine

            keywords = []
            sensitivities = []

            # Use built-in keywords if no custom paths
            if not self.keyword_paths:
                # Check for built-in keywords
                for kw in self.config.keywords:
                    if kw.lower() in [k.value for k in pvporcupine.BuiltInKeywords]:
                        keywords.append(kw.lower())
                        sensitivities.append(self.config.sensitivity)

                if keywords:
                    self._porcupine = pvporcupine.create(
                        access_key=self.access_key,
                        keywords=keywords,
                        sensitivities=sensitivities,
                        model_path=self.model_path,
                    )
            else:
                # Use custom keyword files
                self._porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=self.keyword_paths,
                    sensitivities=[self.config.sensitivity] * len(self.keyword_paths),
                    model_path=self.model_path,
                )

            if self._porcupine:
                # Update config with Porcupine requirements
                self.config.sample_rate = self._porcupine.sample_rate
                self.config.frame_length = self._porcupine.frame_length
                logger.info("Porcupine initialized successfully")

        except ImportError:
            logger.warning("pvporcupine not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")

    def detect(self, audio_data: bytes) -> Optional[WakeWordEvent]:
        """Process audio and detect wake word."""
        if not self._porcupine:
            return None

        # Convert bytes to int16 array
        if len(audio_data) < self._porcupine.frame_length * 2:
            return None

        samples = struct.unpack(
            f"<{self._porcupine.frame_length}h",
            audio_data[: self._porcupine.frame_length * 2],
        )

        try:
            keyword_index = self._porcupine.process(list(samples))

            if keyword_index >= 0:
                keywords = self.get_keywords()
                keyword = keywords[keyword_index] if keyword_index < len(keywords) else "unknown"
                return WakeWordEvent(
                    keyword=keyword,
                    confidence=1.0,  # Porcupine doesn't provide confidence
                    audio_data=audio_data,
                )
        except Exception as e:
            logger.error(f"Porcupine process error: {e}")

        return None

    def is_available(self) -> bool:
        """Check if Porcupine is initialized."""
        return self._porcupine is not None

    def get_keywords(self) -> List[str]:
        """Get configured keywords."""
        if self.keyword_paths:
            return [Path(p).stem for p in self.keyword_paths]
        return self.config.keywords

    def __del__(self):
        """Clean up Porcupine resources."""
        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass


# =============================================================================
# Snowboy Wake Word Detector (Legacy)
# =============================================================================


class SnowboyWakeWordDetector(WakeWordDetector):
    """
    Wake word detector using Snowboy (legacy, discontinued).

    Note: Snowboy is no longer actively maintained but still works
    for existing installations.

    Install:
        pip install snowboy
    """

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        model_path: Optional[str] = None,
        resource_path: Optional[str] = None,
    ):
        super().__init__(config)
        self.model_path = model_path
        self.resource_path = resource_path
        self._detector = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Snowboy detector."""
        if not self.model_path:
            logger.warning("Snowboy model path not provided")
            return

        try:
            import snowboy.snowboydecoder as snowboy

            resource = self.resource_path or "resources/common.res"

            self._detector = snowboy.HotwordDetector(
                self.model_path,
                resource,
                sensitivity=str(self.config.sensitivity),
                audio_gain=self.config.audio_gain,
            )

            logger.info("Snowboy initialized successfully")

        except ImportError:
            logger.warning("snowboy not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Snowboy: {e}")

    def detect(self, audio_data: bytes) -> Optional[WakeWordEvent]:
        """Process audio with Snowboy."""
        if not self._detector:
            return None

        try:
            result = self._detector.detector.RunDetection(audio_data)
            if result > 0:
                keywords = self.get_keywords()
                keyword = keywords[result - 1] if result <= len(keywords) else "unknown"
                return WakeWordEvent(
                    keyword=keyword,
                    confidence=0.9,
                    audio_data=audio_data,
                )
        except Exception as e:
            logger.error(f"Snowboy detection error: {e}")

        return None

    def is_available(self) -> bool:
        """Check if Snowboy is initialized."""
        return self._detector is not None

    def get_keywords(self) -> List[str]:
        """Get model keywords."""
        if self.model_path:
            return [Path(self.model_path).stem]
        return self.config.keywords

    def __del__(self):
        """Clean up Snowboy resources."""
        if self._detector:
            try:
                self._detector.terminate()
            except Exception:
                pass


# =============================================================================
# Keyword Spotter (Simple Pattern Matching)
# =============================================================================


class KeywordSpotter(WakeWordDetector):
    """
    Simple keyword spotter using STT and pattern matching.

    This is a fallback option when dedicated wake word engines
    are not available. It transcribes audio and looks for keywords.
    """

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        stt_engine: Optional[Any] = None,  # STTEngine
    ):
        super().__init__(config)
        self.stt_engine = stt_engine
        self._audio_buffer: List[bytes] = []
        self._buffer_duration = 2.0  # seconds
        self._buffer_samples = int(self._buffer_duration * self.config.sample_rate)

    def detect(self, audio_data: bytes) -> Optional[WakeWordEvent]:
        """Transcribe and search for keywords."""
        if not self.stt_engine:
            return None

        # Add to buffer
        self._audio_buffer.append(audio_data)

        # Calculate total samples in buffer
        total_samples = sum(len(d) // 2 for d in self._audio_buffer)

        if total_samples >= self._buffer_samples:
            # Transcribe buffer
            combined = b"".join(self._audio_buffer)
            self._audio_buffer.clear()

            try:
                from .audio import AudioFormat
                from .stt import STTResult

                result = self.stt_engine.transcribe(combined, AudioFormat.PCM)

                if result and result.text:
                    text_lower = result.text.lower()
                    for keyword in self.config.keywords:
                        if keyword.lower() in text_lower:
                            return WakeWordEvent(
                                keyword=keyword,
                                confidence=result.confidence,
                                audio_data=combined,
                            )
            except Exception as e:
                logger.error(f"Keyword spotting error: {e}")

        return None

    def is_available(self) -> bool:
        """Check if STT engine is available."""
        return self.stt_engine is not None and self.stt_engine.is_available()

    def get_keywords(self) -> List[str]:
        """Get configured keywords."""
        return self.config.keywords


# =============================================================================
# Factory Function
# =============================================================================


def create_wake_word_detector(
    detector_type: str = "auto",
    config: Optional[WakeWordConfig] = None,
    **kwargs,
) -> WakeWordDetector:
    """
    Create a wake word detector.

    Args:
        detector_type: Type of detector ("porcupine", "snowboy", "energy", "mock", "auto")
        config: Wake word configuration
        **kwargs: Additional detector-specific arguments

    Returns:
        WakeWordDetector instance
    """
    if detector_type == "mock":
        return MockWakeWordDetector(config)

    if detector_type == "energy":
        return EnergyWakeWordDetector(config, **kwargs)

    if detector_type == "porcupine":
        return PorcupineWakeWordDetector(config, **kwargs)

    if detector_type == "snowboy":
        return SnowboyWakeWordDetector(config, **kwargs)

    if detector_type == "keyword":
        return KeywordSpotter(config, **kwargs)

    # Auto-detect best available detector
    if detector_type == "auto":
        # Try Porcupine first
        access_key = kwargs.get("access_key")
        if access_key:
            porcupine = PorcupineWakeWordDetector(config, access_key=access_key)
            if porcupine.is_available():
                logger.info("Using Porcupine for wake word detection")
                return porcupine

        # Fall back to energy-based detection
        logger.warning("No wake word engine available, using energy-based detection")
        return EnergyWakeWordDetector(config)

    raise ValueError(f"Unknown wake word detector type: {detector_type}")
