"""
Speech-to-Text (STT) Engine

Provides speech recognition functionality using Whisper or compatible engines.
"""

import json
import logging
import re
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .audio import AudioBuffer, AudioFormat, pcm_to_wav

logger = logging.getLogger(__name__)


# =============================================================================
# Security: Path Validation for Subprocess Calls
# =============================================================================


class PathValidationError(Exception):
    """Raised when a path fails security validation."""

    pass


def validate_audio_path(path: Path) -> Path:
    """
    Validate an audio file path before passing to subprocess.

    This prevents command injection attacks by:
    1. Resolving to absolute path to prevent path traversal
    2. Checking the path doesn't contain shell metacharacters
    3. Ensuring the file exists and is a regular file

    Args:
        path: Path to validate

    Returns:
        Validated absolute path

    Raises:
        PathValidationError: If path is invalid or potentially malicious
    """
    # Resolve to absolute path (handles .., symlinks, etc.)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Cannot resolve path: {path}") from e

    # Ensure it's a regular file (not a directory, device, etc.)
    if not resolved.is_file():
        raise PathValidationError(f"Path is not a regular file: {resolved}")

    # Check for shell metacharacters that could enable command injection
    path_str = str(resolved)

    # Disallow null bytes (can truncate strings in C programs)
    if "\x00" in path_str:
        raise PathValidationError("Path contains null byte")

    # Disallow newlines (can inject additional commands in some contexts)
    if "\n" in path_str or "\r" in path_str:
        raise PathValidationError("Path contains newline characters")

    # Disallow common shell metacharacters as a defense-in-depth measure
    dangerous_patterns = [
        r"[;|&`$]",  # Command separators and execution
        r"^\s*-",  # Leading dash (could be interpreted as option)
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, path_str):
            raise PathValidationError(f"Path contains potentially dangerous characters: {path_str}")

    return resolved


# =============================================================================
# Models
# =============================================================================


class STTLanguage(str, Enum):
    """Supported languages for STT."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"
    AUTO = "auto"  # Auto-detect


class STTModel(str, Enum):
    """Available Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class STTConfig:
    """Configuration for STT engine."""

    model: STTModel = STTModel.BASE
    language: STTLanguage = STTLanguage.ENGLISH
    translate: bool = False  # Translate to English
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    no_timestamps: bool = True
    word_timestamps: bool = False


@dataclass
class WordTimestamp:
    """Timestamp for a single word."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    probability: float = 1.0


@dataclass
class STTSegment:
    """A segment of transcribed text."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    words: List[WordTimestamp] = field(default_factory=list)
    language: Optional[str] = None
    probability: float = 1.0


@dataclass
class STTResult:
    """Result from speech-to-text transcription."""

    text: str
    segments: List[STTSegment] = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0  # Audio duration in seconds
    processing_time: float = 0.0  # Time taken to process
    confidence: float = 1.0
    is_final: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def words(self) -> List[WordTimestamp]:
        """Get all words from all segments."""
        words = []
        for segment in self.segments:
            words.extend(segment.words)
        return words


# =============================================================================
# STT Engine Interface
# =============================================================================


class STTEngine(ABC):
    """
    Abstract base class for Speech-to-Text engines.

    Implementations can use Whisper, Whisper.cpp, or other STT systems.
    """

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self._callbacks: List[Callable[[STTResult], None]] = []

    @abstractmethod
    def transcribe(self, audio_data: bytes, format: AudioFormat = AudioFormat.PCM) -> STTResult:
        """
        Transcribe audio to text.

        Args:
            audio_data: Audio bytes to transcribe
            format: Audio format

        Returns:
            STTResult with transcribed text
        """
        pass

    @abstractmethod
    def transcribe_file(self, file_path: Union[str, Path]) -> STTResult:
        """
        Transcribe audio file to text.

        Args:
            file_path: Path to audio file

        Returns:
            STTResult with transcribed text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the STT engine is available."""
        pass

    def transcribe_buffer(self, buffer: AudioBuffer) -> STTResult:
        """
        Transcribe audio from buffer.

        Args:
            buffer: AudioBuffer containing audio data

        Returns:
            STTResult with transcribed text
        """
        wav_data = buffer.to_wav()
        return self.transcribe(wav_data, format=AudioFormat.WAV)

    def on_transcription(self, callback: Callable[[STTResult], None]) -> None:
        """Register callback for transcription results."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, result: STTResult) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"STT callback error: {e}")


# =============================================================================
# Mock STT Engine
# =============================================================================


class MockSTT(STTEngine):
    """
    Mock STT engine for testing.

    Returns configurable responses for testing purposes.
    """

    def __init__(self, config: Optional[STTConfig] = None):
        super().__init__(config)
        self._responses: List[str] = ["Hello, Agent OS."]
        self._response_index = 0
        self._latency: float = 0.1  # Simulated processing time

    def set_responses(self, responses: List[str]) -> None:
        """Set responses to return for transcription."""
        self._responses = responses
        self._response_index = 0

    def set_latency(self, seconds: float) -> None:
        """Set simulated processing latency."""
        self._latency = seconds

    def transcribe(self, audio_data: bytes, format: AudioFormat = AudioFormat.PCM) -> STTResult:
        """Return mock transcription."""
        start_time = time.time()

        # Simulate processing time
        time.sleep(self._latency)

        # Get next response
        text = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1

        result = STTResult(
            text=text,
            language=self.config.language.value,
            duration=len(audio_data) / 32000,  # Estimate
            processing_time=time.time() - start_time,
            confidence=0.95,
            segments=[STTSegment(text=text, start=0.0, end=1.0)],
        )

        self._notify_callbacks(result)
        return result

    def transcribe_file(self, file_path: Union[str, Path]) -> STTResult:
        """Mock file transcription."""
        with open(file_path, "rb") as f:
            audio_data = f.read()
        return self.transcribe(audio_data, format=AudioFormat.WAV)

    def is_available(self) -> bool:
        """Mock is always available."""
        return True


# =============================================================================
# Whisper.cpp STT Engine
# =============================================================================


class WhisperSTT(STTEngine):
    """
    STT engine using whisper.cpp.

    Requires whisper.cpp to be installed and accessible.

    Install:
        git clone https://github.com/ggerganov/whisper.cpp
        cd whisper.cpp
        make
        ./models/download-ggml-model.sh base
    """

    def __init__(
        self,
        config: Optional[STTConfig] = None,
        whisper_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__(config)
        self.whisper_path = whisper_path or self._find_whisper()
        self.model_path = model_path or self._find_model()

    def _find_whisper(self) -> Optional[str]:
        """Find whisper.cpp executable."""
        # Check common locations
        locations = [
            "/usr/local/bin/whisper",
            "/usr/bin/whisper",
            "~/.local/bin/whisper",
            "./whisper.cpp/main",
            "./whisper",
        ]

        for loc in locations:
            path = Path(loc).expanduser()
            if path.exists() and path.is_file():
                return str(path)

        # Try to find in PATH
        try:
            result = subprocess.run(
                ["which", "whisper"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def _find_model(self) -> Optional[str]:
        """Find Whisper model file."""
        model_name = f"ggml-{self.config.model.value}.bin"

        # Check common locations
        locations = [
            f"~/.cache/whisper/{model_name}",
            f"./whisper.cpp/models/{model_name}",
            f"./models/{model_name}",
            f"/usr/share/whisper/models/{model_name}",
        ]

        for loc in locations:
            path = Path(loc).expanduser()
            if path.exists():
                return str(path)

        return None

    def is_available(self) -> bool:
        """Check if whisper.cpp is available."""
        return self.whisper_path is not None and self.model_path is not None

    def transcribe(self, audio_data: bytes, format: AudioFormat = AudioFormat.PCM) -> STTResult:
        """Transcribe audio using whisper.cpp."""
        start_time = time.time()

        # Convert to WAV if needed
        if format == AudioFormat.PCM:
            audio_data = pcm_to_wav(audio_data)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            result = self.transcribe_file(temp_path)
            result.processing_time = time.time() - start_time
            return result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def transcribe_file(self, file_path: Union[str, Path]) -> STTResult:
        """Transcribe audio file using whisper.cpp."""
        if not self.is_available():
            return STTResult(
                text="",
                confidence=0.0,
                metadata={"error": "whisper.cpp not available"},
            )

        start_time = time.time()

        try:
            # Validate path before passing to subprocess to prevent command injection
            validated_path = validate_audio_path(Path(file_path))

            # Build command
            cmd = [
                self.whisper_path,
                "-m",
                self.model_path,
                "-f",
                str(validated_path),
                "-l",
                self.config.language.value,
                "--output-json",
                "-nt",  # No timestamps in output
            ]

            if self.config.translate:
                cmd.append("--translate")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                logger.error(f"Whisper error: {result.stderr}")
                return STTResult(
                    text="",
                    confidence=0.0,
                    metadata={"error": result.stderr},
                )

            # Parse output
            text = result.stdout.strip()

            stt_result = STTResult(
                text=text,
                language=self.config.language.value,
                processing_time=time.time() - start_time,
                confidence=0.9,
                segments=[STTSegment(text=text, start=0.0, end=0.0)],
            )

            self._notify_callbacks(stt_result)
            return stt_result

        except PathValidationError as e:
            logger.error(f"Path validation failed: {e}")
            return STTResult(
                text="",
                confidence=0.0,
                metadata={"error": f"Invalid audio path: {e}"},
            )
        except subprocess.TimeoutExpired:
            logger.error("Whisper transcription timed out")
            return STTResult(
                text="",
                confidence=0.0,
                metadata={"error": "timeout"},
            )
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return STTResult(
                text="",
                confidence=0.0,
                metadata={"error": str(e)},
            )


# =============================================================================
# OpenAI Whisper API STT
# =============================================================================


class WhisperAPISTT(STTEngine):
    """
    STT engine using OpenAI Whisper API.

    Requires: pip install openai
    Set OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        config: Optional[STTConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)
        self.api_key = api_key

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        import os

        return bool(self.api_key or os.getenv("OPENAI_API_KEY"))

    def transcribe(self, audio_data: bytes, format: AudioFormat = AudioFormat.PCM) -> STTResult:
        """Transcribe audio using OpenAI Whisper API."""
        start_time = time.time()

        try:
            import openai
        except ImportError:
            return STTResult(
                text="",
                confidence=0.0,
                metadata={"error": "openai package not installed"},
            )

        # Convert to WAV if needed
        if format == AudioFormat.PCM:
            audio_data = pcm_to_wav(audio_data)

        # Write to temp file (API requires file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            client = openai.OpenAI(api_key=self.api_key)

            with open(temp_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=(
                        self.config.language.value
                        if self.config.language != STTLanguage.AUTO
                        else None
                    ),
                )

            result = STTResult(
                text=response.text,
                language=self.config.language.value,
                processing_time=time.time() - start_time,
                confidence=0.95,
                segments=[STTSegment(text=response.text, start=0.0, end=0.0)],
            )

            self._notify_callbacks(result)
            return result

        except Exception as e:
            logger.error(f"OpenAI Whisper API error: {e}")
            return STTResult(
                text="",
                confidence=0.0,
                metadata={"error": str(e)},
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def transcribe_file(self, file_path: Union[str, Path]) -> STTResult:
        """Transcribe file using OpenAI Whisper API."""
        with open(file_path, "rb") as f:
            audio_data = f.read()

        # Determine format from extension
        ext = Path(file_path).suffix.lower()
        format_map = {
            ".wav": AudioFormat.WAV,
            ".mp3": AudioFormat.MP3,
            ".ogg": AudioFormat.OGG,
        }
        audio_format = format_map.get(ext, AudioFormat.WAV)

        return self.transcribe(audio_data, format=audio_format)


# =============================================================================
# Factory Function
# =============================================================================


def create_stt_engine(
    engine_type: str = "auto",
    config: Optional[STTConfig] = None,
    **kwargs,
) -> STTEngine:
    """
    Create an STT engine.

    Args:
        engine_type: Type of engine ("whisper", "whisper_api", "mock", "auto")
        config: STT configuration
        **kwargs: Additional engine-specific arguments

    Returns:
        STTEngine instance
    """
    if engine_type == "mock":
        return MockSTT(config)

    if engine_type == "whisper_api":
        return WhisperAPISTT(config, **kwargs)

    if engine_type == "whisper":
        return WhisperSTT(config, **kwargs)

    # Auto-detect best available engine
    if engine_type == "auto":
        # Try whisper.cpp first
        whisper = WhisperSTT(config)
        if whisper.is_available():
            logger.info("Using whisper.cpp for STT")
            return whisper

        # Try OpenAI API
        api = WhisperAPISTT(config)
        if api.is_available():
            logger.info("Using OpenAI Whisper API for STT")
            return api

        # Fall back to mock
        logger.warning("No STT engine available, using mock")
        return MockSTT(config)

    raise ValueError(f"Unknown STT engine type: {engine_type}")
