"""
Text-to-Speech (TTS) Engine

Provides speech synthesis functionality using Piper or compatible engines.
"""

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

from .audio import AudioFormat, pcm_to_wav, wav_to_pcm

logger = logging.getLogger(__name__)


# =============================================================================
# Security: Path and Input Validation
# =============================================================================


class PathValidationError(Exception):
    """Raised when a path fails security validation."""

    pass


def validate_output_path(path: Path) -> Path:
    """
    Validate an output file path before writing.

    This prevents path traversal attacks by:
    1. Resolving to absolute path
    2. Checking the path doesn't contain shell metacharacters
    3. Ensuring the parent directory exists

    Args:
        path: Path to validate

    Returns:
        Validated absolute path

    Raises:
        PathValidationError: If path is invalid or potentially malicious
    """
    # Resolve to absolute path (handles .., symlinks, etc.)
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Cannot resolve path: {path}") from e

    # Ensure parent directory exists
    if not resolved.parent.exists():
        raise PathValidationError(f"Parent directory does not exist: {resolved.parent}")

    # Check for shell metacharacters that could enable command injection
    path_str = str(resolved)

    # Disallow null bytes
    if "\x00" in path_str:
        raise PathValidationError("Path contains null byte")

    # Disallow newlines
    if "\n" in path_str or "\r" in path_str:
        raise PathValidationError("Path contains newline characters")

    # Disallow common shell metacharacters as a defense-in-depth measure
    dangerous_patterns = [
        r"[;|&`$]",  # Command separators and execution
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, path_str):
            raise PathValidationError(f"Path contains potentially dangerous characters: {path_str}")

    return resolved


def sanitize_tts_text(text: str) -> str:
    """
    Sanitize text input for TTS to prevent potential issues.

    Removes control characters and other potentially problematic content.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text safe for TTS processing
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove other control characters except common whitespace
    # Keep: newline, carriage return, tab
    sanitized = "".join(
        char for char in text if char in "\n\r\t" or (ord(char) >= 32 and ord(char) != 127)
    )

    return sanitized


# =============================================================================
# Models
# =============================================================================


class TTSVoice(str, Enum):
    """Available TTS voices."""

    # English voices
    EN_US_AMY = "en_US-amy-medium"
    EN_US_DANNY = "en_US-danny-low"
    EN_US_KATHLEEN = "en_US-kathleen-low"
    EN_US_LESSAC = "en_US-lessac-medium"
    EN_US_LIBRITTS = "en_US-libritts-high"
    EN_GB_ALAN = "en_GB-alan-medium"
    EN_GB_ALBA = "en_GB-alba-medium"

    # Other languages
    ES_ES_DAVEFX = "es_ES-davefx-medium"
    FR_FR_SIWIS = "fr_FR-siwis-medium"
    DE_DE_THORSTEN = "de_DE-thorsten-medium"

    # Default
    DEFAULT = "en_US-lessac-medium"


@dataclass
class TTSConfig:
    """Configuration for TTS engine."""

    voice: TTSVoice = TTSVoice.DEFAULT
    speed: float = 1.0  # Speech rate (0.5 - 2.0)
    pitch: float = 1.0  # Pitch adjustment
    volume: float = 1.0  # Volume (0.0 - 1.0)
    sample_rate: int = 22050  # Output sample rate
    output_format: AudioFormat = AudioFormat.WAV


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""

    audio_data: bytes
    text: str
    format: AudioFormat
    duration: float  # Audio duration in seconds
    processing_time: float  # Time taken to synthesize
    sample_rate: int = 22050
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Check if result has audio."""
        return len(self.audio_data) == 0


# =============================================================================
# TTS Engine Interface
# =============================================================================


class TTSEngine(ABC):
    """
    Abstract base class for Text-to-Speech engines.

    Implementations can use Piper, Coqui, espeak, or other TTS systems.
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._callbacks: List[Callable[[TTSResult], None]] = []

    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """
        Synthesize text to speech.

        Args:
            text: Text to convert to speech

        Returns:
            TTSResult with audio data
        """
        pass

    @abstractmethod
    def synthesize_to_file(self, text: str, output_path: Union[str, Path]) -> bool:
        """
        Synthesize text and save to file.

        Args:
            text: Text to convert
            output_path: Path to save audio file

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS engine is available."""
        pass

    @abstractmethod
    def list_voices(self) -> List[str]:
        """List available voices."""
        pass

    def on_synthesis(self, callback: Callable[[TTSResult], None]) -> None:
        """Register callback for synthesis results."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, result: TTSResult) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"TTS callback error: {e}")


# =============================================================================
# Mock TTS Engine
# =============================================================================


class MockTTS(TTSEngine):
    """
    Mock TTS engine for testing.

    Generates silent audio or test patterns.
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        super().__init__(config)
        self._synthesized: List[str] = []

    def synthesize(self, text: str) -> TTSResult:
        """Generate mock audio for text."""
        start_time = time.time()
        self._synthesized.append(text)

        # Generate silence based on text length
        # Approximate: 150 words per minute, average word length 5
        words = len(text.split())
        duration = max(0.5, words / 2.5)  # At least 0.5 seconds

        # Generate silent WAV
        sample_rate = self.config.sample_rate
        samples = int(duration * sample_rate)
        silence = bytes(samples * 2)  # 16-bit = 2 bytes per sample

        audio_data = pcm_to_wav(
            silence,
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
        )

        result = TTSResult(
            audio_data=audio_data,
            text=text,
            format=AudioFormat.WAV,
            duration=duration,
            processing_time=time.time() - start_time,
            sample_rate=sample_rate,
        )

        self._notify_callbacks(result)
        return result

    def synthesize_to_file(self, text: str, output_path: Union[str, Path]) -> bool:
        """Save mock audio to file."""
        try:
            validated_path = validate_output_path(Path(output_path))
        except PathValidationError as e:
            logger.error(f"Output path validation failed: {e}")
            return False

        result = self.synthesize(text)
        with open(validated_path, "wb") as f:
            f.write(result.audio_data)
        return True

    def is_available(self) -> bool:
        """Mock is always available."""
        return True

    def list_voices(self) -> List[str]:
        """Return mock voice list."""
        return ["mock-voice-1", "mock-voice-2"]

    def get_synthesized_texts(self) -> List[str]:
        """Get list of synthesized texts for testing."""
        return self._synthesized

    def clear_synthesized(self) -> None:
        """Clear synthesized text history."""
        self._synthesized.clear()


# =============================================================================
# Piper TTS Engine
# =============================================================================


class PiperTTS(TTSEngine):
    """
    TTS engine using Piper (https://github.com/rhasspy/piper).

    Piper is a fast, local neural text-to-speech system.

    Install:
        pip install piper-tts
        # Or download binary from releases
    """

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        piper_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__(config)
        self.piper_path = piper_path or self._find_piper()
        self.model_path = model_path or self._find_model()

    def _find_piper(self) -> Optional[str]:
        """Find Piper executable."""
        locations = [
            "/usr/local/bin/piper",
            "/usr/bin/piper",
            "~/.local/bin/piper",
            "./piper/piper",
            "./piper",
        ]

        for loc in locations:
            path = Path(loc).expanduser()
            if path.exists() and path.is_file():
                return str(path)

        # Try to find in PATH
        try:
            result = subprocess.run(
                ["which", "piper"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def _find_model(self) -> Optional[str]:
        """Find Piper model file."""
        voice_name = self.config.voice.value

        locations = [
            f"~/.local/share/piper/voices/{voice_name}.onnx",
            f"./piper/voices/{voice_name}.onnx",
            f"/usr/share/piper/voices/{voice_name}.onnx",
        ]

        for loc in locations:
            path = Path(loc).expanduser()
            if path.exists():
                return str(path)

        return None

    def is_available(self) -> bool:
        """Check if Piper is available."""
        return self.piper_path is not None

    def list_voices(self) -> List[str]:
        """List available Piper voices."""
        if not self.is_available():
            return []

        voices = []
        voice_dirs = [
            Path("~/.local/share/piper/voices").expanduser(),
            Path("./piper/voices"),
            Path("/usr/share/piper/voices"),
        ]

        for vdir in voice_dirs:
            if vdir.exists():
                for model in vdir.glob("*.onnx"):
                    voices.append(model.stem)

        return list(set(voices))

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize text using Piper."""
        start_time = time.time()

        # Sanitize text input to remove control characters
        sanitized_text = sanitize_tts_text(text)

        if not self.is_available():
            return TTSResult(
                audio_data=b"",
                text=text,
                format=AudioFormat.WAV,
                duration=0.0,
                processing_time=0.0,
                metadata={"error": "Piper not available"},
            )

        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Build command
            cmd = [self.piper_path]

            if self.model_path:
                cmd.extend(["--model", self.model_path])

            cmd.extend(
                [
                    "--output_file",
                    output_path,
                ]
            )

            # Run Piper with sanitized text as stdin
            result = subprocess.run(
                cmd,
                input=sanitized_text,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"Piper error: {result.stderr}")
                return TTSResult(
                    audio_data=b"",
                    text=text,
                    format=AudioFormat.WAV,
                    duration=0.0,
                    processing_time=time.time() - start_time,
                    metadata={"error": result.stderr},
                )

            # Read output file
            with open(output_path, "rb") as f:
                audio_data = f.read()

            # Calculate duration from WAV
            _, sample_rate, _, _ = wav_to_pcm(audio_data)
            samples = len(audio_data) // 2  # Approximate
            duration = samples / sample_rate

            tts_result = TTSResult(
                audio_data=audio_data,
                text=text,
                format=AudioFormat.WAV,
                duration=duration,
                processing_time=time.time() - start_time,
                sample_rate=sample_rate,
            )

            self._notify_callbacks(tts_result)
            return tts_result

        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out")
            return TTSResult(
                audio_data=b"",
                text=text,
                format=AudioFormat.WAV,
                duration=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": "timeout"},
            )
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            return TTSResult(
                audio_data=b"",
                text=text,
                format=AudioFormat.WAV,
                duration=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
            )
        finally:
            Path(output_path).unlink(missing_ok=True)

    def synthesize_to_file(self, text: str, output_path: Union[str, Path]) -> bool:
        """Synthesize and save to file."""
        try:
            validated_path = validate_output_path(Path(output_path))
        except PathValidationError as e:
            logger.error(f"Output path validation failed: {e}")
            return False

        result = self.synthesize(text)
        if result.is_empty:
            return False

        with open(validated_path, "wb") as f:
            f.write(result.audio_data)
        return True


# =============================================================================
# espeak TTS Engine
# =============================================================================


class EspeakTTS(TTSEngine):
    """
    TTS engine using espeak-ng.

    espeak-ng is a compact, free, open-source speech synthesizer.

    Install:
        apt-get install espeak-ng  # Debian/Ubuntu
        brew install espeak        # macOS
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        super().__init__(config)

    def is_available(self) -> bool:
        """Check if espeak is available."""
        try:
            result = subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            try:
                result = subprocess.run(
                    ["espeak", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                return result.returncode == 0
            except Exception:
                return False

    def _get_espeak_cmd(self) -> str:
        """Get the espeak command name."""
        try:
            subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=2)
            return "espeak-ng"
        except Exception:
            return "espeak"

    def list_voices(self) -> List[str]:
        """List available espeak voices."""
        if not self.is_available():
            return []

        try:
            cmd = self._get_espeak_cmd()
            result = subprocess.run(
                [cmd, "--voices"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            voices = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 4:
                    voices.append(parts[3])  # Voice name

            return voices
        except Exception:
            return []

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize text using espeak."""
        start_time = time.time()

        # Sanitize text input to remove control characters
        sanitized_text = sanitize_tts_text(text)

        if not self.is_available():
            return TTSResult(
                audio_data=b"",
                text=text,
                format=AudioFormat.WAV,
                duration=0.0,
                processing_time=0.0,
                metadata={"error": "espeak not available"},
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            cmd = self._get_espeak_cmd()

            # Calculate speed (words per minute)
            # espeak default is 175 wpm, we scale based on config
            speed = int(175 * self.config.speed)

            result = subprocess.run(
                [
                    cmd,
                    "-w",
                    output_path,
                    "-s",
                    str(speed),
                    sanitized_text,
                ],
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                return TTSResult(
                    audio_data=b"",
                    text=text,
                    format=AudioFormat.WAV,
                    duration=0.0,
                    processing_time=time.time() - start_time,
                    metadata={"error": result.stderr.decode()},
                )

            with open(output_path, "rb") as f:
                audio_data = f.read()

            # Calculate duration
            try:
                _, sample_rate, _, _ = wav_to_pcm(audio_data)
                duration = len(audio_data) / (sample_rate * 2)
            except Exception:
                duration = 0.0
                sample_rate = 22050

            tts_result = TTSResult(
                audio_data=audio_data,
                text=text,
                format=AudioFormat.WAV,
                duration=duration,
                processing_time=time.time() - start_time,
                sample_rate=sample_rate,
            )

            self._notify_callbacks(tts_result)
            return tts_result

        except Exception as e:
            return TTSResult(
                audio_data=b"",
                text=text,
                format=AudioFormat.WAV,
                duration=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
            )
        finally:
            Path(output_path).unlink(missing_ok=True)

    def synthesize_to_file(self, text: str, output_path: Union[str, Path]) -> bool:
        """Synthesize and save to file."""
        try:
            validated_path = validate_output_path(Path(output_path))
        except PathValidationError as e:
            logger.error(f"Output path validation failed: {e}")
            return False

        result = self.synthesize(text)
        if result.is_empty:
            return False

        with open(validated_path, "wb") as f:
            f.write(result.audio_data)
        return True


# =============================================================================
# Factory Function
# =============================================================================


def create_tts_engine(
    engine_type: str = "auto",
    config: Optional[TTSConfig] = None,
    **kwargs,
) -> TTSEngine:
    """
    Create a TTS engine.

    Args:
        engine_type: Type of engine ("piper", "espeak", "mock", "auto")
        config: TTS configuration
        **kwargs: Additional engine-specific arguments

    Returns:
        TTSEngine instance
    """
    if engine_type == "mock":
        return MockTTS(config)

    if engine_type == "piper":
        return PiperTTS(config, **kwargs)

    if engine_type == "espeak":
        return EspeakTTS(config)

    # Auto-detect best available engine
    if engine_type == "auto":
        # Try Piper first (better quality)
        piper = PiperTTS(config)
        if piper.is_available():
            logger.info("Using Piper for TTS")
            return piper

        # Try espeak (widely available)
        espeak = EspeakTTS(config)
        if espeak.is_available():
            logger.info("Using espeak for TTS")
            return espeak

        # Fall back to mock
        logger.warning("No TTS engine available, using mock")
        return MockTTS(config)

    raise ValueError(f"Unknown TTS engine type: {engine_type}")
