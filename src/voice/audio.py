"""
Audio Capture and Playback

Provides audio input/output functionality for voice interaction.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import io
import struct
import wave

logger = logging.getLogger(__name__)


# =============================================================================
# Audio Format
# =============================================================================


class AudioFormat(Enum):
    """Supported audio formats."""

    WAV = "wav"
    PCM = "pcm"
    MP3 = "mp3"
    OGG = "ogg"


@dataclass
class AudioConfig:
    """Audio configuration."""

    sample_rate: int = 16000  # Hz (16kHz is standard for speech)
    channels: int = 1  # Mono
    sample_width: int = 2  # 16-bit audio
    chunk_size: int = 1024  # Samples per chunk
    format: AudioFormat = AudioFormat.PCM


# =============================================================================
# Audio Buffer
# =============================================================================


@dataclass
class AudioChunk:
    """A chunk of audio data."""

    data: bytes
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2

    @property
    def duration_ms(self) -> float:
        """Duration of this chunk in milliseconds."""
        samples = len(self.data) // (self.channels * self.sample_width)
        return (samples / self.sample_rate) * 1000

    @property
    def samples(self) -> int:
        """Number of samples in this chunk."""
        return len(self.data) // (self.channels * self.sample_width)


class AudioBuffer:
    """
    Thread-safe audio buffer for accumulating audio data.

    Used for collecting audio chunks for processing.
    """

    def __init__(
        self,
        max_duration_seconds: float = 30.0,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ):
        self._chunks: List[AudioChunk] = []
        self._lock = threading.Lock()
        self.max_duration = max_duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._bytes_per_second = sample_rate * channels * sample_width

    def append(self, chunk: AudioChunk) -> None:
        """Add audio chunk to buffer."""
        with self._lock:
            self._chunks.append(chunk)
            # Trim if exceeds max duration
            self._trim_to_max()

    def append_raw(self, data: bytes) -> None:
        """Add raw audio bytes to buffer."""
        chunk = AudioChunk(
            data=data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
        )
        self.append(chunk)

    def _trim_to_max(self) -> None:
        """Remove oldest chunks if buffer exceeds max duration."""
        max_bytes = int(self.max_duration * self._bytes_per_second)
        total_bytes = sum(len(c.data) for c in self._chunks)

        while total_bytes > max_bytes and len(self._chunks) > 1:
            removed = self._chunks.pop(0)
            total_bytes -= len(removed.data)

    def get_all(self) -> bytes:
        """Get all audio data as a single bytes object."""
        with self._lock:
            return b"".join(c.data for c in self._chunks)

    def get_last(self, duration_seconds: float) -> bytes:
        """Get the last N seconds of audio."""
        with self._lock:
            target_bytes = int(duration_seconds * self._bytes_per_second)
            result = []
            total = 0

            for chunk in reversed(self._chunks):
                if total >= target_bytes:
                    break
                result.insert(0, chunk.data)
                total += len(chunk.data)

            return b"".join(result)[-target_bytes:]

    def clear(self) -> None:
        """Clear all buffered audio."""
        with self._lock:
            self._chunks.clear()

    @property
    def duration_seconds(self) -> float:
        """Total duration of buffered audio."""
        with self._lock:
            total_bytes = sum(len(c.data) for c in self._chunks)
            return total_bytes / self._bytes_per_second

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._chunks) == 0

    def to_wav(self) -> bytes:
        """Convert buffer contents to WAV format."""
        audio_data = self.get_all()
        return pcm_to_wav(
            audio_data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
        )


# =============================================================================
# Audio Utilities
# =============================================================================


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Convert raw PCM audio to WAV format."""
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    return buffer.getvalue()


def wav_to_pcm(wav_data: bytes) -> tuple:
    """
    Convert WAV to raw PCM audio.

    Returns:
        Tuple of (pcm_data, sample_rate, channels, sample_width)
    """
    buffer = io.BytesIO(wav_data)

    with wave.open(buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        pcm_data = wav_file.readframes(wav_file.getnframes())

    return pcm_data, sample_rate, channels, sample_width


def calculate_rms(audio_data: bytes, sample_width: int = 2) -> float:
    """
    Calculate Root Mean Square (RMS) of audio data.

    Used for detecting audio level/volume.
    """
    if len(audio_data) == 0:
        return 0.0

    if sample_width == 2:
        # 16-bit audio
        fmt = f"<{len(audio_data) // 2}h"
        samples = struct.unpack(fmt, audio_data)
    elif sample_width == 1:
        # 8-bit audio
        samples = [b - 128 for b in audio_data]
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if len(samples) == 0:
        return 0.0

    sum_squares = sum(s * s for s in samples)
    mean_squares = sum_squares / len(samples)
    return mean_squares ** 0.5


def detect_silence(
    audio_data: bytes,
    threshold: float = 500.0,
    sample_width: int = 2,
) -> bool:
    """
    Detect if audio data is silence.

    Args:
        audio_data: Raw PCM audio bytes
        threshold: RMS threshold below which is considered silence
        sample_width: Bytes per sample

    Returns:
        True if audio is silence (below threshold)
    """
    rms = calculate_rms(audio_data, sample_width)
    return rms < threshold


# =============================================================================
# Audio Capture
# =============================================================================


class AudioCapture(ABC):
    """
    Abstract base class for audio capture.

    Implementations handle platform-specific audio input.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._running = False
        self._callbacks: List[Callable[[AudioChunk], None]] = []

    @abstractmethod
    def start(self) -> bool:
        """
        Start capturing audio.

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if audio capture is available on this system."""
        pass

    def on_audio(self, callback: Callable[[AudioChunk], None]) -> None:
        """Register callback for audio chunks."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, chunk: AudioChunk) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(chunk)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running


class MockAudioCapture(AudioCapture):
    """
    Mock audio capture for testing and systems without audio hardware.

    Generates silence or test patterns.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        super().__init__(config)
        self._thread: Optional[threading.Thread] = None
        self._generate_silence = True

    def start(self) -> bool:
        """Start generating mock audio."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._generate_audio, daemon=True)
        self._thread.start()
        logger.info("Mock audio capture started")
        return True

    def stop(self) -> None:
        """Stop generating mock audio."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Mock audio capture stopped")

    def is_available(self) -> bool:
        """Mock capture is always available."""
        return True

    def _generate_audio(self) -> None:
        """Generate mock audio chunks."""
        chunk_duration = self.config.chunk_size / self.config.sample_rate
        bytes_per_chunk = (
            self.config.chunk_size *
            self.config.channels *
            self.config.sample_width
        )

        while self._running:
            if self._generate_silence:
                # Generate silence
                data = bytes(bytes_per_chunk)
            else:
                # Generate low-level noise
                import random
                data = bytes([random.randint(127, 129) for _ in range(bytes_per_chunk)])

            chunk = AudioChunk(
                data=data,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                sample_width=self.config.sample_width,
            )
            self._notify_callbacks(chunk)
            time.sleep(chunk_duration)

    def inject_audio(self, data: bytes) -> None:
        """Inject audio data for testing."""
        chunk = AudioChunk(
            data=data,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            sample_width=self.config.sample_width,
        )
        self._notify_callbacks(chunk)


class PyAudioCapture(AudioCapture):
    """
    Audio capture using PyAudio.

    Requires: pip install pyaudio
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        super().__init__(config)
        self._pyaudio = None
        self._stream = None

    def is_available(self) -> bool:
        """Check if PyAudio is available."""
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            # Check for input devices
            has_input = pa.get_default_input_device_info() is not None
            pa.terminate()
            return has_input
        except Exception:
            return False

    def start(self) -> bool:
        """Start capturing audio via PyAudio."""
        if self._running:
            return True

        try:
            import pyaudio

            self._pyaudio = pyaudio.PyAudio()

            # Map sample width to PyAudio format
            format_map = {1: pyaudio.paInt8, 2: pyaudio.paInt16, 4: pyaudio.paInt32}
            audio_format = format_map.get(self.config.sample_width, pyaudio.paInt16)

            self._stream = self._pyaudio.open(
                format=audio_format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback,
            )

            self._running = True
            logger.info("PyAudio capture started")
            return True

        except Exception as e:
            logger.error(f"Failed to start PyAudio capture: {e}")
            return False

    def stop(self) -> None:
        """Stop PyAudio capture."""
        self._running = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None

        logger.info("PyAudio capture stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback."""
        import pyaudio

        if self._running and in_data:
            chunk = AudioChunk(
                data=in_data,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                sample_width=self.config.sample_width,
            )
            self._notify_callbacks(chunk)

        return (None, pyaudio.paContinue if self._running else pyaudio.paComplete)


# =============================================================================
# Audio Playback
# =============================================================================


class AudioPlayer(ABC):
    """
    Abstract base class for audio playback.

    Implementations handle platform-specific audio output.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    @abstractmethod
    def play(self, audio_data: bytes, format: AudioFormat = AudioFormat.PCM) -> bool:
        """
        Play audio data.

        Args:
            audio_data: Audio bytes to play
            format: Audio format

        Returns:
            True if playback started successfully
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop current playback."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if audio playback is available."""
        pass

    @abstractmethod
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        pass


class MockAudioPlayer(AudioPlayer):
    """
    Mock audio player for testing.

    Records played audio for verification.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        super().__init__(config)
        self._playing = False
        self._played_audio: List[bytes] = []

    def play(self, audio_data: bytes, format: AudioFormat = AudioFormat.PCM) -> bool:
        """Record audio instead of playing."""
        self._playing = True
        self._played_audio.append(audio_data)

        # Simulate playback duration
        if format == AudioFormat.PCM:
            duration = len(audio_data) / (
                self.config.sample_rate *
                self.config.channels *
                self.config.sample_width
            )
        else:
            duration = 0.1  # Estimate for other formats

        # Mark as done immediately for testing
        self._playing = False
        logger.debug(f"Mock played {len(audio_data)} bytes ({duration:.2f}s)")
        return True

    def stop(self) -> None:
        """Stop mock playback."""
        self._playing = False

    def is_available(self) -> bool:
        """Mock player is always available."""
        return True

    def is_playing(self) -> bool:
        """Check if mock playback is active."""
        return self._playing

    def get_played_audio(self) -> List[bytes]:
        """Get list of played audio for testing."""
        return self._played_audio

    def clear_played(self) -> None:
        """Clear played audio history."""
        self._played_audio.clear()


class PyAudioPlayer(AudioPlayer):
    """
    Audio playback using PyAudio.

    Requires: pip install pyaudio
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        super().__init__(config)
        self._pyaudio = None
        self._stream = None
        self._playing = False

    def is_available(self) -> bool:
        """Check if PyAudio is available for playback."""
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            has_output = pa.get_default_output_device_info() is not None
            pa.terminate()
            return has_output
        except Exception:
            return False

    def play(self, audio_data: bytes, format: AudioFormat = AudioFormat.PCM) -> bool:
        """Play audio via PyAudio."""
        try:
            import pyaudio

            # Convert to PCM if needed
            if format == AudioFormat.WAV:
                audio_data, sample_rate, channels, sample_width = wav_to_pcm(audio_data)
            else:
                sample_rate = self.config.sample_rate
                channels = self.config.channels
                sample_width = self.config.sample_width

            self._pyaudio = pyaudio.PyAudio()

            format_map = {1: pyaudio.paInt8, 2: pyaudio.paInt16, 4: pyaudio.paInt32}
            audio_format = format_map.get(sample_width, pyaudio.paInt16)

            self._stream = self._pyaudio.open(
                format=audio_format,
                channels=channels,
                rate=sample_rate,
                output=True,
            )

            self._playing = True
            self._stream.write(audio_data)
            self._playing = False

            self._stream.stop_stream()
            self._stream.close()
            self._pyaudio.terminate()

            return True

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            self._playing = False
            return False

    def stop(self) -> None:
        """Stop current playback."""
        self._playing = False
        if self._stream:
            self._stream.stop_stream()

    def is_playing(self) -> bool:
        """Check if audio is playing."""
        return self._playing


# =============================================================================
# Factory Functions
# =============================================================================


def create_audio_capture(
    use_mock: bool = False,
    config: Optional[AudioConfig] = None,
) -> AudioCapture:
    """
    Create an audio capture instance.

    Args:
        use_mock: Force use of mock capture
        config: Audio configuration

    Returns:
        AudioCapture instance
    """
    if use_mock:
        return MockAudioCapture(config)

    # Try PyAudio first
    pyaudio_capture = PyAudioCapture(config)
    if pyaudio_capture.is_available():
        return pyaudio_capture

    # Fall back to mock
    logger.warning("No audio hardware available, using mock capture")
    return MockAudioCapture(config)


def create_audio_player(
    use_mock: bool = False,
    config: Optional[AudioConfig] = None,
) -> AudioPlayer:
    """
    Create an audio player instance.

    Args:
        use_mock: Force use of mock player
        config: Audio configuration

    Returns:
        AudioPlayer instance
    """
    if use_mock:
        return MockAudioPlayer(config)

    # Try PyAudio first
    pyaudio_player = PyAudioPlayer(config)
    if pyaudio_player.is_available():
        return pyaudio_player

    # Fall back to mock
    logger.warning("No audio output available, using mock player")
    return MockAudioPlayer(config)
