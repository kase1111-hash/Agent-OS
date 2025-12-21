"""
Voice Assistant Integration

Provides a complete voice interaction system combining wake word detection,
speech recognition, and text-to-speech synthesis.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .audio import (
    AudioBuffer,
    AudioCapture,
    AudioFormat,
    AudioPlayer,
    create_audio_capture,
    create_audio_player,
)
from .stt import STTEngine, STTResult, create_stt_engine
from .tts import TTSEngine, TTSResult, create_tts_engine
from .wakeword import WakeWordDetector, WakeWordEvent, create_wake_word_detector

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class VoiceState(str, Enum):
    """Voice assistant state."""

    IDLE = "idle"  # Waiting for wake word
    LISTENING = "listening"  # Actively listening for command
    PROCESSING = "processing"  # Processing speech
    SPEAKING = "speaking"  # Playing TTS response
    ERROR = "error"  # Error state


class InteractionMode(str, Enum):
    """Voice interaction mode."""

    WAKE_WORD = "wake_word"  # Requires wake word to activate
    PUSH_TO_TALK = "push_to_talk"  # Manual activation
    CONTINUOUS = "continuous"  # Always listening


@dataclass
class VoiceConfig:
    """Configuration for voice assistant."""

    # Interaction settings
    mode: InteractionMode = InteractionMode.WAKE_WORD
    wake_words: List[str] = field(default_factory=lambda: ["agent"])
    listen_timeout: float = 10.0  # Max listening time in seconds
    silence_timeout: float = 2.0  # Silence before ending utterance

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1

    # Engine settings
    stt_engine: str = "auto"
    tts_engine: str = "auto"
    wake_word_engine: str = "auto"

    # Feedback settings
    play_activation_sound: bool = True
    play_deactivation_sound: bool = True
    confirmation_beep: bool = True


@dataclass
class VoiceInteraction:
    """Record of a voice interaction."""

    id: str
    timestamp: datetime
    wake_word: Optional[str]
    user_text: str
    response_text: Optional[str]
    stt_result: Optional[STTResult]
    tts_result: Optional[TTSResult]
    duration: float
    state: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceEvent:
    """Event emitted by voice assistant."""

    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Voice Assistant
# =============================================================================


class VoiceAssistant:
    """
    Complete voice assistant integrating all voice components.

    Handles the full voice interaction loop:
    1. Wake word detection (optional)
    2. Audio capture during speaking
    3. Speech-to-text transcription
    4. Response generation (via callback)
    5. Text-to-speech playback
    """

    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        audio_capture: Optional[AudioCapture] = None,
        audio_player: Optional[AudioPlayer] = None,
        stt_engine: Optional[STTEngine] = None,
        tts_engine: Optional[TTSEngine] = None,
        wake_word_detector: Optional[WakeWordDetector] = None,
    ):
        self.config = config or VoiceConfig()

        # Audio components
        self._audio_capture = audio_capture
        self._audio_player = audio_player

        # Speech engines
        self._stt = stt_engine
        self._tts = tts_engine
        self._wake_word = wake_word_detector

        # State
        self._state = VoiceState.IDLE
        self._is_running = False
        self._interaction_id = 0

        # Threading
        self._main_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._response_queue: queue.Queue = queue.Queue()

        # Callbacks
        self._state_callbacks: List[Callable[[VoiceState], None]] = []
        self._interaction_callbacks: List[Callable[[VoiceInteraction], None]] = []
        self._event_callbacks: List[Callable[[VoiceEvent], None]] = []
        self._response_handler: Optional[Callable[[str], str]] = None

        # Interaction history
        self._interactions: List[VoiceInteraction] = []

    def _initialize_components(self) -> bool:
        """Initialize all voice components."""
        try:
            from .audio import AudioConfig

            audio_config = AudioConfig(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
            )

            # Audio capture
            if not self._audio_capture:
                self._audio_capture = create_audio_capture(config=audio_config)

            # Audio player
            if not self._audio_player:
                self._audio_player = create_audio_player(config=audio_config)

            # STT engine
            if not self._stt:
                self._stt = create_stt_engine(self.config.stt_engine)

            # TTS engine
            if not self._tts:
                self._tts = create_tts_engine(self.config.tts_engine)

            # Wake word detector (if using wake word mode)
            if self.config.mode == InteractionMode.WAKE_WORD and not self._wake_word:
                from .wakeword import WakeWordConfig

                ww_config = WakeWordConfig(
                    keywords=self.config.wake_words,
                    sample_rate=self.config.sample_rate,
                )
                self._wake_word = create_wake_word_detector(
                    self.config.wake_word_engine,
                    config=ww_config,
                )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize voice components: {e}")
            return False

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_state_change(self, callback: Callable[[VoiceState], None]) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    def on_interaction(self, callback: Callable[[VoiceInteraction], None]) -> None:
        """Register callback for completed interactions."""
        self._interaction_callbacks.append(callback)

    def on_event(self, callback: Callable[[VoiceEvent], None]) -> None:
        """Register callback for voice events."""
        self._event_callbacks.append(callback)

    def set_response_handler(self, handler: Callable[[str], str]) -> None:
        """
        Set the response handler function.

        The handler receives the user's transcribed text and returns
        the response text to be spoken.
        """
        self._response_handler = handler

    def _set_state(self, state: VoiceState) -> None:
        """Update state and notify callbacks."""
        old_state = self._state
        self._state = state

        if old_state != state:
            logger.debug(f"Voice state: {old_state.value} -> {state.value}")
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

            self._emit_event("state_change", {"old": old_state.value, "new": state.value})

    def _emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit a voice event."""
        event = VoiceEvent(event_type=event_type, data=data or {})
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------

    @property
    def state(self) -> VoiceState:
        """Get current state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if assistant is running."""
        return self._is_running

    def start(self) -> bool:
        """
        Start the voice assistant.

        Returns:
            True if started successfully
        """
        if self._is_running:
            return True

        if not self._initialize_components():
            return False

        self._is_running = True
        self._set_state(VoiceState.IDLE)

        self._main_thread = threading.Thread(
            target=self._main_loop,
            daemon=True,
        )
        self._main_thread.start()

        self._emit_event("started")
        logger.info("Voice assistant started")
        return True

    def stop(self) -> None:
        """Stop the voice assistant."""
        if not self._is_running:
            return

        self._is_running = False
        self._set_state(VoiceState.IDLE)

        # Stop components
        if self._audio_capture:
            self._audio_capture.stop()
        if self._audio_player:
            self._audio_player.stop()
        if self._wake_word:
            self._wake_word.stop_listening()

        # Wait for thread
        if self._main_thread:
            self._main_thread.join(timeout=2.0)
            self._main_thread = None

        self._emit_event("stopped")
        logger.info("Voice assistant stopped")

    def activate(self) -> None:
        """
        Manually activate listening (for push-to-talk mode).
        """
        if self._state == VoiceState.IDLE:
            self._set_state(VoiceState.LISTENING)
            self._emit_event("activated", {"mode": "manual"})

    def deactivate(self) -> None:
        """
        Stop current interaction and return to idle.
        """
        if self._state in (VoiceState.LISTENING, VoiceState.PROCESSING):
            self._set_state(VoiceState.IDLE)
            self._emit_event("deactivated", {"mode": "manual"})

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def _main_loop(self) -> None:
        """Main voice assistant loop."""
        try:
            if self._audio_capture:
                self._audio_capture.start()

            while self._is_running:
                if self.config.mode == InteractionMode.WAKE_WORD:
                    self._wake_word_loop()
                elif self.config.mode == InteractionMode.CONTINUOUS:
                    self._continuous_loop()
                else:
                    # Push-to-talk: wait for manual activation
                    if self._state == VoiceState.LISTENING:
                        self._process_utterance()
                    else:
                        time.sleep(0.1)

        except Exception as e:
            logger.error(f"Voice assistant loop error: {e}")
            self._set_state(VoiceState.ERROR)
        finally:
            if self._audio_capture:
                self._audio_capture.stop()

    def _wake_word_loop(self) -> None:
        """Handle wake word detection mode."""
        if not self._wake_word or not self._audio_capture:
            time.sleep(0.1)
            return

        # Read audio
        audio_data = self._audio_capture.read(1024)
        if not audio_data:
            return

        if self._state == VoiceState.IDLE:
            # Check for wake word
            event = self._wake_word.detect(audio_data)
            if event:
                logger.info(f"Wake word detected: {event.keyword}")
                self._emit_event("wake_word", {"keyword": event.keyword})
                self._set_state(VoiceState.LISTENING)
                self._process_utterance(wake_word=event.keyword)
                self._set_state(VoiceState.IDLE)

    def _continuous_loop(self) -> None:
        """Handle continuous listening mode."""
        if self._state == VoiceState.IDLE:
            self._set_state(VoiceState.LISTENING)

        self._process_utterance()
        self._set_state(VoiceState.LISTENING)

    def _process_utterance(self, wake_word: Optional[str] = None) -> None:
        """
        Process a single voice utterance.

        1. Capture audio until silence
        2. Transcribe with STT
        3. Generate response
        4. Play response with TTS
        """
        start_time = time.time()
        self._interaction_id += 1
        interaction_id = f"voice_{self._interaction_id}"

        # Capture audio
        audio_buffer = AudioBuffer(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )

        if self.config.play_activation_sound:
            self._play_beep(frequency=880, duration=0.1)

        self._capture_speech(audio_buffer)

        if self.config.play_deactivation_sound:
            self._play_beep(frequency=440, duration=0.1)

        # Get audio data
        audio_data = audio_buffer.get_all()
        if not audio_data or len(audio_data) < 1000:
            logger.debug("No speech captured")
            return

        # Transcribe
        self._set_state(VoiceState.PROCESSING)
        stt_result = self._transcribe(audio_data)

        if not stt_result or not stt_result.text.strip():
            logger.debug("No transcription result")
            return

        logger.info(f"User said: {stt_result.text}")
        self._emit_event("transcription", {"text": stt_result.text})

        # Generate response
        response_text = self._generate_response(stt_result.text)

        tts_result = None
        if response_text:
            # Speak response
            self._set_state(VoiceState.SPEAKING)
            tts_result = self._speak(response_text)

        # Record interaction
        duration = time.time() - start_time
        interaction = VoiceInteraction(
            id=interaction_id,
            timestamp=datetime.now(),
            wake_word=wake_word,
            user_text=stt_result.text,
            response_text=response_text,
            stt_result=stt_result,
            tts_result=tts_result,
            duration=duration,
            state="completed",
        )
        self._interactions.append(interaction)

        for callback in self._interaction_callbacks:
            try:
                callback(interaction)
            except Exception as e:
                logger.error(f"Interaction callback error: {e}")

        self._emit_event("interaction_complete", {
            "id": interaction_id,
            "user_text": stt_result.text,
            "response_text": response_text,
            "duration": duration,
        })

    def _capture_speech(self, buffer: AudioBuffer) -> None:
        """Capture speech audio until silence is detected."""
        if not self._audio_capture:
            return

        silence_frames = 0
        max_silence_frames = int(self.config.silence_timeout * self.config.sample_rate / 1024)
        max_frames = int(self.config.listen_timeout * self.config.sample_rate / 1024)
        frame_count = 0

        while self._is_running and frame_count < max_frames:
            audio_data = self._audio_capture.read(1024)
            if not audio_data:
                continue

            frame_count += 1
            buffer.append_raw(audio_data)

            # Check for silence
            if self._is_silence(audio_data):
                silence_frames += 1
                if silence_frames >= max_silence_frames:
                    logger.debug("Silence detected, ending capture")
                    break
            else:
                silence_frames = 0

    def _is_silence(self, audio_data: bytes, threshold: float = 0.01) -> bool:
        """Check if audio is silence."""
        import struct

        if len(audio_data) < 2:
            return True

        samples = struct.unpack(f"<{len(audio_data) // 2}h", audio_data)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        normalized = rms / 32768.0

        return normalized < threshold

    def _transcribe(self, audio_data: bytes) -> Optional[STTResult]:
        """Transcribe audio using STT engine."""
        if not self._stt:
            return None

        try:
            return self._stt.transcribe(audio_data, AudioFormat.PCM)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _generate_response(self, text: str) -> Optional[str]:
        """Generate response text using handler."""
        if not self._response_handler:
            return None

        try:
            return self._response_handler(text)
        except Exception as e:
            logger.error(f"Response handler error: {e}")
            return None

    def _speak(self, text: str) -> Optional[TTSResult]:
        """Speak text using TTS engine."""
        if not self._tts or not self._audio_player:
            return None

        try:
            result = self._tts.synthesize(text)
            if result and result.audio_data:
                self._audio_player.play(result.audio_data)
                logger.info(f"Speaking: {text[:50]}...")
            return result
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    def _play_beep(self, frequency: float = 440, duration: float = 0.1) -> None:
        """Play a feedback beep."""
        if not self._audio_player:
            return

        try:
            import math
            import struct

            sample_rate = self.config.sample_rate
            num_samples = int(duration * sample_rate)

            samples = []
            for i in range(num_samples):
                t = i / sample_rate
                value = int(16384 * math.sin(2 * math.pi * frequency * t))
                samples.append(value)

            audio_data = struct.pack(f"<{len(samples)}h", *samples)
            self._audio_player.play(audio_data)

        except Exception as e:
            logger.debug(f"Failed to play beep: {e}")

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------

    def say(self, text: str) -> Optional[TTSResult]:
        """
        Speak text immediately.

        Args:
            text: Text to speak

        Returns:
            TTSResult if successful
        """
        old_state = self._state
        self._set_state(VoiceState.SPEAKING)

        try:
            return self._speak(text)
        finally:
            self._set_state(old_state)

    def listen(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Listen for speech and return transcription.

        Args:
            timeout: Max listening time (uses config default if None)

        Returns:
            Transcribed text or None
        """
        if not self._audio_capture or not self._stt:
            if not self._initialize_components():
                return None

        old_timeout = self.config.listen_timeout
        if timeout:
            self.config.listen_timeout = timeout

        try:
            old_state = self._state
            self._set_state(VoiceState.LISTENING)

            buffer = AudioBuffer(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
            )

            if self._audio_capture:
                self._audio_capture.start()
                self._capture_speech(buffer)
                self._audio_capture.stop()

            self._set_state(VoiceState.PROCESSING)
            audio_data = buffer.get_all()

            if audio_data:
                result = self._transcribe(audio_data)
                if result:
                    return result.text

            return None

        finally:
            self.config.listen_timeout = old_timeout
            self._set_state(old_state)

    def get_interactions(self) -> List[VoiceInteraction]:
        """Get list of past interactions."""
        return self._interactions.copy()

    def clear_interactions(self) -> None:
        """Clear interaction history."""
        self._interactions.clear()


# =============================================================================
# Factory Function
# =============================================================================


def create_voice_assistant(
    config: Optional[VoiceConfig] = None,
    response_handler: Optional[Callable[[str], str]] = None,
    **kwargs,
) -> VoiceAssistant:
    """
    Create a voice assistant.

    Args:
        config: Voice configuration
        response_handler: Function to generate responses
        **kwargs: Additional component overrides

    Returns:
        VoiceAssistant instance
    """
    assistant = VoiceAssistant(
        config=config,
        audio_capture=kwargs.get("audio_capture"),
        audio_player=kwargs.get("audio_player"),
        stt_engine=kwargs.get("stt_engine"),
        tts_engine=kwargs.get("tts_engine"),
        wake_word_detector=kwargs.get("wake_word_detector"),
    )

    if response_handler:
        assistant.set_response_handler(response_handler)

    return assistant
