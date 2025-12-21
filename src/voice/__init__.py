"""
Agent OS Voice Interaction Module

Provides voice-based interaction with Agent OS including:
- Speech-to-Text (STT) via Whisper
- Text-to-Speech (TTS) via Piper/Coqui
- Wake word detection
- Voice assistant integration

Usage:
    from src.voice import VoiceAssistant, create_voice_assistant

    assistant = create_voice_assistant()
    assistant.start()  # Begins listening for wake word
"""

from .audio import (
    AudioCapture,
    AudioPlayer,
    AudioBuffer,
    AudioFormat,
    create_audio_capture,
    create_audio_player,
)
from .stt import (
    STTEngine,
    STTResult,
    WhisperSTT,
    create_stt_engine,
)
from .tts import (
    TTSEngine,
    TTSResult,
    PiperTTS,
    create_tts_engine,
)
from .wakeword import (
    WakeWordDetector,
    WakeWordEvent,
    create_wake_word_detector,
)
from .assistant import (
    VoiceAssistant,
    VoiceConfig,
    VoiceState,
    create_voice_assistant,
)

__all__ = [
    # Audio
    "AudioCapture",
    "AudioPlayer",
    "AudioBuffer",
    "AudioFormat",
    "create_audio_capture",
    "create_audio_player",
    # STT
    "STTEngine",
    "STTResult",
    "WhisperSTT",
    "create_stt_engine",
    # TTS
    "TTSEngine",
    "TTSResult",
    "PiperTTS",
    "create_tts_engine",
    # Wake Word
    "WakeWordDetector",
    "WakeWordEvent",
    "create_wake_word_detector",
    # Assistant
    "VoiceAssistant",
    "VoiceConfig",
    "VoiceState",
    "create_voice_assistant",
]
