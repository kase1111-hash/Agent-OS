"""
Tests for the Voice Interaction module (UC-018).

Tests cover:
- Audio capture and playback
- Speech-to-Text engines
- Text-to-Speech engines
- Wake word detection
- Voice assistant integration
"""

import struct
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Audio Module Tests
# =============================================================================


class TestAudioFormat:
    """Tests for AudioFormat enum."""

    def test_audio_formats(self):
        """Test audio format values."""
        from src.voice.audio import AudioFormat

        assert AudioFormat.PCM.value == "pcm"
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.OGG.value == "ogg"


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_audio_chunk_creation(self):
        """Test creating audio chunk."""
        from src.voice.audio import AudioChunk

        data = b"\x00\x01\x02\x03"
        chunk = AudioChunk(
            data=data,
            sample_rate=16000,
            channels=1,
            sample_width=2,
        )

        assert chunk.data == data
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.sample_width == 2

    def test_audio_chunk_duration(self):
        """Test audio chunk duration calculation."""
        from src.voice.audio import AudioChunk

        # 16000 samples/sec * 2 bytes * 1 channel = 32000 bytes/sec
        # 3200 bytes = 0.1 seconds = 100 ms
        data = bytes(3200)
        chunk = AudioChunk(
            data=data,
            sample_rate=16000,
            channels=1,
            sample_width=2,
        )

        assert abs(chunk.duration_ms - 100.0) < 1.0


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_buffer_creation(self):
        """Test creating audio buffer."""
        from src.voice.audio import AudioBuffer

        buffer = AudioBuffer(sample_rate=16000, channels=1)
        assert buffer.sample_rate == 16000
        assert buffer.channels == 1
        assert buffer.is_empty

    def test_buffer_append(self):
        """Test appending to buffer."""
        from src.voice.audio import AudioBuffer, AudioChunk

        buffer = AudioBuffer()
        chunk = AudioChunk(
            data=b"\x00\x01\x02\x03",
            sample_rate=16000,
            channels=1,
            sample_width=2,
        )

        buffer.append(chunk)
        assert not buffer.is_empty

    def test_buffer_append_raw(self):
        """Test appending raw data."""
        from src.voice.audio import AudioBuffer

        buffer = AudioBuffer()
        buffer.append_raw(b"\x00\x01\x02\x03")

        assert len(buffer.get_all()) == 4

    def test_buffer_get_all(self):
        """Test getting all buffer data."""
        from src.voice.audio import AudioBuffer

        buffer = AudioBuffer()
        buffer.append_raw(b"\x00\x01")
        buffer.append_raw(b"\x02\x03")

        data = buffer.get_all()
        assert data == b"\x00\x01\x02\x03"

    def test_buffer_clear(self):
        """Test clearing buffer."""
        from src.voice.audio import AudioBuffer

        buffer = AudioBuffer()
        buffer.append_raw(b"\x00\x01\x02\x03")
        buffer.clear()

        assert buffer.is_empty
        assert buffer.get_all() == b""

    def test_buffer_to_wav(self):
        """Test converting buffer to WAV."""
        from src.voice.audio import AudioBuffer

        buffer = AudioBuffer(sample_rate=16000, channels=1)
        # Add some audio data
        samples = [0, 1000, 2000, 1000, 0, -1000, -2000, -1000]
        audio_data = struct.pack(f"<{len(samples)}h", *samples)
        buffer.append_raw(audio_data)

        wav_data = buffer.to_wav()
        assert wav_data[:4] == b"RIFF"
        assert wav_data[8:12] == b"WAVE"

    def test_buffer_thread_safety(self):
        """Test buffer is thread-safe."""
        from src.voice.audio import AudioBuffer

        buffer = AudioBuffer()
        errors = []

        def writer():
            try:
                for _ in range(100):
                    buffer.append_raw(b"\x00\x01")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    buffer.get_all()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        threads.extend([threading.Thread(target=reader) for _ in range(5)])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestMockAudioCapture:
    """Tests for MockAudioCapture."""

    def test_mock_capture_creation(self):
        """Test creating mock capture."""
        from src.voice.audio import MockAudioCapture

        capture = MockAudioCapture()
        assert capture.is_available()

    def test_mock_capture_start_stop(self):
        """Test starting and stopping capture."""
        from src.voice.audio import MockAudioCapture

        capture = MockAudioCapture()
        assert capture.start()
        capture.stop()
        assert not capture.is_running

    @pytest.mark.slow
    def test_mock_capture_callbacks(self):
        """Test audio callbacks."""
        from src.voice.audio import MockAudioCapture
        import time

        capture = MockAudioCapture()
        chunks = []
        capture.on_audio(lambda c: chunks.append(c))

        capture.start()
        time.sleep(0.15)  # Wait for at least one chunk
        capture.stop()

        assert len(chunks) > 0

    def test_mock_capture_inject(self):
        """Test injecting test audio."""
        from src.voice.audio import MockAudioCapture

        capture = MockAudioCapture()
        test_data = b"\x00\x01\x02\x03"
        chunks = []
        capture.on_audio(lambda c: chunks.append(c))

        capture.inject_audio(test_data)

        assert len(chunks) == 1
        assert chunks[0].data == test_data


class TestMockAudioPlayer:
    """Tests for MockAudioPlayer."""

    def test_mock_player_creation(self):
        """Test creating mock player."""
        from src.voice.audio import MockAudioPlayer

        player = MockAudioPlayer()
        assert player.is_available()

    def test_mock_player_play(self):
        """Test playing audio."""
        from src.voice.audio import MockAudioPlayer

        player = MockAudioPlayer()
        assert player.play(b"\x00\x01\x02\x03")

    def test_mock_player_history(self):
        """Test played audio history."""
        from src.voice.audio import MockAudioPlayer

        player = MockAudioPlayer()
        player.play(b"audio1")
        player.play(b"audio2")

        history = player.get_played_audio()
        assert len(history) == 2


class TestAudioFactories:
    """Tests for audio factory functions."""

    def test_create_audio_capture_mock(self):
        """Test creating mock audio capture."""
        from src.voice.audio import create_audio_capture

        capture = create_audio_capture(use_mock=True)
        assert capture.is_available()

    def test_create_audio_player_mock(self):
        """Test creating mock audio player."""
        from src.voice.audio import create_audio_player

        player = create_audio_player(use_mock=True)
        assert player.is_available()


class TestAudioConversion:
    """Tests for audio format conversion."""

    def test_pcm_to_wav(self):
        """Test PCM to WAV conversion."""
        from src.voice.audio import pcm_to_wav

        # Generate simple PCM data
        samples = [0, 1000, 2000, 1000, 0, -1000, -2000, -1000]
        pcm_data = struct.pack(f"<{len(samples)}h", *samples)

        wav_data = pcm_to_wav(pcm_data, sample_rate=16000)

        assert wav_data[:4] == b"RIFF"
        assert wav_data[8:12] == b"WAVE"

    def test_wav_to_pcm(self):
        """Test WAV to PCM conversion."""
        from src.voice.audio import pcm_to_wav, wav_to_pcm

        original_pcm = struct.pack("<8h", 0, 1000, 2000, 1000, 0, -1000, -2000, -1000)
        wav_data = pcm_to_wav(original_pcm, sample_rate=16000)

        pcm_data, sample_rate, channels, sample_width = wav_to_pcm(wav_data)

        assert sample_rate == 16000
        assert channels == 1
        assert sample_width == 2

    def test_calculate_rms(self):
        """Test RMS calculation."""
        from src.voice.audio import calculate_rms

        # Silence should have 0 RMS
        silence = bytes(100)
        assert calculate_rms(silence) == 0.0

        # Loud signal should have high RMS
        loud = struct.pack("<50h", *([20000] * 50))
        assert calculate_rms(loud) > 0.5

    def test_detect_silence(self):
        """Test silence detection."""
        from src.voice.audio import detect_silence

        silence = bytes(100)
        assert detect_silence(silence)

        loud = struct.pack("<50h", *([20000] * 50))
        assert not detect_silence(loud)


# =============================================================================
# STT Module Tests
# =============================================================================


class TestSTTResult:
    """Tests for STTResult dataclass."""

    def test_stt_result_creation(self):
        """Test creating STT result."""
        from src.voice.stt import STTResult

        result = STTResult(
            text="hello world",
            segments=[],
            language="en",
            confidence=0.95,
            processing_time=0.5,
        )

        assert result.text == "hello world"
        assert result.confidence == 0.95
        assert result.language == "en"


class TestMockSTT:
    """Tests for MockSTT engine."""

    def test_mock_stt_creation(self):
        """Test creating mock STT."""
        from src.voice.stt import MockSTT

        stt = MockSTT()
        assert stt.is_available()

    def test_mock_stt_transcribe(self):
        """Test mock transcription."""
        from src.voice.audio import AudioFormat
        from src.voice.stt import MockSTT

        stt = MockSTT()
        result = stt.transcribe(b"\x00\x01\x02\x03", AudioFormat.PCM)

        assert result is not None
        assert result.text == "Hello, Agent OS."

    def test_mock_stt_set_responses(self):
        """Test setting mock responses."""
        from src.voice.audio import AudioFormat
        from src.voice.stt import MockSTT

        stt = MockSTT()
        stt.set_responses(["custom response"])

        result = stt.transcribe(b"audio", AudioFormat.PCM)
        assert result.text == "custom response"

    def test_mock_stt_callbacks(self):
        """Test STT callbacks."""
        from src.voice.audio import AudioFormat
        from src.voice.stt import MockSTT

        stt = MockSTT()
        results = []
        stt.on_transcription(lambda r: results.append(r))

        stt.transcribe(b"audio", AudioFormat.PCM)
        assert len(results) == 1


class TestWhisperSTT:
    """Tests for WhisperSTT engine."""

    def test_whisper_stt_creation(self):
        """Test creating Whisper STT."""
        from src.voice.stt import WhisperSTT

        stt = WhisperSTT()
        # Will be unavailable without whisper.cpp installed
        assert isinstance(stt.is_available(), bool)

    def test_whisper_stt_unavailable_transcribe(self):
        """Test transcription when unavailable."""
        from src.voice.audio import AudioFormat
        from src.voice.stt import WhisperSTT

        stt = WhisperSTT()
        if not stt.is_available():
            result = stt.transcribe(b"audio", AudioFormat.PCM)
            assert result.text == ""
            assert "error" in result.metadata


class TestSTTFactory:
    """Tests for STT factory function."""

    def test_create_stt_mock(self):
        """Test creating mock STT."""
        from src.voice.stt import create_stt_engine

        stt = create_stt_engine("mock")
        assert stt.is_available()

    def test_create_stt_auto(self):
        """Test auto STT creation."""
        from src.voice.stt import create_stt_engine

        stt = create_stt_engine("auto")
        assert stt is not None

    def test_create_stt_invalid(self):
        """Test invalid STT type."""
        from src.voice.stt import create_stt_engine

        with pytest.raises(ValueError):
            create_stt_engine("invalid_engine")


# =============================================================================
# TTS Module Tests
# =============================================================================


class TestTTSResult:
    """Tests for TTSResult dataclass."""

    def test_tts_result_creation(self):
        """Test creating TTS result."""
        from src.voice.audio import AudioFormat
        from src.voice.tts import TTSResult

        result = TTSResult(
            audio_data=b"audio",
            text="hello",
            format=AudioFormat.WAV,
            duration=1.0,
            processing_time=0.1,
        )

        assert result.text == "hello"
        assert result.duration == 1.0
        assert not result.is_empty

    def test_tts_result_empty(self):
        """Test empty TTS result."""
        from src.voice.audio import AudioFormat
        from src.voice.tts import TTSResult

        result = TTSResult(
            audio_data=b"",
            text="hello",
            format=AudioFormat.WAV,
            duration=0.0,
            processing_time=0.0,
        )

        assert result.is_empty


class TestTTSVoice:
    """Tests for TTSVoice enum."""

    def test_tts_voices(self):
        """Test TTS voice values."""
        from src.voice.tts import TTSVoice

        assert "lessac" in TTSVoice.DEFAULT.value
        assert TTSVoice.EN_US_AMY.value.startswith("en_US")


class TestMockTTS:
    """Tests for MockTTS engine."""

    def test_mock_tts_creation(self):
        """Test creating mock TTS."""
        from src.voice.tts import MockTTS

        tts = MockTTS()
        assert tts.is_available()

    def test_mock_tts_synthesize(self):
        """Test mock synthesis."""
        from src.voice.tts import MockTTS

        tts = MockTTS()
        result = tts.synthesize("hello world")

        assert result is not None
        assert not result.is_empty
        assert result.text == "hello world"
        assert result.duration > 0

    def test_mock_tts_synthesize_to_file(self):
        """Test synthesis to file."""
        from src.voice.tts import MockTTS

        tts = MockTTS()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            result = tts.synthesize_to_file("test", output_path)
            assert result is True
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_mock_tts_list_voices(self):
        """Test listing voices."""
        from src.voice.tts import MockTTS

        tts = MockTTS()
        voices = tts.list_voices()

        assert isinstance(voices, list)
        assert len(voices) > 0

    def test_mock_tts_callbacks(self):
        """Test TTS callbacks."""
        from src.voice.tts import MockTTS

        tts = MockTTS()
        results = []
        tts.on_synthesis(lambda r: results.append(r))

        tts.synthesize("test")
        assert len(results) == 1

    def test_mock_tts_synthesized_history(self):
        """Test synthesized text history."""
        from src.voice.tts import MockTTS

        tts = MockTTS()
        tts.synthesize("hello")
        tts.synthesize("world")

        history = tts.get_synthesized_texts()
        assert len(history) == 2
        assert "hello" in history
        assert "world" in history


class TestPiperTTS:
    """Tests for PiperTTS engine."""

    def test_piper_tts_creation(self):
        """Test creating Piper TTS."""
        from src.voice.tts import PiperTTS

        tts = PiperTTS()
        # Will be unavailable without piper installed
        assert isinstance(tts.is_available(), bool)

    def test_piper_tts_unavailable_synthesize(self):
        """Test synthesis when unavailable."""
        from src.voice.tts import PiperTTS

        tts = PiperTTS()
        if not tts.is_available():
            result = tts.synthesize("test")
            assert result.is_empty
            assert "error" in result.metadata


class TestEspeakTTS:
    """Tests for EspeakTTS engine."""

    def test_espeak_tts_creation(self):
        """Test creating espeak TTS."""
        from src.voice.tts import EspeakTTS

        tts = EspeakTTS()
        assert isinstance(tts.is_available(), bool)


class TestTTSFactory:
    """Tests for TTS factory function."""

    def test_create_tts_mock(self):
        """Test creating mock TTS."""
        from src.voice.tts import create_tts_engine

        tts = create_tts_engine("mock")
        assert tts.is_available()

    def test_create_tts_auto(self):
        """Test auto TTS creation."""
        from src.voice.tts import create_tts_engine

        tts = create_tts_engine("auto")
        assert tts is not None

    def test_create_tts_invalid(self):
        """Test invalid TTS type."""
        from src.voice.tts import create_tts_engine

        with pytest.raises(ValueError):
            create_tts_engine("invalid_engine")


# =============================================================================
# Wake Word Module Tests
# =============================================================================


class TestWakeWordEvent:
    """Tests for WakeWordEvent dataclass."""

    def test_wake_word_event_creation(self):
        """Test creating wake word event."""
        from src.voice.wakeword import WakeWordEvent

        event = WakeWordEvent(
            keyword="agent",
            confidence=0.95,
        )

        assert event.keyword == "agent"
        assert event.confidence == 0.95
        assert event.timestamp is not None


class TestWakeWordConfig:
    """Tests for WakeWordConfig dataclass."""

    def test_wake_word_config_defaults(self):
        """Test default config values."""
        from src.voice.wakeword import WakeWordConfig

        config = WakeWordConfig()

        assert "agent" in config.keywords
        assert 0 <= config.sensitivity <= 1
        assert config.sample_rate == 16000


class TestMockWakeWordDetector:
    """Tests for MockWakeWordDetector."""

    def test_mock_detector_creation(self):
        """Test creating mock detector."""
        from src.voice.wakeword import MockWakeWordDetector

        detector = MockWakeWordDetector()
        assert detector.is_available()

    def test_mock_detector_no_detection(self):
        """Test no detection by default."""
        from src.voice.wakeword import MockWakeWordDetector

        detector = MockWakeWordDetector()
        result = detector.detect(b"\x00\x01\x02\x03")

        assert result is None

    def test_mock_detector_trigger(self):
        """Test triggered detection."""
        from src.voice.wakeword import MockWakeWordDetector

        detector = MockWakeWordDetector()
        detector.trigger_detection("agent")

        result = detector.detect(b"\x00\x01\x02\x03")

        assert result is not None
        assert result.keyword == "agent"
        assert result.confidence > 0

    def test_mock_detector_callbacks(self):
        """Test wake word callbacks."""
        from src.voice.wakeword import MockWakeWordDetector

        detector = MockWakeWordDetector()
        events = []
        detector.on_wake_word(lambda e: events.append(e))

        detector.trigger_detection()
        event = detector.detect(b"audio")

        if event:
            detector._notify_callbacks(event)

        assert len(events) == 1

    def test_mock_detector_count(self):
        """Test detection count."""
        from src.voice.wakeword import MockWakeWordDetector

        detector = MockWakeWordDetector()

        detector.trigger_detection()
        detector.detect(b"audio")

        detector.trigger_detection()
        detector.detect(b"audio")

        assert detector.get_detection_count() == 2

    def test_mock_detector_reset(self):
        """Test resetting detector."""
        from src.voice.wakeword import MockWakeWordDetector

        detector = MockWakeWordDetector()
        detector.trigger_detection()
        detector.detect(b"audio")

        detector.reset()

        assert detector.get_detection_count() == 0


class TestEnergyWakeWordDetector:
    """Tests for EnergyWakeWordDetector."""

    def test_energy_detector_creation(self):
        """Test creating energy detector."""
        from src.voice.wakeword import EnergyWakeWordDetector

        detector = EnergyWakeWordDetector()
        assert detector.is_available()

    def test_energy_detector_silence(self):
        """Test no detection on silence."""
        from src.voice.wakeword import EnergyWakeWordDetector

        detector = EnergyWakeWordDetector()
        result = detector.detect(bytes(512))

        assert result is None

    def test_energy_detector_loud(self):
        """Test detection on loud audio."""
        from src.voice.wakeword import EnergyWakeWordDetector, WakeWordConfig

        config = WakeWordConfig(frame_length=512)
        detector = EnergyWakeWordDetector(
            config=config,
            energy_threshold=0.1,
            min_duration=0.05,  # Very short for testing
        )

        # Generate loud audio
        loud = struct.pack("<256h", *([20000] * 256))

        # Need multiple frames to trigger
        for _ in range(10):
            result = detector.detect(loud)
            if result:
                break

        # Should eventually detect
        assert result is not None or detector._active_frames > 0


class TestWakeWordFactory:
    """Tests for wake word factory function."""

    def test_create_detector_mock(self):
        """Test creating mock detector."""
        from src.voice.wakeword import create_wake_word_detector

        detector = create_wake_word_detector("mock")
        assert detector.is_available()

    def test_create_detector_energy(self):
        """Test creating energy detector."""
        from src.voice.wakeword import create_wake_word_detector

        detector = create_wake_word_detector("energy")
        assert detector.is_available()

    def test_create_detector_auto(self):
        """Test auto detector creation."""
        from src.voice.wakeword import create_wake_word_detector

        detector = create_wake_word_detector("auto")
        assert detector is not None

    def test_create_detector_invalid(self):
        """Test invalid detector type."""
        from src.voice.wakeword import create_wake_word_detector

        with pytest.raises(ValueError):
            create_wake_word_detector("invalid_detector")


# =============================================================================
# Voice Assistant Tests
# =============================================================================


class TestVoiceState:
    """Tests for VoiceState enum."""

    def test_voice_states(self):
        """Test voice state values."""
        from src.voice.assistant import VoiceState

        assert VoiceState.IDLE.value == "idle"
        assert VoiceState.LISTENING.value == "listening"
        assert VoiceState.PROCESSING.value == "processing"
        assert VoiceState.SPEAKING.value == "speaking"


class TestInteractionMode:
    """Tests for InteractionMode enum."""

    def test_interaction_modes(self):
        """Test interaction mode values."""
        from src.voice.assistant import InteractionMode

        assert InteractionMode.WAKE_WORD.value == "wake_word"
        assert InteractionMode.PUSH_TO_TALK.value == "push_to_talk"
        assert InteractionMode.CONTINUOUS.value == "continuous"


class TestVoiceConfig:
    """Tests for VoiceConfig dataclass."""

    def test_voice_config_defaults(self):
        """Test default config values."""
        from src.voice.assistant import InteractionMode, VoiceConfig

        config = VoiceConfig()

        assert config.mode == InteractionMode.WAKE_WORD
        assert "agent" in config.wake_words
        assert config.sample_rate == 16000


class TestVoiceInteraction:
    """Tests for VoiceInteraction dataclass."""

    def test_voice_interaction_creation(self):
        """Test creating voice interaction."""
        from src.voice.assistant import VoiceInteraction

        interaction = VoiceInteraction(
            id="voice_1",
            timestamp=datetime.now(),
            wake_word="agent",
            user_text="hello",
            response_text="hi there",
            stt_result=None,
            tts_result=None,
            duration=1.5,
            state="completed",
        )

        assert interaction.id == "voice_1"
        assert interaction.user_text == "hello"
        assert interaction.duration == 1.5


class TestVoiceAssistant:
    """Tests for VoiceAssistant class."""

    def test_assistant_creation(self):
        """Test creating voice assistant."""
        from src.voice.assistant import VoiceAssistant

        assistant = VoiceAssistant()
        assert assistant.state.value == "idle"
        assert not assistant.is_running

    def test_assistant_with_config(self):
        """Test assistant with custom config."""
        from src.voice.assistant import InteractionMode, VoiceAssistant, VoiceConfig

        config = VoiceConfig(
            mode=InteractionMode.PUSH_TO_TALK,
            wake_words=["computer"],
        )
        assistant = VoiceAssistant(config=config)

        assert assistant.config.mode == InteractionMode.PUSH_TO_TALK

    def test_assistant_with_mock_components(self):
        """Test assistant with mock components."""
        from src.voice.assistant import VoiceAssistant
        from src.voice.audio import MockAudioCapture, MockAudioPlayer
        from src.voice.stt import MockSTT
        from src.voice.tts import MockTTS

        assistant = VoiceAssistant(
            audio_capture=MockAudioCapture(),
            audio_player=MockAudioPlayer(),
            stt_engine=MockSTT(),
            tts_engine=MockTTS(),
        )

        assert assistant is not None

    def test_assistant_state_callbacks(self):
        """Test state change callbacks."""
        from src.voice.assistant import VoiceAssistant, VoiceState

        assistant = VoiceAssistant()
        states = []
        assistant.on_state_change(lambda s: states.append(s))

        assistant._set_state(VoiceState.LISTENING)
        assistant._set_state(VoiceState.PROCESSING)

        assert len(states) == 2
        assert states[0] == VoiceState.LISTENING

    def test_assistant_event_callbacks(self):
        """Test event callbacks."""
        from src.voice.assistant import VoiceAssistant

        assistant = VoiceAssistant()
        events = []
        assistant.on_event(lambda e: events.append(e))

        assistant._emit_event("test_event", {"key": "value"})

        assert len(events) == 1
        assert events[0].event_type == "test_event"

    def test_assistant_response_handler(self):
        """Test setting response handler."""
        from src.voice.assistant import VoiceAssistant

        assistant = VoiceAssistant()

        def handler(text: str) -> str:
            return f"You said: {text}"

        assistant.set_response_handler(handler)
        response = assistant._generate_response("hello")

        assert response == "You said: hello"

    def test_assistant_say(self):
        """Test say method."""
        from src.voice.assistant import VoiceAssistant
        from src.voice.audio import MockAudioPlayer
        from src.voice.tts import MockTTS

        player = MockAudioPlayer()
        tts = MockTTS()

        assistant = VoiceAssistant(
            audio_player=player,
            tts_engine=tts,
        )
        assistant._initialize_components()

        result = assistant.say("hello world")

        assert result is not None
        assert len(player.get_played_audio()) == 1

    def test_assistant_interactions(self):
        """Test interaction history."""
        from src.voice.assistant import VoiceAssistant, VoiceInteraction
        from datetime import datetime

        assistant = VoiceAssistant()

        # Add mock interaction
        interaction = VoiceInteraction(
            id="test_1",
            timestamp=datetime.now(),
            wake_word=None,
            user_text="test",
            response_text="response",
            stt_result=None,
            tts_result=None,
            duration=1.0,
            state="completed",
        )
        assistant._interactions.append(interaction)

        history = assistant.get_interactions()
        assert len(history) == 1

        assistant.clear_interactions()
        assert len(assistant.get_interactions()) == 0

    def test_assistant_silence_detection(self):
        """Test silence detection."""
        from src.voice.assistant import VoiceAssistant

        assistant = VoiceAssistant()

        # Silence
        silence = bytes(100)
        assert assistant._is_silence(silence)

        # Loud audio
        loud = struct.pack("<50h", *([20000] * 50))
        assert not assistant._is_silence(loud)


class TestVoiceAssistantFactory:
    """Tests for voice assistant factory function."""

    def test_create_assistant(self):
        """Test creating voice assistant."""
        from src.voice.assistant import create_voice_assistant

        assistant = create_voice_assistant()
        assert assistant is not None

    def test_create_assistant_with_config(self):
        """Test creating assistant with config."""
        from src.voice.assistant import VoiceConfig, create_voice_assistant

        config = VoiceConfig(listen_timeout=5.0)
        assistant = create_voice_assistant(config=config)

        assert assistant.config.listen_timeout == 5.0

    def test_create_assistant_with_handler(self):
        """Test creating assistant with response handler."""
        from src.voice.assistant import create_voice_assistant

        def handler(text: str) -> str:
            return "response"

        assistant = create_voice_assistant(response_handler=handler)

        # Verify handler is set by testing response generation
        assert assistant._generate_response("test") == "response"


# =============================================================================
# Integration Tests
# =============================================================================


class TestVoiceModuleIntegration:
    """Integration tests for voice module."""

    def test_full_tts_pipeline(self):
        """Test full TTS pipeline."""
        from src.voice.audio import MockAudioPlayer
        from src.voice.tts import MockTTS

        tts = MockTTS()
        player = MockAudioPlayer()

        result = tts.synthesize("Hello, I am Agent OS")
        assert not result.is_empty

        player.play(result.audio_data)
        assert len(player.get_played_audio()) == 1

    def test_full_stt_pipeline(self):
        """Test full STT pipeline."""
        from src.voice.audio import AudioFormat, MockAudioCapture
        from src.voice.stt import MockSTT

        capture = MockAudioCapture()
        stt = MockSTT()
        stt.set_responses(["test transcription"])

        # Capture audio via callback
        chunks = []
        capture.on_audio(lambda c: chunks.append(c))
        capture.inject_audio(b"\x00" * 1000)

        # Transcribe
        result = stt.transcribe(chunks[0].data if chunks else b"audio", AudioFormat.PCM)
        assert result.text == "test transcription"

    def test_wake_word_to_stt(self):
        """Test wake word triggering STT."""
        from src.voice.audio import AudioFormat
        from src.voice.stt import MockSTT
        from src.voice.wakeword import MockWakeWordDetector

        detector = MockWakeWordDetector()
        stt = MockSTT()
        stt.set_responses(["turn on lights"])

        # Trigger wake word
        detector.trigger_detection("agent")
        event = detector.detect(b"audio")

        assert event is not None
        assert event.keyword == "agent"

        # Now transcribe the command
        result = stt.transcribe(b"command audio", AudioFormat.PCM)
        assert result.text == "turn on lights"


class TestModuleExports:
    """Test module exports are correct."""

    def test_voice_module_imports(self):
        """Test main module exports."""
        from src.voice import (
            AudioBuffer,
            AudioCapture,
            AudioFormat,
            AudioPlayer,
            PiperTTS,
            STTEngine,
            STTResult,
            TTSEngine,
            TTSResult,
            VoiceAssistant,
            VoiceConfig,
            VoiceState,
            WakeWordDetector,
            WakeWordEvent,
            WhisperSTT,
            create_audio_capture,
            create_audio_player,
            create_stt_engine,
            create_tts_engine,
            create_voice_assistant,
            create_wake_word_detector,
        )

        # Verify all exports exist
        assert AudioCapture is not None
        assert AudioPlayer is not None
        assert STTEngine is not None
        assert TTSEngine is not None
        assert WakeWordDetector is not None
        assert VoiceAssistant is not None
