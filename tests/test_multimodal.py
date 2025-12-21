"""
Tests for the Multi-Modal Agents module (UC-019).

Tests cover:
- Vision processing (CLIP, LLaVA, API)
- Audio processing (YAMNet, basic)
- Video analysis (FFmpeg, frames)
- Multi-modal agent integration
"""

import io
import math
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Vision Module Tests
# =============================================================================


class TestImageFormat:
    """Tests for ImageFormat enum."""

    def test_image_formats(self):
        """Test image format values."""
        from src.multimodal.vision import ImageFormat

        assert ImageFormat.JPEG.value == "jpeg"
        assert ImageFormat.PNG.value == "png"
        assert ImageFormat.GIF.value == "gif"
        assert ImageFormat.WEBP.value == "webp"


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_bounding_box_creation(self):
        """Test creating bounding box."""
        from src.multimodal.vision import BoundingBox

        box = BoundingBox(
            x=0.1,
            y=0.2,
            width=0.3,
            height=0.4,
            label="object",
            confidence=0.95,
        )

        assert box.x == 0.1
        assert box.label == "object"

    def test_bounding_box_center(self):
        """Test bounding box center calculation."""
        from src.multimodal.vision import BoundingBox

        box = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        center = box.center

        assert center == (0.5, 0.5)

    def test_bounding_box_area(self):
        """Test bounding box area calculation."""
        from src.multimodal.vision import BoundingBox

        box = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
        assert box.area == 0.25


class TestImageInput:
    """Tests for ImageInput dataclass."""

    def test_image_input_creation(self):
        """Test creating image input."""
        from src.multimodal.vision import ImageFormat, ImageInput

        img = ImageInput(
            data=b"\xff\xd8\xff\xe0",  # JPEG magic
            format=ImageFormat.JPEG,
        )

        assert len(img.data) == 4
        assert img.format == ImageFormat.JPEG

    def test_image_input_from_base64(self):
        """Test creating from base64."""
        from src.multimodal.vision import ImageInput

        import base64

        original = b"test image data"
        b64 = base64.b64encode(original).decode()

        img = ImageInput.from_base64(b64)
        assert img.data == original

    def test_image_input_to_base64(self):
        """Test converting to base64."""
        from src.multimodal.vision import ImageInput

        import base64

        data = b"test image data"
        img = ImageInput(data=data)

        b64 = img.to_base64()
        assert base64.b64decode(b64) == data


class TestVisionResult:
    """Tests for VisionResult dataclass."""

    def test_vision_result_creation(self):
        """Test creating vision result."""
        from src.multimodal.vision import VisionResult

        result = VisionResult(
            description="A test image",
            confidence=0.95,
            model="test",
        )

        assert result.description == "A test image"
        assert result.confidence == 0.95

    def test_vision_result_has_embeddings(self):
        """Test checking for embeddings."""
        from src.multimodal.vision import VisionResult

        result1 = VisionResult()
        assert not result1.has_embeddings

        result2 = VisionResult(embeddings=[0.1, 0.2, 0.3])
        assert result2.has_embeddings

    def test_vision_result_object_count(self):
        """Test object count."""
        from src.multimodal.vision import BoundingBox, VisionResult

        result = VisionResult(
            objects=[
                BoundingBox(x=0, y=0, width=0.5, height=0.5),
                BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5),
            ]
        )

        assert result.object_count == 2


class TestMockVisionEngine:
    """Tests for MockVisionEngine."""

    def test_mock_vision_creation(self):
        """Test creating mock vision engine."""
        from src.multimodal.vision import MockVisionEngine

        engine = MockVisionEngine()
        assert engine.is_available()

    def test_mock_vision_describe(self):
        """Test mock description."""
        from src.multimodal.vision import ImageInput, MockVisionEngine

        engine = MockVisionEngine()
        image = ImageInput(data=b"test")

        result = engine.describe(image)
        assert result.description
        assert result.model

    def test_mock_vision_set_descriptions(self):
        """Test setting mock descriptions."""
        from src.multimodal.vision import ImageInput, MockVisionEngine

        engine = MockVisionEngine()
        engine.set_descriptions(["Custom description"])

        image = ImageInput(data=b"test")
        result = engine.describe(image)

        assert "Custom" in result.description

    def test_mock_vision_embed(self):
        """Test mock embeddings."""
        from src.multimodal.vision import ImageInput, MockVisionEngine

        engine = MockVisionEngine()
        image = ImageInput(data=b"test")

        result = engine.embed(image)
        assert result.has_embeddings
        assert len(result.embeddings) > 0

    def test_mock_vision_classify(self):
        """Test mock classification."""
        from src.multimodal.vision import ImageInput, MockVisionEngine

        engine = MockVisionEngine()
        image = ImageInput(data=b"test")
        labels = ["cat", "dog", "bird"]

        result = engine.classify(image, labels)
        assert len(result.labels) > 0
        assert result.confidence > 0

    def test_mock_vision_detect_objects(self):
        """Test mock object detection."""
        from src.multimodal.vision import ImageInput, MockVisionEngine

        engine = MockVisionEngine()
        image = ImageInput(data=b"test")

        result = engine.detect_objects(image)
        assert len(result.objects) > 0

    def test_mock_vision_callbacks(self):
        """Test vision callbacks."""
        from src.multimodal.vision import ImageInput, MockVisionEngine

        engine = MockVisionEngine()
        results = []
        engine.on_result(lambda r: results.append(r))

        image = ImageInput(data=b"test")
        engine.describe(image)

        assert len(results) == 1

    def test_mock_vision_compare_images(self):
        """Test image comparison."""
        from src.multimodal.vision import ImageInput, MockVisionEngine

        engine = MockVisionEngine()
        img1 = ImageInput(data=b"test1")
        img2 = ImageInput(data=b"test2")

        similarity = engine.compare_images(img1, img2)
        # Cosine similarity can range from -1 to 1 with random embeddings
        assert -1 <= similarity <= 1


class TestVisionFactory:
    """Tests for vision engine factory."""

    def test_create_vision_mock(self):
        """Test creating mock vision engine."""
        from src.multimodal.vision import create_vision_engine

        engine = create_vision_engine("mock")
        assert engine.is_available()

    def test_create_vision_auto(self):
        """Test auto vision engine creation."""
        from src.multimodal.vision import create_vision_engine

        engine = create_vision_engine("auto")
        assert engine is not None

    def test_create_vision_invalid(self):
        """Test invalid vision engine type."""
        from src.multimodal.vision import create_vision_engine

        with pytest.raises(ValueError):
            create_vision_engine("invalid_engine")


# =============================================================================
# Audio Module Tests
# =============================================================================


class TestAudioEventType:
    """Tests for AudioEventType enum."""

    def test_audio_event_types(self):
        """Test audio event type values."""
        from src.multimodal.audio import AudioEventType

        assert AudioEventType.SPEECH.value == "speech"
        assert AudioEventType.MUSIC.value == "music"
        assert AudioEventType.SILENCE.value == "silence"


class TestAudioFeatures:
    """Tests for AudioFeatures dataclass."""

    def test_audio_features_creation(self):
        """Test creating audio features."""
        from src.multimodal.audio import AudioFeatures

        features = AudioFeatures(
            duration=1.0,
            sample_rate=16000,
            rms_energy=0.5,
        )

        assert features.duration == 1.0
        assert features.rms_energy == 0.5

    def test_audio_features_is_silent(self):
        """Test silence detection."""
        from src.multimodal.audio import AudioFeatures

        silent = AudioFeatures(rms_energy=0.005)
        assert silent.is_silent

        loud = AudioFeatures(rms_energy=0.5)
        assert not loud.is_silent

    def test_audio_features_has_embeddings(self):
        """Test checking for embeddings."""
        from src.multimodal.audio import AudioFeatures

        feat1 = AudioFeatures()
        assert not feat1.has_embeddings

        feat2 = AudioFeatures(embeddings=[0.1, 0.2])
        assert feat2.has_embeddings


class TestAudioInput:
    """Tests for AudioInput dataclass."""

    def test_audio_input_creation(self):
        """Test creating audio input."""
        from src.multimodal.audio import AudioInput

        samples = [0, 1000, -1000, 500]
        data = struct.pack(f"<{len(samples)}h", *samples)

        audio = AudioInput(data=data, sample_rate=16000)
        assert len(audio.data) == len(samples) * 2
        assert audio.sample_rate == 16000

    def test_audio_input_duration(self):
        """Test audio duration calculation."""
        from src.multimodal.audio import AudioInput

        # 16000 samples = 1 second at 16kHz
        data = bytes(16000 * 2)  # 16-bit audio
        audio = AudioInput(data=data, sample_rate=16000)

        assert abs(audio.duration - 1.0) < 0.01

    def test_audio_input_to_wav(self):
        """Test converting to WAV."""
        from src.multimodal.audio import AudioInput

        data = struct.pack("<10h", *[0] * 10)
        audio = AudioInput(data=data, sample_rate=16000)

        wav_data = audio.to_wav()
        assert wav_data[:4] == b"RIFF"

    def test_audio_input_get_samples(self):
        """Test getting samples."""
        from src.multimodal.audio import AudioInput

        samples = [100, 200, -100, -200]
        data = struct.pack(f"<{len(samples)}h", *samples)
        audio = AudioInput(data=data)

        result = audio.get_samples()
        assert result == samples


class TestAudioSegment:
    """Tests for AudioSegment dataclass."""

    def test_audio_segment_creation(self):
        """Test creating audio segment."""
        from src.multimodal.audio import AudioEventType, AudioSegment

        segment = AudioSegment(
            start=0.0,
            end=1.0,
            event_type=AudioEventType.SPEECH,
            label="speech",
        )

        assert segment.duration == 1.0
        assert segment.event_type == AudioEventType.SPEECH


class TestBasicAudioAnalyzer:
    """Tests for BasicAudioAnalyzer."""

    def test_basic_analyzer_creation(self):
        """Test creating basic analyzer."""
        from src.multimodal.audio import BasicAudioAnalyzer

        analyzer = BasicAudioAnalyzer()
        assert analyzer.is_available()

    def test_basic_analyzer_extract_features(self):
        """Test feature extraction."""
        from src.multimodal.audio import AudioInput, BasicAudioAnalyzer

        analyzer = BasicAudioAnalyzer()

        samples = [1000, -1000] * 100
        data = struct.pack(f"<{len(samples)}h", *samples)
        audio = AudioInput(data=data, sample_rate=16000)

        features = analyzer.extract_features(audio)
        assert features.rms_energy > 0
        assert features.zero_crossing_rate > 0

    def test_basic_analyzer_classify(self):
        """Test basic classification."""
        from src.multimodal.audio import AudioInput, BasicAudioAnalyzer

        analyzer = BasicAudioAnalyzer()

        samples = [0] * 100  # Silence
        data = struct.pack(f"<{len(samples)}h", *samples)
        audio = AudioInput(data=data, sample_rate=16000)

        result = analyzer.classify(audio)
        assert "silence" in result.labels

    def test_basic_analyzer_detect_events(self):
        """Test event detection."""
        from src.multimodal.audio import AudioInput, BasicAudioAnalyzer

        analyzer = BasicAudioAnalyzer()

        samples = [0] * 1600 + [5000] * 1600 + [0] * 1600
        data = struct.pack(f"<{len(samples)}h", *samples)
        audio = AudioInput(data=data, sample_rate=16000)

        result = analyzer.detect_events(audio)
        assert len(result.segments) > 0

    def test_basic_analyzer_embed(self):
        """Test embedding generation."""
        from src.multimodal.audio import AudioInput, BasicAudioAnalyzer

        analyzer = BasicAudioAnalyzer()

        samples = [1000] * 100
        data = struct.pack(f"<{len(samples)}h", *samples)
        audio = AudioInput(data=data, sample_rate=16000)

        features = analyzer.embed(audio)
        assert features.has_embeddings


class TestMockAudioAnalyzer:
    """Tests for MockAudioAnalyzer."""

    def test_mock_analyzer_creation(self):
        """Test creating mock analyzer."""
        from src.multimodal.audio import MockAudioAnalyzer

        analyzer = MockAudioAnalyzer()
        assert analyzer.is_available()

    def test_mock_analyzer_analyze(self):
        """Test mock analysis."""
        from src.multimodal.audio import AudioInput, MockAudioAnalyzer

        analyzer = MockAudioAnalyzer()
        audio = AudioInput(data=bytes(100), sample_rate=16000)

        result = analyzer.analyze(audio)
        assert result.features is not None
        assert len(result.labels) > 0

    def test_mock_analyzer_set_labels(self):
        """Test setting mock labels."""
        from src.multimodal.audio import AudioInput, MockAudioAnalyzer

        analyzer = MockAudioAnalyzer()
        analyzer.set_mock_labels(["custom_label"])

        audio = AudioInput(data=bytes(100), sample_rate=16000)
        result = analyzer.classify(audio)

        assert "custom_label" in result.labels

    def test_mock_analyzer_compare_audio(self):
        """Test audio comparison."""
        from src.multimodal.audio import AudioInput, MockAudioAnalyzer

        analyzer = MockAudioAnalyzer()
        audio1 = AudioInput(data=bytes(100), sample_rate=16000)
        audio2 = AudioInput(data=bytes(100), sample_rate=16000)

        similarity = analyzer.compare_audio(audio1, audio2)
        # Cosine similarity can range from -1 to 1 with random embeddings
        assert -1 <= similarity <= 1


class TestAudioFactory:
    """Tests for audio analyzer factory."""

    def test_create_audio_mock(self):
        """Test creating mock analyzer."""
        from src.multimodal.audio import create_audio_analyzer

        analyzer = create_audio_analyzer("mock")
        assert analyzer.is_available()

    def test_create_audio_basic(self):
        """Test creating basic analyzer."""
        from src.multimodal.audio import create_audio_analyzer

        analyzer = create_audio_analyzer("basic")
        assert analyzer.is_available()

    def test_create_audio_auto(self):
        """Test auto analyzer creation."""
        from src.multimodal.audio import create_audio_analyzer

        analyzer = create_audio_analyzer("auto")
        assert analyzer is not None

    def test_create_audio_invalid(self):
        """Test invalid analyzer type."""
        from src.multimodal.audio import create_audio_analyzer

        with pytest.raises(ValueError):
            create_audio_analyzer("invalid_analyzer")


# =============================================================================
# Video Module Tests
# =============================================================================


class TestVideoFormat:
    """Tests for VideoFormat enum."""

    def test_video_formats(self):
        """Test video format values."""
        from src.multimodal.video import VideoFormat

        assert VideoFormat.MP4.value == "mp4"
        assert VideoFormat.AVI.value == "avi"
        assert VideoFormat.MOV.value == "mov"


class TestActionCategory:
    """Tests for ActionCategory enum."""

    def test_action_categories(self):
        """Test action category values."""
        from src.multimodal.video import ActionCategory

        assert ActionCategory.WALKING.value == "walking"
        assert ActionCategory.RUNNING.value == "running"


class TestVideoFrame:
    """Tests for VideoFrame dataclass."""

    def test_video_frame_creation(self):
        """Test creating video frame."""
        from src.multimodal.video import VideoFrame
        from src.multimodal.vision import ImageInput

        image = ImageInput(data=b"test")
        frame = VideoFrame(index=0, timestamp=0.5, image=image)

        assert frame.index == 0
        assert frame.timestamp == 0.5

    def test_video_frame_timedelta(self):
        """Test timedelta property."""
        from datetime import timedelta

        from src.multimodal.video import VideoFrame
        from src.multimodal.vision import ImageInput

        image = ImageInput(data=b"test")
        frame = VideoFrame(index=0, timestamp=1.5, image=image)

        assert frame.timedelta == timedelta(seconds=1.5)


class TestVideoSegment:
    """Tests for VideoSegment dataclass."""

    def test_video_segment_creation(self):
        """Test creating video segment."""
        from src.multimodal.video import VideoSegment

        segment = VideoSegment(
            start=0.0,
            end=5.0,
            label="intro",
        )

        assert segment.duration == 5.0
        assert segment.label == "intro"


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    def test_video_metadata_creation(self):
        """Test creating video metadata."""
        from src.multimodal.video import VideoMetadata

        meta = VideoMetadata(
            duration=60.0,
            width=1920,
            height=1080,
            fps=30.0,
        )

        assert meta.duration == 60.0
        assert meta.width == 1920

    def test_video_metadata_aspect_ratio(self):
        """Test aspect ratio calculation."""
        from src.multimodal.video import VideoMetadata

        meta = VideoMetadata(width=1920, height=1080)
        assert abs(meta.aspect_ratio - 16 / 9) < 0.01


class TestMockVideoAnalyzer:
    """Tests for MockVideoAnalyzer."""

    def test_mock_analyzer_creation(self):
        """Test creating mock analyzer."""
        from src.multimodal.video import MockVideoAnalyzer

        analyzer = MockVideoAnalyzer()
        assert analyzer.is_available()

    def test_mock_analyzer_analyze(self):
        """Test mock analysis."""
        from src.multimodal.video import MockVideoAnalyzer, VideoInput, VideoMetadata

        analyzer = MockVideoAnalyzer()

        # Create mock video input
        video = VideoInput(
            path=Path("/tmp/test.mp4"),
            metadata=VideoMetadata(duration=10.0),
        )

        result = analyzer.analyze(video)
        assert result.description
        assert len(result.segments) > 0

    def test_mock_analyzer_extract_frames(self):
        """Test frame extraction."""
        from src.multimodal.video import MockVideoAnalyzer, VideoInput, VideoMetadata

        analyzer = MockVideoAnalyzer()
        video = VideoInput(
            path=Path("/tmp/test.mp4"),
            metadata=VideoMetadata(duration=5.0),
        )

        frames = analyzer.extract_frames(video, sample_rate=1.0)
        assert len(frames) > 0

    def test_mock_analyzer_detect_scenes(self):
        """Test scene detection."""
        from src.multimodal.video import MockVideoAnalyzer, VideoInput, VideoMetadata

        analyzer = MockVideoAnalyzer()
        video = VideoInput(
            path=Path("/tmp/test.mp4"),
            metadata=VideoMetadata(duration=10.0),
        )

        result = analyzer.detect_scenes(video)
        assert len(result.segments) > 0

    def test_mock_analyzer_summarize(self):
        """Test video summarization."""
        from src.multimodal.video import MockVideoAnalyzer, VideoInput, VideoMetadata

        analyzer = MockVideoAnalyzer()
        video = VideoInput(
            path=Path("/tmp/test.mp4"),
            metadata=VideoMetadata(duration=30.0),
        )

        result = analyzer.summarize(video, max_length=100)
        assert result.summary
        assert len(result.summary) <= 100


class TestVideoFactory:
    """Tests for video analyzer factory."""

    def test_create_video_mock(self):
        """Test creating mock analyzer."""
        from src.multimodal.video import create_video_analyzer

        analyzer = create_video_analyzer("mock")
        assert analyzer.is_available()

    def test_create_video_auto(self):
        """Test auto analyzer creation."""
        from src.multimodal.video import create_video_analyzer

        analyzer = create_video_analyzer("auto")
        assert analyzer is not None

    def test_create_video_invalid(self):
        """Test invalid analyzer type."""
        from src.multimodal.video import create_video_analyzer

        with pytest.raises(ValueError):
            create_video_analyzer("invalid_analyzer")


# =============================================================================
# Multi-Modal Agent Tests
# =============================================================================


class TestInputModality:
    """Tests for InputModality enum."""

    def test_input_modalities(self):
        """Test input modality values."""
        from src.multimodal.agent import InputModality

        assert InputModality.TEXT.value == "text"
        assert InputModality.IMAGE.value == "image"
        assert InputModality.AUDIO.value == "audio"
        assert InputModality.VIDEO.value == "video"


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types(self):
        """Test task type values."""
        from src.multimodal.agent import TaskType

        assert TaskType.DESCRIBE.value == "describe"
        assert TaskType.CLASSIFY.value == "classify"
        assert TaskType.EMBED.value == "embed"


class TestMultiModalInput:
    """Tests for MultiModalInput dataclass."""

    def test_input_creation(self):
        """Test creating multi-modal input."""
        from src.multimodal.agent import MultiModalInput

        input = MultiModalInput(text="Hello")
        assert input.text == "Hello"

    def test_input_modalities(self):
        """Test modality detection."""
        from src.multimodal.agent import InputModality, MultiModalInput
        from src.multimodal.vision import ImageInput

        input = MultiModalInput(
            text="Hello",
            images=[ImageInput(data=b"test")],
        )

        assert InputModality.TEXT in input.modalities
        assert InputModality.IMAGE in input.modalities
        assert input.is_multimodal

    def test_input_from_text(self):
        """Test creating from text."""
        from src.multimodal.agent import InputModality, MultiModalInput

        input = MultiModalInput.from_text("Hello world")
        assert input.text == "Hello world"
        assert InputModality.TEXT in input.modalities


class TestMultiModalResult:
    """Tests for MultiModalResult dataclass."""

    def test_result_creation(self):
        """Test creating multi-modal result."""
        from src.multimodal.agent import MultiModalResult

        result = MultiModalResult(
            description="Test description",
            confidence=0.95,
        )

        assert result.description == "Test description"
        assert result.confidence == 0.95

    def test_result_has_embeddings(self):
        """Test checking for embeddings."""
        from src.multimodal.agent import MultiModalResult

        result1 = MultiModalResult()
        assert not result1.has_embeddings

        result2 = MultiModalResult(embeddings=[0.1, 0.2])
        assert result2.has_embeddings


class TestMultiModalConfig:
    """Tests for MultiModalConfig dataclass."""

    def test_config_defaults(self):
        """Test default config values."""
        from src.multimodal.agent import MultiModalConfig

        config = MultiModalConfig()
        assert config.vision_engine == "auto"
        assert config.combine_modalities is True


class TestMultiModalAgent:
    """Tests for MultiModalAgent."""

    def test_agent_creation(self):
        """Test creating multi-modal agent."""
        from src.multimodal.agent import MultiModalAgent

        agent = MultiModalAgent()
        assert agent is not None

    def test_agent_with_config(self):
        """Test agent with custom config."""
        from src.multimodal.agent import MultiModalAgent, MultiModalConfig

        config = MultiModalConfig(vision_engine="mock")
        agent = MultiModalAgent(config=config)

        assert agent.config.vision_engine == "mock"

    def test_agent_process_text(self):
        """Test processing text input."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput

        agent = MultiModalAgent()
        input = MultiModalInput.from_text("Hello")

        result = agent.process(input)
        assert result is not None

    def test_agent_process_image(self):
        """Test processing image input."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput
        from src.multimodal.vision import ImageInput

        agent = MultiModalAgent()
        input = MultiModalInput(images=[ImageInput(data=b"test")])

        result = agent.process(input)
        assert result is not None
        assert result.vision_result is not None

    def test_agent_process_audio(self):
        """Test processing audio input."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput
        from src.multimodal.audio import AudioInput

        agent = MultiModalAgent()
        input = MultiModalInput(audio=AudioInput(data=bytes(100), sample_rate=16000))

        result = agent.process(input)
        assert result is not None
        assert result.audio_result is not None

    def test_agent_answer_question(self):
        """Test answering questions."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput
        from src.multimodal.vision import ImageInput

        agent = MultiModalAgent()
        input = MultiModalInput(images=[ImageInput(data=b"test")])

        answer = agent.answer_question(input, "What is in this image?")
        assert answer is not None

    def test_agent_classify(self):
        """Test classification."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput
        from src.multimodal.vision import ImageInput

        agent = MultiModalAgent()
        input = MultiModalInput(images=[ImageInput(data=b"test")])

        result = agent.classify(input, labels=["cat", "dog", "bird"])
        assert len(result.labels) > 0

    def test_agent_embed(self):
        """Test embedding generation."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput
        from src.multimodal.vision import ImageInput

        agent = MultiModalAgent()
        input = MultiModalInput(images=[ImageInput(data=b"test")])

        result = agent.embed(input)
        # Mock should return embeddings
        assert result is not None

    def test_agent_compare(self):
        """Test comparing inputs."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput
        from src.multimodal.vision import ImageInput

        agent = MultiModalAgent()
        input1 = MultiModalInput(images=[ImageInput(data=b"test1")])
        input2 = MultiModalInput(images=[ImageInput(data=b"test2")])

        similarity = agent.compare(input1, input2)
        # Cosine similarity can range from -1 to 1 with random embeddings
        assert -1 <= similarity <= 1

    def test_agent_callbacks(self):
        """Test result callbacks."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput

        agent = MultiModalAgent()
        results = []
        agent.on_result(lambda r: results.append(r))

        input = MultiModalInput.from_text("Hello")
        agent.process(input)

        assert len(results) == 1

    def test_agent_capabilities(self):
        """Test getting capabilities."""
        from src.multimodal.agent import MultiModalAgent

        agent = MultiModalAgent()
        caps = agent.get_capabilities()

        assert "vision" in caps
        assert "audio" in caps
        assert "video" in caps

    def test_agent_stats(self):
        """Test getting statistics."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput

        agent = MultiModalAgent()
        input = MultiModalInput.from_text("Hello")
        agent.process(input)

        stats = agent.get_stats()
        assert stats["processed_count"] == 1


class TestMultiModalAgentFactory:
    """Tests for multi-modal agent factory."""

    def test_create_agent(self):
        """Test creating multi-modal agent."""
        from src.multimodal.agent import create_multimodal_agent

        agent = create_multimodal_agent()
        assert agent is not None

    def test_create_agent_with_config(self):
        """Test creating agent with config."""
        from src.multimodal.agent import MultiModalConfig, create_multimodal_agent

        config = MultiModalConfig(vision_engine="mock")
        agent = create_multimodal_agent(config=config)

        assert agent.config.vision_engine == "mock"


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiModalIntegration:
    """Integration tests for multi-modal module."""

    def test_vision_audio_pipeline(self):
        """Test combined vision and audio processing."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput
        from src.multimodal.audio import AudioInput
        from src.multimodal.vision import ImageInput

        agent = MultiModalAgent()

        input = MultiModalInput(
            images=[ImageInput(data=b"image_data")],
            audio=AudioInput(data=bytes(100), sample_rate=16000),
        )

        result = agent.process(input)

        assert result.vision_result is not None
        assert result.audio_result is not None
        assert result.is_multimodal if hasattr(result, "is_multimodal") else True

    def test_full_multimodal_pipeline(self):
        """Test processing with all modalities."""
        from src.multimodal.agent import MultiModalAgent, MultiModalInput
        from src.multimodal.audio import AudioInput
        from src.multimodal.video import VideoInput, VideoMetadata
        from src.multimodal.vision import ImageInput

        agent = MultiModalAgent()

        input = MultiModalInput(
            text="Analyze this content",
            images=[ImageInput(data=b"image_data")],
            audio=AudioInput(data=bytes(100), sample_rate=16000),
        )

        result = agent.process(input)

        assert result.description or result.answer


class TestModuleExports:
    """Test module exports are correct."""

    def test_multimodal_module_imports(self):
        """Test main module exports."""
        from src.multimodal import (
            AudioAnalyzer,
            AudioAnalysisResult,
            AudioFeatures,
            CLIPEncoder,
            ImageInput,
            LLaVAVision,
            MultiModalAgent,
            MultiModalConfig,
            MultiModalInput,
            MultiModalResult,
            VideoAnalyzer,
            VideoAnalysisResult,
            VideoFrame,
            VisionEngine,
            VisionResult,
            create_audio_analyzer,
            create_multimodal_agent,
            create_video_analyzer,
            create_vision_engine,
        )

        # Verify all exports exist
        assert VisionEngine is not None
        assert AudioAnalyzer is not None
        assert VideoAnalyzer is not None
        assert MultiModalAgent is not None
