"""
Video Analysis Module

Provides video understanding capabilities including:
- Frame extraction and analysis
- Temporal understanding
- Action recognition
- Scene detection
- Video summarization
"""

import io
import logging
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .vision import ImageInput, VisionEngine, VisionResult, create_vision_engine

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class VideoFormat(str, Enum):
    """Supported video formats."""

    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    GIF = "gif"


class ActionCategory(str, Enum):
    """Action recognition categories."""

    WALKING = "walking"
    RUNNING = "running"
    SITTING = "sitting"
    STANDING = "standing"
    TALKING = "talking"
    EATING = "eating"
    DRINKING = "drinking"
    READING = "reading"
    WRITING = "writing"
    TYPING = "typing"
    DANCING = "dancing"
    COOKING = "cooking"
    DRIVING = "driving"
    UNKNOWN = "unknown"


@dataclass
class VideoFrame:
    """A single frame from video."""

    index: int  # Frame number
    timestamp: float  # Time in seconds
    image: ImageInput  # Frame image data
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timedelta(self) -> timedelta:
        """Get timestamp as timedelta."""
        return timedelta(seconds=self.timestamp)


@dataclass
class VideoSegment:
    """A segment of video with temporal boundaries."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    label: str = ""
    description: str = ""
    actions: List[ActionCategory] = field(default_factory=list)
    confidence: float = 1.0
    key_frame: Optional[VideoFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end - self.start


@dataclass
class VideoMetadata:
    """Video file metadata."""

    duration: float = 0.0  # Duration in seconds
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    format: VideoFormat = VideoFormat.MP4
    codec: str = ""
    bitrate: int = 0
    has_audio: bool = False
    source: Optional[str] = None

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        if self.height > 0:
            return self.width / self.height
        return 0.0


@dataclass
class VideoInput:
    """Input video with metadata."""

    path: Path
    metadata: VideoMetadata = field(default_factory=VideoMetadata)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "VideoInput":
        """Load video from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        # Get metadata using ffprobe if available
        metadata = cls._extract_metadata(path)

        return cls(path=path, metadata=metadata)

    @staticmethod
    def _extract_metadata(path: Path) -> VideoMetadata:
        """Extract video metadata using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                import json

                data = json.loads(result.stdout)

                # Find video stream
                video_stream = None
                has_audio = False

                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video" and not video_stream:
                        video_stream = stream
                    elif stream.get("codec_type") == "audio":
                        has_audio = True

                if video_stream:
                    # Parse frame rate
                    fps_str = video_stream.get("r_frame_rate", "0/1")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        fps = float(num) / float(den) if float(den) > 0 else 0
                    else:
                        fps = float(fps_str)

                    fmt = data.get("format", {})

                    # Detect format from extension
                    ext = path.suffix.lower().lstrip(".")
                    format_map = {
                        "mp4": VideoFormat.MP4,
                        "avi": VideoFormat.AVI,
                        "mov": VideoFormat.MOV,
                        "mkv": VideoFormat.MKV,
                        "webm": VideoFormat.WEBM,
                        "gif": VideoFormat.GIF,
                    }
                    video_format = format_map.get(ext, VideoFormat.MP4)

                    return VideoMetadata(
                        duration=float(fmt.get("duration", 0)),
                        width=int(video_stream.get("width", 0)),
                        height=int(video_stream.get("height", 0)),
                        fps=fps,
                        frame_count=int(video_stream.get("nb_frames", 0)),
                        format=video_format,
                        codec=video_stream.get("codec_name", ""),
                        bitrate=int(fmt.get("bit_rate", 0)),
                        has_audio=has_audio,
                        source=str(path),
                    )

        except FileNotFoundError:
            logger.warning("ffprobe not found, metadata extraction limited")
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")

        return VideoMetadata(source=str(path))


@dataclass
class VideoAnalysisResult:
    """Result from video analysis."""

    description: str = ""
    summary: str = ""
    segments: List[VideoSegment] = field(default_factory=list)
    actions: List[ActionCategory] = field(default_factory=list)
    key_frames: List[VideoFrame] = field(default_factory=list)
    frame_descriptions: Dict[int, str] = field(default_factory=dict)
    model: str = ""
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def scene_count(self) -> int:
        """Get number of detected scenes."""
        return len(self.segments)


# =============================================================================
# Video Analyzer Interface
# =============================================================================


class VideoAnalyzer(ABC):
    """
    Abstract base class for video analysis engines.
    """

    def __init__(self, vision_engine: Optional[VisionEngine] = None):
        self.vision_engine = vision_engine or create_vision_engine("mock")
        self._callbacks: List[Callable[[VideoAnalysisResult], None]] = []

    @abstractmethod
    def analyze(
        self,
        video: VideoInput,
        sample_rate: float = 1.0,
    ) -> VideoAnalysisResult:
        """
        Perform comprehensive video analysis.

        Args:
            video: Input video
            sample_rate: Frames per second to sample

        Returns:
            VideoAnalysisResult with analysis
        """
        pass

    @abstractmethod
    def extract_frames(
        self,
        video: VideoInput,
        sample_rate: float = 1.0,
        max_frames: int = 100,
    ) -> List[VideoFrame]:
        """
        Extract frames from video.

        Args:
            video: Input video
            sample_rate: Frames per second to extract
            max_frames: Maximum frames to extract

        Returns:
            List of VideoFrame objects
        """
        pass

    @abstractmethod
    def detect_scenes(self, video: VideoInput) -> VideoAnalysisResult:
        """
        Detect scene boundaries.

        Args:
            video: Input video

        Returns:
            VideoAnalysisResult with scene segments
        """
        pass

    @abstractmethod
    def summarize(
        self,
        video: VideoInput,
        max_length: int = 200,
    ) -> VideoAnalysisResult:
        """
        Generate video summary.

        Args:
            video: Input video
            max_length: Maximum summary length

        Returns:
            VideoAnalysisResult with summary
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the analyzer is available."""
        pass

    def describe_frame(self, frame: VideoFrame, prompt: Optional[str] = None) -> str:
        """Describe a single frame."""
        result = self.vision_engine.describe(frame.image, prompt)
        return result.description

    def answer_question(self, video: VideoInput, question: str) -> str:
        """Answer a question about the video."""
        result = self.analyze(video)
        return f"Based on the video: {result.description}"

    def on_result(self, callback: Callable[[VideoAnalysisResult], None]) -> None:
        """Register callback for analysis results."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, result: VideoAnalysisResult) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Video analyzer callback error: {e}")


# =============================================================================
# Mock Video Analyzer
# =============================================================================


class MockVideoAnalyzer(VideoAnalyzer):
    """
    Mock video analyzer for testing.
    """

    def __init__(self, vision_engine: Optional[VisionEngine] = None):
        super().__init__(vision_engine)
        self._mock_description = "A sample video showing various scenes."
        self._mock_actions = [ActionCategory.WALKING, ActionCategory.TALKING]

    def set_mock_description(self, description: str) -> None:
        """Set mock description."""
        self._mock_description = description

    def set_mock_actions(self, actions: List[ActionCategory]) -> None:
        """Set mock actions."""
        self._mock_actions = actions

    def analyze(
        self,
        video: VideoInput,
        sample_rate: float = 1.0,
    ) -> VideoAnalysisResult:
        """Return mock analysis."""
        import random

        frames = self.extract_frames(video, sample_rate)

        segments = [
            VideoSegment(
                start=0.0,
                end=video.metadata.duration / 2,
                label="scene_1",
                actions=self._mock_actions[:1],
            ),
            VideoSegment(
                start=video.metadata.duration / 2,
                end=video.metadata.duration,
                label="scene_2",
                actions=self._mock_actions[1:],
            ),
        ]

        result = VideoAnalysisResult(
            description=self._mock_description,
            summary=f"Video summary: {self._mock_description[:50]}...",
            segments=segments,
            actions=self._mock_actions,
            key_frames=frames[:3] if frames else [],
            model="mock",
        )

        self._notify_callbacks(result)
        return result

    def extract_frames(
        self,
        video: VideoInput,
        sample_rate: float = 1.0,
        max_frames: int = 100,
    ) -> List[VideoFrame]:
        """Return mock frames."""
        duration = video.metadata.duration or 10.0
        num_frames = min(int(duration * sample_rate), max_frames)

        frames = []
        for i in range(num_frames):
            timestamp = i / sample_rate
            # Create a mock image
            mock_image = ImageInput(
                data=b"\x00" * 100,
                source=f"frame_{i}",
            )
            frames.append(
                VideoFrame(
                    index=i,
                    timestamp=timestamp,
                    image=mock_image,
                )
            )

        return frames

    def detect_scenes(self, video: VideoInput) -> VideoAnalysisResult:
        """Return mock scenes."""
        duration = video.metadata.duration or 10.0

        segments = [
            VideoSegment(start=0.0, end=duration * 0.3, label="intro"),
            VideoSegment(start=duration * 0.3, end=duration * 0.7, label="main"),
            VideoSegment(start=duration * 0.7, end=duration, label="outro"),
        ]

        result = VideoAnalysisResult(
            segments=segments,
            model="mock",
        )

        self._notify_callbacks(result)
        return result

    def summarize(
        self,
        video: VideoInput,
        max_length: int = 200,
    ) -> VideoAnalysisResult:
        """Return mock summary."""
        summary = f"This is a {video.metadata.duration:.1f} second video. "
        summary += self._mock_description[:max_length - len(summary)]

        result = VideoAnalysisResult(
            summary=summary,
            model="mock",
        )

        self._notify_callbacks(result)
        return result

    def is_available(self) -> bool:
        """Mock is always available."""
        return True


# =============================================================================
# FFmpeg-based Video Analyzer
# =============================================================================


class FFmpegVideoAnalyzer(VideoAnalyzer):
    """
    Video analyzer using FFmpeg for frame extraction
    and a vision engine for frame analysis.
    """

    def __init__(
        self,
        vision_engine: Optional[VisionEngine] = None,
        ffmpeg_path: str = "ffmpeg",
    ):
        super().__init__(vision_engine)
        self.ffmpeg_path = ffmpeg_path

    def analyze(
        self,
        video: VideoInput,
        sample_rate: float = 1.0,
    ) -> VideoAnalysisResult:
        """Analyze video by sampling and analyzing frames."""
        start_time = time.time()

        if not self.is_available():
            return VideoAnalysisResult(
                description="FFmpeg not available",
                metadata={"error": "ffmpeg_not_found"},
            )

        # Extract frames
        frames = self.extract_frames(video, sample_rate, max_frames=10)

        if not frames:
            return VideoAnalysisResult(
                description="Could not extract frames",
                metadata={"error": "no_frames"},
            )

        # Analyze each frame
        frame_descriptions = {}
        for frame in frames:
            desc = self.describe_frame(frame)
            frame_descriptions[frame.index] = desc

        # Generate overall description
        descriptions = list(frame_descriptions.values())
        overall = self._synthesize_description(descriptions)

        # Detect scenes based on frame differences
        scenes = self._detect_scene_changes(frames)

        result = VideoAnalysisResult(
            description=overall,
            summary=overall[:200],
            segments=scenes,
            key_frames=frames,
            frame_descriptions=frame_descriptions,
            model="ffmpeg+vision",
            processing_time=time.time() - start_time,
        )

        self._notify_callbacks(result)
        return result

    def extract_frames(
        self,
        video: VideoInput,
        sample_rate: float = 1.0,
        max_frames: int = 100,
    ) -> List[VideoFrame]:
        """Extract frames using FFmpeg."""
        if not self.is_available():
            return []

        frames = []

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract frames with FFmpeg
                output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")

                cmd = [
                    self.ffmpeg_path,
                    "-i", str(video.path),
                    "-vf", f"fps={sample_rate}",
                    "-frames:v", str(max_frames),
                    "-q:v", "2",
                    output_pattern,
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=120,
                )

                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr.decode()}")
                    return []

                # Load extracted frames
                frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

                for i, frame_path in enumerate(frame_files):
                    timestamp = i / sample_rate
                    image = ImageInput.from_file(frame_path)

                    frames.append(
                        VideoFrame(
                            index=i,
                            timestamp=timestamp,
                            image=image,
                        )
                    )

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")

        return frames

    def detect_scenes(self, video: VideoInput) -> VideoAnalysisResult:
        """Detect scene changes using FFmpeg."""
        if not self.is_available():
            return VideoAnalysisResult(
                metadata={"error": "ffmpeg_not_found"},
            )

        try:
            # Use FFmpeg's scene detection
            cmd = [
                self.ffmpeg_path,
                "-i", str(video.path),
                "-vf", "select='gt(scene,0.3)',showinfo",
                "-f", "null",
                "-",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Parse scene change timestamps from stderr
            import re

            scenes = []
            pattern = r"pts_time:(\d+\.?\d*)"
            matches = re.findall(pattern, result.stderr)

            prev_time = 0.0
            for i, match in enumerate(matches):
                scene_time = float(match)
                scenes.append(
                    VideoSegment(
                        start=prev_time,
                        end=scene_time,
                        label=f"scene_{i + 1}",
                    )
                )
                prev_time = scene_time

            # Add final segment
            if video.metadata.duration > prev_time:
                scenes.append(
                    VideoSegment(
                        start=prev_time,
                        end=video.metadata.duration,
                        label=f"scene_{len(scenes) + 1}",
                    )
                )

            return VideoAnalysisResult(
                segments=scenes,
                model="ffmpeg",
            )

        except Exception as e:
            logger.error(f"Scene detection error: {e}")
            return VideoAnalysisResult(
                metadata={"error": str(e)},
            )

    def summarize(
        self,
        video: VideoInput,
        max_length: int = 200,
    ) -> VideoAnalysisResult:
        """Generate summary by analyzing key frames."""
        # Extract fewer frames for summary
        frames = self.extract_frames(video, sample_rate=0.2, max_frames=5)

        if not frames:
            return VideoAnalysisResult(
                summary="Could not analyze video",
            )

        # Get descriptions
        descriptions = []
        for frame in frames:
            desc = self.describe_frame(frame)
            descriptions.append(desc)

        summary = self._synthesize_description(descriptions)
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."

        result = VideoAnalysisResult(
            summary=summary,
            key_frames=frames,
            model="ffmpeg+vision",
        )

        self._notify_callbacks(result)
        return result

    def _synthesize_description(self, descriptions: List[str]) -> str:
        """Combine frame descriptions into overall description."""
        if not descriptions:
            return "No visual content analyzed."

        if len(descriptions) == 1:
            return descriptions[0]

        # Simple approach: concatenate unique descriptions
        unique = []
        for desc in descriptions:
            if desc not in unique:
                unique.append(desc)

        if len(unique) == 1:
            return unique[0]

        return "The video shows: " + "; ".join(unique[:3])

    def _detect_scene_changes(self, frames: List[VideoFrame]) -> List[VideoSegment]:
        """Detect scene changes based on frame similarity."""
        if len(frames) < 2:
            if frames:
                return [
                    VideoSegment(
                        start=0.0,
                        end=frames[0].timestamp + 1.0,
                        label="scene_1",
                    )
                ]
            return []

        scenes = []
        scene_start = frames[0].timestamp

        for i in range(1, len(frames)):
            # For now, create segments based on time intervals
            # Could use embedding similarity with vision engine
            if i % 3 == 0:  # Simple threshold
                scenes.append(
                    VideoSegment(
                        start=scene_start,
                        end=frames[i].timestamp,
                        label=f"scene_{len(scenes) + 1}",
                    )
                )
                scene_start = frames[i].timestamp

        # Add final scene
        scenes.append(
            VideoSegment(
                start=scene_start,
                end=frames[-1].timestamp + 1.0,
                label=f"scene_{len(scenes) + 1}",
            )
        )

        return scenes

    def is_available(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False


# =============================================================================
# Factory Function
# =============================================================================


def create_video_analyzer(
    analyzer_type: str = "auto",
    vision_engine: Optional[VisionEngine] = None,
    **kwargs,
) -> VideoAnalyzer:
    """
    Create a video analyzer.

    Args:
        analyzer_type: Type of analyzer ("ffmpeg", "mock", "auto")
        vision_engine: Vision engine for frame analysis
        **kwargs: Additional analyzer-specific arguments

    Returns:
        VideoAnalyzer instance
    """
    if analyzer_type == "mock":
        return MockVideoAnalyzer(vision_engine)

    if analyzer_type == "ffmpeg":
        return FFmpegVideoAnalyzer(vision_engine, **kwargs)

    # Auto-detect best available analyzer
    if analyzer_type == "auto":
        ffmpeg = FFmpegVideoAnalyzer(vision_engine)
        if ffmpeg.is_available():
            logger.info("Using FFmpeg for video analysis")
            return ffmpeg

        logger.warning("FFmpeg not available, using mock video analyzer")
        return MockVideoAnalyzer(vision_engine)

    raise ValueError(f"Unknown video analyzer type: {analyzer_type}")
