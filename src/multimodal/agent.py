"""
Multi-Modal Agent Integration

Provides a unified agent that can process multiple modalities:
- Images (vision)
- Audio (speech, music, sounds)
- Video (temporal visual understanding)
- Text (combined with other modalities)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .audio import (
    AudioAnalysisResult,
    AudioAnalyzer,
    AudioInput,
    create_audio_analyzer,
)
from .video import (
    VideoAnalysisResult,
    VideoAnalyzer,
    VideoInput,
    create_video_analyzer,
)
from .vision import (
    ImageInput,
    VisionEngine,
    VisionResult,
    create_vision_engine,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class InputModality(str, Enum):
    """Input modality types."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class TaskType(str, Enum):
    """Multi-modal task types."""

    DESCRIBE = "describe"
    CLASSIFY = "classify"
    EMBED = "embed"
    QUESTION_ANSWER = "question_answer"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    DETECT = "detect"
    ANALYZE = "analyze"


@dataclass
class MultiModalInput:
    """
    Unified input container for multi-modal data.
    """

    text: Optional[str] = None
    images: List[ImageInput] = field(default_factory=list)
    audio: Optional[AudioInput] = None
    video: Optional[VideoInput] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> List[InputModality]:
        """Get list of present modalities."""
        mods = []
        if self.text:
            mods.append(InputModality.TEXT)
        if self.images:
            mods.append(InputModality.IMAGE)
        if self.audio:
            mods.append(InputModality.AUDIO)
        if self.video:
            mods.append(InputModality.VIDEO)
        return mods

    @property
    def is_multimodal(self) -> bool:
        """Check if input has multiple modalities."""
        return len(self.modalities) > 1

    @classmethod
    def from_image(cls, path: Union[str, Path]) -> "MultiModalInput":
        """Create from image file."""
        return cls(images=[ImageInput.from_file(path)])

    @classmethod
    def from_audio(cls, path: Union[str, Path]) -> "MultiModalInput":
        """Create from audio file."""
        return cls(audio=AudioInput.from_file(path))

    @classmethod
    def from_video(cls, path: Union[str, Path]) -> "MultiModalInput":
        """Create from video file."""
        return cls(video=VideoInput.from_file(path))

    @classmethod
    def from_text(cls, text: str) -> "MultiModalInput":
        """Create from text."""
        return cls(text=text)


@dataclass
class MultiModalResult:
    """
    Unified result from multi-modal processing.
    """

    description: str = ""
    answer: str = ""
    labels: List[str] = field(default_factory=list)
    confidence: float = 1.0
    embeddings: Optional[List[float]] = None

    # Modality-specific results
    vision_result: Optional[VisionResult] = None
    audio_result: Optional[AudioAnalysisResult] = None
    video_result: Optional[VideoAnalysisResult] = None

    # Processing metadata
    modalities_processed: List[InputModality] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_embeddings(self) -> bool:
        """Check if embeddings are available."""
        return self.embeddings is not None and len(self.embeddings) > 0


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal agent."""

    # Vision settings
    vision_engine: str = "auto"
    vision_model: Optional[str] = None

    # Audio settings
    audio_analyzer: str = "auto"
    audio_model: Optional[str] = None

    # Video settings
    video_analyzer: str = "auto"
    video_sample_rate: float = 1.0

    # Processing settings
    max_image_size: int = 1024
    max_audio_duration: float = 300.0
    max_video_duration: float = 600.0

    # Output settings
    combine_modalities: bool = True
    include_embeddings: bool = False


# =============================================================================
# Multi-Modal Agent
# =============================================================================


class MultiModalAgent:
    """
    Agent that can process multiple input modalities.

    Combines vision, audio, and video analysis capabilities
    into a unified interface.
    """

    def __init__(
        self,
        config: Optional[MultiModalConfig] = None,
        vision_engine: Optional[VisionEngine] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        video_analyzer: Optional[VideoAnalyzer] = None,
    ):
        self.config = config or MultiModalConfig()

        # Initialize engines
        self._vision = vision_engine
        self._audio = audio_analyzer
        self._video = video_analyzer

        # Callbacks
        self._result_callbacks: List[Callable[[MultiModalResult], None]] = []

        # Stats
        self._processed_count = 0
        self._total_processing_time = 0.0

    def _ensure_vision(self) -> VisionEngine:
        """Ensure vision engine is initialized."""
        if not self._vision:
            self._vision = create_vision_engine(self.config.vision_engine)
        return self._vision

    def _ensure_audio(self) -> AudioAnalyzer:
        """Ensure audio analyzer is initialized."""
        if not self._audio:
            self._audio = create_audio_analyzer(self.config.audio_analyzer)
        return self._audio

    def _ensure_video(self) -> VideoAnalyzer:
        """Ensure video analyzer is initialized."""
        if not self._video:
            self._video = create_video_analyzer(
                self.config.video_analyzer,
                vision_engine=self._ensure_vision(),
            )
        return self._video

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_result(self, callback: Callable[[MultiModalResult], None]) -> None:
        """Register callback for processing results."""
        self._result_callbacks.append(callback)

    def _notify_callbacks(self, result: MultiModalResult) -> None:
        """Notify all registered callbacks."""
        for callback in self._result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"MultiModal callback error: {e}")

    # -------------------------------------------------------------------------
    # Main Processing
    # -------------------------------------------------------------------------

    def process(
        self,
        mm_input: MultiModalInput,
        task: TaskType = TaskType.DESCRIBE,
        prompt: Optional[str] = None,
    ) -> MultiModalResult:
        """
        Process multi-modal input.

        Args:
            mm_input: Multi-modal input data
            task: Type of task to perform
            prompt: Optional prompt/question

        Returns:
            MultiModalResult with combined results
        """
        start_time = time.time()

        results = MultiModalResult()
        results.modalities_processed = mm_input.modalities.copy()

        # Process each modality
        if InputModality.IMAGE in mm_input.modalities:
            results.vision_result = self._process_images(mm_input.images, task, prompt)

        if InputModality.AUDIO in mm_input.modalities:
            results.audio_result = self._process_audio(mm_input.audio, task)

        if InputModality.VIDEO in mm_input.modalities:
            results.video_result = self._process_video(mm_input.video, task, prompt)

        # Combine results
        if self.config.combine_modalities:
            self._combine_results(results, mm_input.text)

        results.processing_time = time.time() - start_time

        # Update stats
        self._processed_count += 1
        self._total_processing_time += results.processing_time

        self._notify_callbacks(results)
        return results

    def _process_images(
        self,
        images: List[ImageInput],
        task: TaskType,
        prompt: Optional[str],
    ) -> Optional[VisionResult]:
        """Process images."""
        if not images:
            return None

        vision = self._ensure_vision()

        # Process first image (could extend to handle multiple)
        image = images[0]

        if task == TaskType.DESCRIBE or task == TaskType.QUESTION_ANSWER:
            return vision.describe(image, prompt)
        elif task == TaskType.EMBED:
            return vision.embed(image)
        elif task == TaskType.DETECT:
            return vision.detect_objects(image)
        else:
            return vision.describe(image, prompt)

    def _process_audio(
        self,
        audio: Optional[AudioInput],
        task: TaskType,
    ) -> Optional[AudioAnalysisResult]:
        """Process audio."""
        if not audio:
            return None

        analyzer = self._ensure_audio()

        if task == TaskType.EMBED:
            features = analyzer.embed(audio)
            return AudioAnalysisResult(features=features)
        elif task == TaskType.CLASSIFY:
            return analyzer.classify(audio)
        elif task == TaskType.DETECT:
            return analyzer.detect_events(audio)
        else:
            return analyzer.analyze(audio)

    def _process_video(
        self,
        video: Optional[VideoInput],
        task: TaskType,
        prompt: Optional[str],
    ) -> Optional[VideoAnalysisResult]:
        """Process video."""
        if not video:
            return None

        analyzer = self._ensure_video()

        if task == TaskType.SUMMARIZE:
            return analyzer.summarize(video)
        elif task == TaskType.DETECT:
            return analyzer.detect_scenes(video)
        else:
            return analyzer.analyze(video, self.config.video_sample_rate)

    def _combine_results(
        self,
        results: MultiModalResult,
        text: Optional[str],
    ) -> None:
        """Combine results from different modalities."""
        descriptions = []

        if results.vision_result:
            descriptions.append(results.vision_result.description)
            results.confidence = results.vision_result.confidence
            if results.vision_result.labels:
                results.labels.extend(results.vision_result.labels)
            if self.config.include_embeddings and results.vision_result.embeddings:
                results.embeddings = results.vision_result.embeddings

        if results.audio_result:
            if results.audio_result.labels:
                descriptions.append(f"Audio: {', '.join(results.audio_result.labels)}")
                results.labels.extend(results.audio_result.labels)
            if results.audio_result.transcription:
                descriptions.append(f"Speech: {results.audio_result.transcription}")

        if results.video_result:
            if results.video_result.description:
                descriptions.append(results.video_result.description)
            elif results.video_result.summary:
                descriptions.append(results.video_result.summary)

        # Combine descriptions
        results.description = " ".join(descriptions)

        # If there was a text prompt, use description as answer
        if text:
            results.answer = results.description

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def describe_image(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
    ) -> MultiModalResult:
        """Describe an image."""
        mm_input = MultiModalInput.from_image(image_path)
        if prompt:
            mm_input.text = prompt
        return self.process(mm_input, TaskType.DESCRIBE, prompt)

    def describe_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
    ) -> MultiModalResult:
        """Describe a video."""
        mm_input = MultiModalInput.from_video(video_path)
        if prompt:
            mm_input.text = prompt
        return self.process(mm_input, TaskType.DESCRIBE, prompt)

    def analyze_audio(self, audio_path: Union[str, Path]) -> MultiModalResult:
        """Analyze audio file."""
        mm_input = MultiModalInput.from_audio(audio_path)
        return self.process(mm_input, TaskType.ANALYZE)

    def answer_question(
        self,
        mm_input: MultiModalInput,
        question: str,
    ) -> str:
        """Answer a question about multi-modal input."""
        result = self.process(mm_input, TaskType.QUESTION_ANSWER, question)
        return result.answer or result.description

    def classify(
        self,
        mm_input: MultiModalInput,
        labels: Optional[List[str]] = None,
    ) -> MultiModalResult:
        """Classify multi-modal input."""
        if labels and InputModality.IMAGE in mm_input.modalities:
            # Use CLIP-style classification for images
            vision = self._ensure_vision()
            result = vision.classify(mm_input.images[0], labels)
            return MultiModalResult(
                labels=result.labels,
                confidence=result.confidence,
                vision_result=result,
            )

        return self.process(mm_input, TaskType.CLASSIFY)

    def embed(self, mm_input: MultiModalInput) -> MultiModalResult:
        """Generate embeddings for input."""
        result = self.process(mm_input, TaskType.EMBED)

        # Collect embeddings from all modalities
        all_embeddings = []

        if result.vision_result and result.vision_result.embeddings:
            all_embeddings.extend(result.vision_result.embeddings)

        if result.audio_result and result.audio_result.features.embeddings:
            all_embeddings.extend(result.audio_result.features.embeddings)

        result.embeddings = all_embeddings or None
        return result

    def compare(
        self,
        input1: MultiModalInput,
        input2: MultiModalInput,
    ) -> float:
        """
        Compare two inputs by embedding similarity.

        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embed(input1)
        emb2 = self.embed(input2)

        if not emb1.has_embeddings or not emb2.has_embeddings:
            return 0.0

        # Cosine similarity
        import math

        dot = sum(a * b for a, b in zip(emb1.embeddings, emb2.embeddings))
        norm1 = math.sqrt(sum(a * a for a in emb1.embeddings))
        norm2 = math.sqrt(sum(b * b for b in emb2.embeddings))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if at least one modality processor is available."""
        return any(
            [
                self._ensure_vision().is_available(),
                self._ensure_audio().is_available(),
                self._ensure_video().is_available(),
            ]
        )

    def get_capabilities(self) -> Dict[str, bool]:
        """Get available capabilities."""
        return {
            "vision": self._ensure_vision().is_available(),
            "audio": self._ensure_audio().is_available(),
            "video": self._ensure_video().is_available(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_count": self._processed_count,
            "total_processing_time": self._total_processing_time,
            "average_processing_time": (
                self._total_processing_time / self._processed_count
                if self._processed_count > 0
                else 0.0
            ),
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_multimodal_agent(
    config: Optional[MultiModalConfig] = None,
    **kwargs,
) -> MultiModalAgent:
    """
    Create a multi-modal agent.

    Args:
        config: Agent configuration
        **kwargs: Additional engine overrides

    Returns:
        MultiModalAgent instance
    """
    return MultiModalAgent(
        config=config,
        vision_engine=kwargs.get("vision_engine"),
        audio_analyzer=kwargs.get("audio_analyzer"),
        video_analyzer=kwargs.get("video_analyzer"),
    )
