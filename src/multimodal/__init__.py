"""
Agent OS Multi-Modal Agents Module

Provides multi-modal processing capabilities including:
- Vision processing (LLaVA, CLIP, image understanding)
- Audio processing (speech, music, sound analysis)
- Video analysis (temporal understanding, action recognition)
- Multi-modal agent integration

Usage:
    from src.multimodal import MultiModalAgent, create_multimodal_agent

    agent = create_multimodal_agent()
    result = agent.process_image("path/to/image.jpg", "Describe this image")
"""

from .agent import (
    MultiModalAgent,
    MultiModalConfig,
    MultiModalInput,
    MultiModalResult,
    create_multimodal_agent,
)
from .audio import (
    AudioAnalysisResult,
    AudioAnalyzer,
    AudioFeatures,
    create_audio_analyzer,
)
from .video import (
    VideoAnalysisResult,
    VideoAnalyzer,
    VideoFrame,
    create_video_analyzer,
)
from .vision import (
    CLIPEncoder,
    ImageInput,
    LLaVAVision,
    VisionEngine,
    VisionResult,
    create_vision_engine,
)

__all__ = [
    # Vision
    "VisionEngine",
    "VisionResult",
    "ImageInput",
    "CLIPEncoder",
    "LLaVAVision",
    "create_vision_engine",
    # Audio
    "AudioAnalyzer",
    "AudioAnalysisResult",
    "AudioFeatures",
    "create_audio_analyzer",
    # Video
    "VideoAnalyzer",
    "VideoAnalysisResult",
    "VideoFrame",
    "create_video_analyzer",
    # Agent
    "MultiModalAgent",
    "MultiModalConfig",
    "MultiModalInput",
    "MultiModalResult",
    "create_multimodal_agent",
]
