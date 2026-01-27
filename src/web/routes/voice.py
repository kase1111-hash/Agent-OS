"""
Voice API Routes

REST and WebSocket endpoints for Speech-to-Text (STT) and Text-to-Speech (TTS) functionality.
"""

import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Models
# =============================================================================


class STTModelSize(str, Enum):
    """Available STT model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class STTLanguageCode(str, Enum):
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
    AUTO = "auto"


class TTSVoiceId(str, Enum):
    """Available TTS voices."""

    EN_US_AMY = "en_US-amy-medium"
    EN_US_DANNY = "en_US-danny-low"
    EN_US_LESSAC = "en_US-lessac-medium"
    EN_GB_ALAN = "en_GB-alan-medium"
    ES_ES_DAVEFX = "es_ES-davefx-medium"
    FR_FR_SIWIS = "fr_FR-siwis-medium"
    DE_DE_THORSTEN = "de_DE-thorsten-medium"
    DEFAULT = "en_US-lessac-medium"


class TranscriptionRequest(BaseModel):
    """Request for transcription with base64-encoded audio."""

    audio_data: str = Field(..., description="Base64-encoded audio data")
    audio_format: str = Field(default="wav", description="Audio format (wav, mp3, ogg, pcm)")
    language: STTLanguageCode = Field(default=STTLanguageCode.AUTO, description="Language code")
    translate: bool = Field(default=False, description="Translate to English")


class TranscriptionResponse(BaseModel):
    """Response from transcription."""

    text: str
    language: str
    duration: float
    processing_time_ms: int
    confidence: float
    segments: List[Dict[str, Any]] = Field(default_factory=list)


class SynthesisRequest(BaseModel):
    """Request for speech synthesis."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: TTSVoiceId = Field(default=TTSVoiceId.DEFAULT, description="Voice to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech rate")
    output_format: str = Field(default="wav", description="Output format (wav, mp3)")


class SynthesisResponse(BaseModel):
    """Response from synthesis (metadata only)."""

    audio_data: str = Field(..., description="Base64-encoded audio data")
    format: str
    duration: float
    processing_time_ms: int
    sample_rate: int


class VoiceStatus(BaseModel):
    """Status of voice capabilities."""

    stt_available: bool
    stt_engine: str
    tts_available: bool
    tts_engine: str
    available_voices: List[str]


class STTConfigUpdate(BaseModel):
    """Update STT configuration."""

    model: Optional[STTModelSize] = None
    language: Optional[STTLanguageCode] = None
    translate: Optional[bool] = None


class TTSConfigUpdate(BaseModel):
    """Update TTS configuration."""

    voice: Optional[TTSVoiceId] = None
    speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)


# =============================================================================
# Voice Engine Manager
# =============================================================================


class VoiceManager:
    """Manages STT and TTS engines."""

    def __init__(self):
        self._stt_engine = None
        self._tts_engine = None
        self._stt_config = None
        self._tts_config = None

    def get_stt_engine(self):
        """Get or create STT engine."""
        if self._stt_engine is None:
            try:
                from src.voice import create_stt_engine
                from src.voice.stt import STTConfig, STTLanguage, STTModel

                self._stt_config = STTConfig()
                self._stt_engine = create_stt_engine("auto", self._stt_config)
            except ImportError as e:
                logger.warning(f"Voice module not fully available: {e}")
                return None
        return self._stt_engine

    def get_tts_engine(self):
        """Get or create TTS engine."""
        if self._tts_engine is None:
            try:
                from src.voice import create_tts_engine
                from src.voice.tts import TTSConfig

                self._tts_config = TTSConfig()
                self._tts_engine = create_tts_engine("auto", self._tts_config)
            except ImportError as e:
                logger.warning(f"Voice module not fully available: {e}")
                return None
        return self._tts_engine

    def get_status(self) -> Dict[str, Any]:
        """Get voice capabilities status."""
        stt = self.get_stt_engine()
        tts = self.get_tts_engine()

        return {
            "stt_available": stt is not None and stt.is_available(),
            "stt_engine": type(stt).__name__ if stt else "None",
            "tts_available": tts is not None and tts.is_available(),
            "tts_engine": type(tts).__name__ if tts else "None",
            "available_voices": tts.list_voices() if tts and tts.is_available() else [],
        }

    def update_stt_config(self, model: str = None, language: str = None, translate: bool = None):
        """Update STT configuration and recreate engine."""
        try:
            from src.voice import create_stt_engine
            from src.voice.stt import STTConfig, STTLanguage, STTModel

            current = self._stt_config or STTConfig()

            new_config = STTConfig(
                model=STTModel(model) if model else current.model,
                language=STTLanguage(language) if language else current.language,
                translate=translate if translate is not None else current.translate,
            )

            self._stt_config = new_config
            self._stt_engine = create_stt_engine("auto", new_config)
            return True
        except Exception as e:
            logger.error(f"Failed to update STT config: {e}")
            return False

    def update_tts_config(self, voice: str = None, speed: float = None):
        """Update TTS configuration and recreate engine."""
        try:
            from src.voice import create_tts_engine
            from src.voice.tts import TTSConfig, TTSVoice

            current = self._tts_config or TTSConfig()

            new_config = TTSConfig(
                voice=TTSVoice(voice) if voice else current.voice,
                speed=speed if speed is not None else current.speed,
            )

            self._tts_config = new_config
            self._tts_engine = create_tts_engine("auto", new_config)
            return True
        except Exception as e:
            logger.error(f"Failed to update TTS config: {e}")
            return False


# Global voice manager
_voice_manager: Optional[VoiceManager] = None


def get_voice_manager() -> VoiceManager:
    """Get the voice manager instance."""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager()
    return _voice_manager


# =============================================================================
# REST Endpoints
# =============================================================================


@router.get("/status", response_model=VoiceStatus)
async def get_voice_status(
    manager: VoiceManager = Depends(get_voice_manager),
) -> VoiceStatus:
    """Get voice system status and capabilities."""
    status = manager.get_status()
    return VoiceStatus(**status)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    request: TranscriptionRequest,
    manager: VoiceManager = Depends(get_voice_manager),
) -> TranscriptionResponse:
    """
    Transcribe audio to text using STT engine.

    Accepts base64-encoded audio data in various formats.
    """
    stt = manager.get_stt_engine()
    if stt is None or not stt.is_available():
        raise HTTPException(
            status_code=503,
            detail="STT engine not available. Install whisper.cpp or set OPENAI_API_KEY.",
        )

    start_time = time.time()

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
    except Exception as e:
        logger.warning(f"Invalid base64 audio data: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    # Determine audio format
    from src.voice.audio import AudioFormat

    format_map = {
        "wav": AudioFormat.WAV,
        "pcm": AudioFormat.PCM,
        "mp3": AudioFormat.MP3,
        "ogg": AudioFormat.OGG,
    }
    audio_format = format_map.get(request.audio_format.lower(), AudioFormat.WAV)

    # Run transcription in thread pool (may be blocking)
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: stt.transcribe(audio_bytes, format=audio_format),
        )

    processing_time = int((time.time() - start_time) * 1000)

    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        duration=result.duration,
        processing_time_ms=processing_time,
        confidence=result.confidence,
        segments=[
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
            }
            for seg in result.segments
        ],
    )


@router.post("/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: STTLanguageCode = Form(default=STTLanguageCode.AUTO),
    translate: bool = Form(default=False),
    manager: VoiceManager = Depends(get_voice_manager),
) -> TranscriptionResponse:
    """
    Transcribe an uploaded audio file to text.

    Supports WAV, MP3, OGG, and other common audio formats.
    """
    stt = manager.get_stt_engine()
    if stt is None or not stt.is_available():
        raise HTTPException(
            status_code=503,
            detail="STT engine not available. Install whisper.cpp or set OPENAI_API_KEY.",
        )

    start_time = time.time()

    # Read uploaded file
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Determine format from filename
    ext = Path(file.filename or "audio.wav").suffix.lower()
    from src.voice.audio import AudioFormat

    format_map = {
        ".wav": AudioFormat.WAV,
        ".mp3": AudioFormat.MP3,
        ".ogg": AudioFormat.OGG,
        ".pcm": AudioFormat.PCM,
    }
    audio_format = format_map.get(ext, AudioFormat.WAV)

    # Run transcription in thread pool
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: stt.transcribe(content, format=audio_format),
        )

    processing_time = int((time.time() - start_time) * 1000)

    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        duration=result.duration,
        processing_time_ms=processing_time,
        confidence=result.confidence,
        segments=[
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
            }
            for seg in result.segments
        ],
    )


@router.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_speech(
    request: SynthesisRequest,
    manager: VoiceManager = Depends(get_voice_manager),
) -> SynthesisResponse:
    """
    Synthesize text to speech.

    Returns base64-encoded audio data.
    """
    tts = manager.get_tts_engine()
    if tts is None or not tts.is_available():
        raise HTTPException(
            status_code=503,
            detail="TTS engine not available. Install Piper or espeak-ng.",
        )

    start_time = time.time()

    # Run synthesis in thread pool
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: tts.synthesize(request.text),
        )

    if result.is_empty:
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {result.metadata.get('error', 'unknown error')}",
        )

    processing_time = int((time.time() - start_time) * 1000)

    return SynthesisResponse(
        audio_data=base64.b64encode(result.audio_data).decode("ascii"),
        format=result.format.value,
        duration=result.duration,
        processing_time_ms=processing_time,
        sample_rate=result.sample_rate,
    )


@router.post("/synthesize/stream")
async def synthesize_speech_stream(
    request: SynthesisRequest,
    manager: VoiceManager = Depends(get_voice_manager),
) -> StreamingResponse:
    """
    Synthesize text to speech and stream audio directly.

    Returns audio as a binary stream with appropriate Content-Type.
    """
    tts = manager.get_tts_engine()
    if tts is None or not tts.is_available():
        raise HTTPException(
            status_code=503,
            detail="TTS engine not available. Install Piper or espeak-ng.",
        )

    # Run synthesis in thread pool
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: tts.synthesize(request.text),
        )

    if result.is_empty:
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {result.metadata.get('error', 'unknown error')}",
        )

    # Stream the audio data
    content_type = "audio/wav" if request.output_format == "wav" else "audio/mpeg"

    return StreamingResponse(
        io.BytesIO(result.audio_data),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{request.output_format}",
            "X-Duration": str(result.duration),
            "X-Sample-Rate": str(result.sample_rate),
        },
    )


@router.get("/voices")
async def list_voices(
    manager: VoiceManager = Depends(get_voice_manager),
) -> Dict[str, Any]:
    """List available TTS voices."""
    tts = manager.get_tts_engine()
    if tts is None:
        return {"voices": [], "available": False}

    voices = tts.list_voices() if tts.is_available() else []
    return {
        "voices": voices,
        "available": tts.is_available(),
        "engine": type(tts).__name__,
    }


@router.put("/config/stt")
async def update_stt_config(
    config: STTConfigUpdate,
    manager: VoiceManager = Depends(get_voice_manager),
) -> Dict[str, Any]:
    """Update STT engine configuration."""
    success = manager.update_stt_config(
        model=config.model.value if config.model else None,
        language=config.language.value if config.language else None,
        translate=config.translate,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update STT configuration")

    return {
        "status": "updated",
        "config": {
            "model": config.model.value if config.model else None,
            "language": config.language.value if config.language else None,
            "translate": config.translate,
        },
    }


@router.put("/config/tts")
async def update_tts_config(
    config: TTSConfigUpdate,
    manager: VoiceManager = Depends(get_voice_manager),
) -> Dict[str, Any]:
    """Update TTS engine configuration."""
    success = manager.update_tts_config(
        voice=config.voice.value if config.voice else None,
        speed=config.speed,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update TTS configuration")

    return {
        "status": "updated",
        "config": {
            "voice": config.voice.value if config.voice else None,
            "speed": config.speed,
        },
    }


# =============================================================================
# WebSocket Endpoint for Real-time Voice
# =============================================================================


@router.websocket("/ws")
async def voice_websocket(
    websocket: WebSocket,
    manager: VoiceManager = Depends(get_voice_manager),
):
    """
    WebSocket endpoint for real-time voice interaction.

    Protocol:
    - Send: {"type": "transcribe", "audio": "<base64>", "format": "wav"}
    - Receive: {"type": "transcription", "text": "...", "confidence": 0.95}

    - Send: {"type": "synthesize", "text": "Hello world"}
    - Receive: {"type": "audio", "data": "<base64>", "format": "wav"}

    - Send: {"type": "ping"}
    - Receive: {"type": "pong"}
    """
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    logger.info(f"Voice WebSocket connected: {connection_id}")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await websocket.send_json(
                    {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            elif msg_type == "transcribe":
                stt = manager.get_stt_engine()
                if stt is None or not stt.is_available():
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "STT engine not available",
                        }
                    )
                    continue

                try:
                    audio_b64 = data.get("audio", "")
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_format_str = data.get("format", "wav")

                    from src.voice.audio import AudioFormat

                    format_map = {
                        "wav": AudioFormat.WAV,
                        "pcm": AudioFormat.PCM,
                        "mp3": AudioFormat.MP3,
                    }
                    audio_format = format_map.get(audio_format_str, AudioFormat.WAV)

                    # Run in thread pool
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor,
                            lambda: stt.transcribe(audio_bytes, format=audio_format),
                        )

                    await websocket.send_json(
                        {
                            "type": "transcription",
                            "text": result.text,
                            "language": result.language,
                            "confidence": result.confidence,
                            "duration": result.duration,
                        }
                    )

                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Transcription failed: {e}",
                        }
                    )

            elif msg_type == "synthesize":
                tts = manager.get_tts_engine()
                if tts is None or not tts.is_available():
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "TTS engine not available",
                        }
                    )
                    continue

                try:
                    text = data.get("text", "")
                    if not text:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "No text provided",
                            }
                        )
                        continue

                    # Run in thread pool
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor,
                            lambda: tts.synthesize(text),
                        )

                    if result.is_empty:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": f"Synthesis failed: {result.metadata.get('error', 'unknown')}",
                            }
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "audio",
                                "data": base64.b64encode(result.audio_data).decode("ascii"),
                                "format": result.format.value,
                                "duration": result.duration,
                                "sample_rate": result.sample_rate,
                            }
                        )

                except Exception as e:
                    logger.error(f"Synthesis error: {e}")
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Synthesis failed: {e}",
                        }
                    )

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }
                )

    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
