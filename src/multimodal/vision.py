"""
Vision Processing Module

Provides image understanding capabilities using:
- CLIP: Contrastive Language-Image Pre-training for embeddings
- LLaVA: Large Language and Vision Assistant for visual reasoning
- Local and API-based vision models
"""

import base64
import io
import ipaddress
import json
import logging
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# =============================================================================
# Security: URL Validation for SSRF Prevention
# =============================================================================

from src.core.exceptions import SSRFProtectionError


def validate_api_url(url: str, allow_localhost: bool = True) -> str:
    """
    Validate a URL to prevent Server-Side Request Forgery (SSRF) attacks.

    This function blocks requests to:
    - Internal/private IP ranges (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
    - Link-local addresses (169.254.x.x)
    - Loopback addresses (127.x.x.x) unless allow_localhost is True
    - IPv6 internal addresses
    - Non-HTTP(S) schemes

    Args:
        url: URL to validate
        allow_localhost: If True, allow localhost/127.0.0.1 (for local Ollama)

    Returns:
        The validated URL

    Raises:
        SSRFProtectionError: If the URL is potentially dangerous
    """
    if not url:
        raise SSRFProtectionError("URL cannot be empty")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SSRFProtectionError(f"Invalid URL format: {e}")

    # Only allow HTTP and HTTPS schemes
    if parsed.scheme not in ("http", "https"):
        raise SSRFProtectionError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")

    # Get the hostname
    hostname = parsed.hostname
    if not hostname:
        raise SSRFProtectionError("URL must have a valid hostname")

    # Check for localhost variants
    localhost_names = {"localhost", "localhost.localdomain", "127.0.0.1", "::1"}
    is_localhost = hostname.lower() in localhost_names

    if is_localhost:
        if allow_localhost:
            return url
        else:
            raise SSRFProtectionError("Localhost URLs are not allowed in this context")

    # Try to resolve hostname to IP and check if it's internal
    try:
        # Check if it's already an IP address
        ip = ipaddress.ip_address(hostname)
        _check_ip_safety(ip)
    except ValueError:
        # It's a hostname, not an IP - check for suspicious patterns
        # Block hostnames that might resolve to internal IPs
        suspicious_patterns = [
            r"^10\.",
            r"^192\.168\.",
            r"^172\.(1[6-9]|2[0-9]|3[0-1])\.",
            r"^127\.",
            r"^169\.254\.",
            r"^0\.",
            r"metadata",  # Cloud metadata services
            r"internal",
            r"\.local$",
            r"\.internal$",
            r"\.corp$",
            r"\.lan$",
        ]

        hostname_lower = hostname.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, hostname_lower):
                raise SSRFProtectionError(f"Hostname matches suspicious pattern: {hostname}")

    return url


def _check_ip_safety(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """
    Check if an IP address is safe (not internal/private).

    Raises:
        SSRFProtectionError: If the IP is internal/private
    """
    # Check for private/internal addresses
    if ip.is_private:
        raise SSRFProtectionError(f"Private IP addresses are not allowed: {ip}")

    if ip.is_loopback:
        raise SSRFProtectionError(f"Loopback addresses are not allowed: {ip}")

    if ip.is_link_local:
        raise SSRFProtectionError(f"Link-local addresses are not allowed: {ip}")

    if ip.is_multicast:
        raise SSRFProtectionError(f"Multicast addresses are not allowed: {ip}")

    if ip.is_reserved:
        raise SSRFProtectionError(f"Reserved addresses are not allowed: {ip}")

    # Check for IPv6-specific cases
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.is_site_local:
            raise SSRFProtectionError(f"Site-local IPv6 addresses are not allowed: {ip}")


# =============================================================================
# Models
# =============================================================================


class VisionModel(str, Enum):
    """Available vision models."""

    CLIP_VIT_B32 = "clip-vit-base-patch32"
    CLIP_VIT_L14 = "clip-vit-large-patch14"
    LLAVA_7B = "llava-v1.5-7b"
    LLAVA_13B = "llava-v1.5-13b"
    LLAVA_34B = "llava-v1.6-34b"
    GPT4_VISION = "gpt-4-vision-preview"
    CLAUDE_VISION = "claude-3-opus"


class ImageFormat(str, Enum):
    """Supported image formats."""

    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"


@dataclass
class BoundingBox:
    """Bounding box for detected objects."""

    x: float  # Left coordinate (0-1 normalized)
    y: float  # Top coordinate (0-1 normalized)
    width: float  # Width (0-1 normalized)
    height: float  # Height (0-1 normalized)
    label: str = ""
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        """Get area."""
        return self.width * self.height


@dataclass
class ImageInput:
    """Input image with metadata."""

    data: bytes  # Raw image bytes
    format: ImageFormat = ImageFormat.JPEG
    width: Optional[int] = None
    height: Optional[int] = None
    source: Optional[str] = None  # File path or URL
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ImageInput":
        """Load image from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Detect format from extension
        ext = path.suffix.lower().lstrip(".")
        format_map = {
            "jpg": ImageFormat.JPEG,
            "jpeg": ImageFormat.JPEG,
            "png": ImageFormat.PNG,
            "gif": ImageFormat.GIF,
            "webp": ImageFormat.WEBP,
            "bmp": ImageFormat.BMP,
            "tiff": ImageFormat.TIFF,
            "tif": ImageFormat.TIFF,
        }
        img_format = format_map.get(ext, ImageFormat.JPEG)

        with open(path, "rb") as f:
            data = f.read()

        return cls(
            data=data,
            format=img_format,
            source=str(path),
        )

    @classmethod
    def from_base64(cls, b64_string: str, image_format: ImageFormat = ImageFormat.JPEG) -> "ImageInput":
        """Create from base64 string."""
        data = base64.b64decode(b64_string)
        return cls(data=data, format=image_format)

    def to_base64(self) -> str:
        """Convert to base64 string."""
        return base64.b64encode(self.data).decode("utf-8")

    def get_dimensions(self) -> Tuple[int, int]:
        """Get image dimensions (width, height)."""
        if self.width and self.height:
            return (self.width, self.height)

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(self.data))
            return img.size
        except ImportError:
            logger.warning("PIL not installed, cannot determine dimensions")
            return (0, 0)


@dataclass
class VisionResult:
    """Result from vision processing."""

    description: str = ""
    embeddings: Optional[List[float]] = None
    objects: List[BoundingBox] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    confidence: float = 1.0
    model: str = ""
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_embeddings(self) -> bool:
        """Check if embeddings are available."""
        return self.embeddings is not None and len(self.embeddings) > 0

    @property
    def object_count(self) -> int:
        """Get number of detected objects."""
        return len(self.objects)


# =============================================================================
# Vision Engine Interface
# =============================================================================


class VisionEngine(ABC):
    """
    Abstract base class for vision processing engines.

    Implementations can use CLIP, LLaVA, or API-based models.
    """

    def __init__(self, model: VisionModel = VisionModel.CLIP_VIT_B32):
        self.model = model
        self._callbacks: List[Callable[[VisionResult], None]] = []

    @abstractmethod
    def describe(self, image: ImageInput, prompt: Optional[str] = None) -> VisionResult:
        """
        Generate a description of the image.

        Args:
            image: Input image
            prompt: Optional prompt for guided description

        Returns:
            VisionResult with description
        """
        pass

    @abstractmethod
    def embed(self, image: ImageInput) -> VisionResult:
        """
        Generate embeddings for the image.

        Args:
            image: Input image

        Returns:
            VisionResult with embeddings
        """
        pass

    @abstractmethod
    def classify(self, image: ImageInput, labels: List[str]) -> VisionResult:
        """
        Classify image against provided labels.

        Args:
            image: Input image
            labels: Candidate labels

        Returns:
            VisionResult with matched labels and confidences
        """
        pass

    @abstractmethod
    def detect_objects(self, image: ImageInput) -> VisionResult:
        """
        Detect objects in the image.

        Args:
            image: Input image

        Returns:
            VisionResult with detected objects and bounding boxes
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is available."""
        pass

    def answer_question(self, image: ImageInput, question: str) -> VisionResult:
        """
        Answer a question about the image (VQA).

        Args:
            image: Input image
            question: Question about the image

        Returns:
            VisionResult with answer in description
        """
        return self.describe(image, prompt=question)

    def compare_images(self, image1: ImageInput, image2: ImageInput) -> float:
        """
        Compare two images by cosine similarity of embeddings.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embed(image1)
        emb2 = self.embed(image2)

        if not emb1.has_embeddings or not emb2.has_embeddings:
            return 0.0

        # Compute cosine similarity
        import math

        dot_product = sum(a * b for a, b in zip(emb1.embeddings, emb2.embeddings))
        norm1 = math.sqrt(sum(a * a for a in emb1.embeddings))
        norm2 = math.sqrt(sum(b * b for b in emb2.embeddings))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def on_result(self, callback: Callable[[VisionResult], None]) -> None:
        """Register callback for vision results."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, result: VisionResult) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Vision callback error: {e}")


# =============================================================================
# Mock Vision Engine
# =============================================================================


class MockVisionEngine(VisionEngine):
    """
    Mock vision engine for testing.

    Returns configurable mock responses.
    """

    def __init__(self, model: VisionModel = VisionModel.CLIP_VIT_B32):
        super().__init__(model)
        self._descriptions: List[str] = ["A sample image description."]
        self._description_index = 0
        self._embedding_dim = 512

    def set_descriptions(self, descriptions: List[str]) -> None:
        """Set mock descriptions to return."""
        self._descriptions = descriptions
        self._description_index = 0

    def describe(self, image: ImageInput, prompt: Optional[str] = None) -> VisionResult:
        """Return mock description."""
        desc = self._descriptions[self._description_index % len(self._descriptions)]
        self._description_index += 1

        if prompt:
            desc = f"Response to '{prompt}': {desc}"

        result = VisionResult(
            description=desc,
            model=self.model.value,
            confidence=0.95,
        )
        self._notify_callbacks(result)
        return result

    def embed(self, image: ImageInput) -> VisionResult:
        """Return mock embeddings."""
        import random

        embeddings = [random.gauss(0, 1) for _ in range(self._embedding_dim)]

        # Normalize
        import math

        norm = math.sqrt(sum(e * e for e in embeddings))
        if norm > 0:
            embeddings = [e / norm for e in embeddings]

        result = VisionResult(
            embeddings=embeddings,
            model=self.model.value,
        )
        self._notify_callbacks(result)
        return result

    def classify(self, image: ImageInput, labels: List[str]) -> VisionResult:
        """Return mock classification."""
        import random

        # Random confidences
        scores = [random.random() for _ in labels]
        total = sum(scores)
        scores = [s / total for s in scores]

        # Sort by confidence
        label_scores = sorted(zip(labels, scores), key=lambda x: -x[1])

        result = VisionResult(
            labels=[l for l, _ in label_scores],
            confidence=label_scores[0][1] if label_scores else 0.0,
            model=self.model.value,
            metadata={"scores": {l: s for l, s in label_scores}},
        )
        self._notify_callbacks(result)
        return result

    def detect_objects(self, image: ImageInput) -> VisionResult:
        """Return mock object detection."""
        import random

        num_objects = random.randint(1, 5)
        objects = []

        for i in range(num_objects):
            objects.append(
                BoundingBox(
                    x=random.random() * 0.7,
                    y=random.random() * 0.7,
                    width=random.random() * 0.3,
                    height=random.random() * 0.3,
                    label=f"object_{i}",
                    confidence=random.random(),
                )
            )

        result = VisionResult(
            objects=objects,
            model=self.model.value,
        )
        self._notify_callbacks(result)
        return result

    def is_available(self) -> bool:
        """Mock is always available."""
        return True


# =============================================================================
# CLIP Encoder
# =============================================================================


class CLIPEncoder(VisionEngine):
    """
    Vision engine using OpenAI CLIP.

    CLIP provides image embeddings that can be compared with text embeddings
    for zero-shot classification.

    Install:
        pip install transformers torch pillow
    """

    def __init__(
        self,
        model: VisionModel = VisionModel.CLIP_VIT_B32,
        device: Optional[str] = None,
    ):
        super().__init__(model)
        self.device = device
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize CLIP model."""
        try:
            from transformers import CLIPModel, CLIPProcessor

            model_map = {
                VisionModel.CLIP_VIT_B32: "openai/clip-vit-base-patch32",
                VisionModel.CLIP_VIT_L14: "openai/clip-vit-large-patch14",
            }
            model_name = model_map.get(self.model, "openai/clip-vit-base-patch32")

            self._model = CLIPModel.from_pretrained(model_name)
            self._processor = CLIPProcessor.from_pretrained(model_name)

            if self.device:
                self._model = self._model.to(self.device)

            logger.info(f"CLIP initialized: {model_name}")

        except ImportError:
            logger.warning("transformers not installed for CLIP")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP: {e}")

    def _load_image(self, image: ImageInput):
        """Load image for processing."""
        from PIL import Image

        return Image.open(io.BytesIO(image.data))

    def describe(self, image: ImageInput, prompt: Optional[str] = None) -> VisionResult:
        """
        CLIP doesn't generate descriptions directly.
        For description, use with candidate captions.
        """
        if not self.is_available():
            return VisionResult(
                description="CLIP not available",
                metadata={"error": "model_not_loaded"},
            )

        if prompt:
            # Use prompt as candidate and return similarity
            result = self.classify(image, [prompt])
            return VisionResult(
                description=f"Match confidence: {result.confidence:.2%}",
                confidence=result.confidence,
                model=self.model.value,
            )

        return VisionResult(
            description="CLIP requires candidate labels for classification",
            model=self.model.value,
        )

    def embed(self, image: ImageInput) -> VisionResult:
        """Generate CLIP embeddings for image."""
        import time

        if not self.is_available():
            return VisionResult(metadata={"error": "model_not_loaded"})

        start_time = time.time()

        try:
            import torch

            pil_image = self._load_image(image)
            inputs = self._processor(images=pil_image, return_tensors="pt")

            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)

            # Normalize embeddings
            embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings = embeddings.squeeze().cpu().numpy().tolist()

            result = VisionResult(
                embeddings=embeddings,
                model=self.model.value,
                processing_time=time.time() - start_time,
            )
            self._notify_callbacks(result)
            return result

        except Exception as e:
            logger.error(f"CLIP embedding error: {e}")
            return VisionResult(metadata={"error": str(e)})

    def classify(self, image: ImageInput, labels: List[str]) -> VisionResult:
        """Zero-shot classification with CLIP."""
        import time

        if not self.is_available():
            return VisionResult(metadata={"error": "model_not_loaded"})

        start_time = time.time()

        try:
            import torch

            pil_image = self._load_image(image)

            inputs = self._processor(
                text=labels,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )

            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get similarity scores
            logits = outputs.logits_per_image.squeeze()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # Create sorted results
            label_scores = sorted(
                zip(labels, probs.tolist()),
                key=lambda x: -x[1],
            )

            result = VisionResult(
                labels=[l for l, _ in label_scores],
                confidence=label_scores[0][1] if label_scores else 0.0,
                model=self.model.value,
                processing_time=time.time() - start_time,
                metadata={"scores": {l: float(s) for l, s in label_scores}},
            )
            self._notify_callbacks(result)
            return result

        except Exception as e:
            logger.error(f"CLIP classification error: {e}")
            return VisionResult(metadata={"error": str(e)})

    def detect_objects(self, image: ImageInput) -> VisionResult:
        """CLIP doesn't support object detection."""
        return VisionResult(
            description="CLIP does not support object detection",
            model=self.model.value,
            metadata={"error": "not_supported"},
        )

    def is_available(self) -> bool:
        """Check if CLIP is loaded."""
        return self._model is not None and self._processor is not None


# =============================================================================
# LLaVA Vision
# =============================================================================


class LLaVAVision(VisionEngine):
    """
    Vision engine using LLaVA (Large Language and Vision Assistant).

    LLaVA provides visual reasoning and question answering capabilities.

    Install:
        pip install llava
        # Or use ollama with llava model
    """

    def __init__(
        self,
        model: VisionModel = VisionModel.LLAVA_7B,
        use_ollama: bool = True,
        api_base: Optional[str] = None,
    ):
        super().__init__(model)
        self.use_ollama = use_ollama

        # Validate api_base URL to prevent SSRF attacks
        # Allow localhost by default since Ollama typically runs locally
        raw_api_base = api_base or "http://localhost:11434"
        try:
            self.api_base = validate_api_url(raw_api_base, allow_localhost=True)
        except SSRFProtectionError as e:
            logger.error(f"Invalid API base URL: {e}")
            raise ValueError(f"Invalid api_base URL: {e}") from e

        self._model_name = self._get_ollama_model_name()

    def _get_ollama_model_name(self) -> str:
        """Get Ollama model name."""
        model_map = {
            VisionModel.LLAVA_7B: "llava",
            VisionModel.LLAVA_13B: "llava:13b",
            VisionModel.LLAVA_34B: "llava:34b",
        }
        return model_map.get(self.model, "llava")

    def _call_ollama(self, prompt: str, image: Optional[ImageInput] = None) -> Optional[str]:
        """Call Ollama API."""
        try:
            import requests

            url = f"{self.api_base}/api/generate"

            payload = {
                "model": self._model_name,
                "prompt": prompt,
                "stream": False,
            }

            if image:
                payload["images"] = [image.to_base64()]

            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            return response.json().get("response", "")

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None

    def describe(self, image: ImageInput, prompt: Optional[str] = None) -> VisionResult:
        """Generate description using LLaVA."""
        import time

        if not self.is_available():
            return VisionResult(
                description="LLaVA not available",
                metadata={"error": "model_not_available"},
            )

        start_time = time.time()

        if prompt:
            query = prompt
        else:
            query = "Describe this image in detail."

        response = self._call_ollama(query, image)

        if response:
            result = VisionResult(
                description=response,
                model=self.model.value,
                confidence=0.9,
                processing_time=time.time() - start_time,
            )
        else:
            result = VisionResult(
                description="Failed to get response",
                metadata={"error": "api_error"},
            )

        self._notify_callbacks(result)
        return result

    def embed(self, image: ImageInput) -> VisionResult:
        """LLaVA doesn't provide embeddings directly."""
        return VisionResult(
            description="LLaVA does not provide embeddings. Use CLIP instead.",
            model=self.model.value,
            metadata={"error": "not_supported"},
        )

    def classify(self, image: ImageInput, labels: List[str]) -> VisionResult:
        """Classify image by asking LLaVA."""
        import time

        if not self.is_available():
            return VisionResult(metadata={"error": "model_not_available"})

        start_time = time.time()

        labels_str = ", ".join(labels)
        prompt = f"Classify this image. Choose the most appropriate label from: {labels_str}. Respond with only the label name."

        response = self._call_ollama(prompt, image)

        if response:
            # Try to match response to one of the labels
            response_lower = response.lower().strip()
            matched_labels = []

            for label in labels:
                if label.lower() in response_lower:
                    matched_labels.append(label)

            result = VisionResult(
                labels=matched_labels or [response.strip()],
                confidence=0.8,
                model=self.model.value,
                processing_time=time.time() - start_time,
            )
        else:
            result = VisionResult(metadata={"error": "api_error"})

        self._notify_callbacks(result)
        return result

    def detect_objects(self, image: ImageInput) -> VisionResult:
        """Ask LLaVA to identify objects."""
        import time

        if not self.is_available():
            return VisionResult(metadata={"error": "model_not_available"})

        start_time = time.time()

        prompt = "List all objects you can see in this image. For each object, provide its approximate location (left/right, top/bottom)."

        response = self._call_ollama(prompt, image)

        if response:
            result = VisionResult(
                description=response,
                model=self.model.value,
                processing_time=time.time() - start_time,
            )
        else:
            result = VisionResult(metadata={"error": "api_error"})

        self._notify_callbacks(result)
        return result

    def is_available(self) -> bool:
        """Check if LLaVA is available via Ollama."""
        if not self.use_ollama:
            return False

        try:
            import requests

            response = requests.get(
                f"{self.api_base}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return any(self._model_name in name for name in model_names)
        except Exception:
            pass

        return False


# =============================================================================
# API Vision (OpenAI, Anthropic)
# =============================================================================


class APIVisionEngine(VisionEngine):
    """
    Vision engine using cloud APIs (OpenAI GPT-4 Vision, Claude).
    """

    def __init__(
        self,
        model: VisionModel = VisionModel.GPT4_VISION,
        api_key: Optional[str] = None,
    ):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def _call_openai_vision(self, image: ImageInput, prompt: str) -> Optional[str]:
        """Call OpenAI Vision API."""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image.format.value};base64,{image.to_base64()}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 1024,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            return None

    def describe(self, image: ImageInput, prompt: Optional[str] = None) -> VisionResult:
        """Generate description using API."""
        import time

        if not self.is_available():
            return VisionResult(
                description="API not available",
                metadata={"error": "no_api_key"},
            )

        start_time = time.time()

        query = prompt or "Describe this image in detail."
        response = self._call_openai_vision(image, query)

        if response:
            result = VisionResult(
                description=response,
                model=self.model.value,
                confidence=0.95,
                processing_time=time.time() - start_time,
            )
        else:
            result = VisionResult(
                description="API call failed",
                metadata={"error": "api_error"},
            )

        self._notify_callbacks(result)
        return result

    def embed(self, image: ImageInput) -> VisionResult:
        """API doesn't provide embeddings."""
        return VisionResult(
            description="API vision does not provide embeddings. Use CLIP instead.",
            metadata={"error": "not_supported"},
        )

    def classify(self, image: ImageInput, labels: List[str]) -> VisionResult:
        """Classify using API."""
        import time

        start_time = time.time()

        labels_str = ", ".join(labels)
        prompt = f"Classify this image. Choose the most appropriate label from: [{labels_str}]. Respond with only the label."

        response = self._call_openai_vision(image, prompt)

        if response:
            response_clean = response.strip()
            matched = [l for l in labels if l.lower() in response_clean.lower()]

            result = VisionResult(
                labels=matched or [response_clean],
                confidence=0.9,
                model=self.model.value,
                processing_time=time.time() - start_time,
            )
        else:
            result = VisionResult(metadata={"error": "api_error"})

        self._notify_callbacks(result)
        return result

    def detect_objects(self, image: ImageInput) -> VisionResult:
        """Detect objects using API."""
        import time

        start_time = time.time()

        prompt = "List all objects in this image with their approximate positions (e.g., 'dog in center', 'tree on left'). Format as a list."

        response = self._call_openai_vision(image, prompt)

        if response:
            result = VisionResult(
                description=response,
                model=self.model.value,
                processing_time=time.time() - start_time,
            )
        else:
            result = VisionResult(metadata={"error": "api_error"})

        self._notify_callbacks(result)
        return result

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)


# =============================================================================
# Factory Function
# =============================================================================


def create_vision_engine(
    engine_type: str = "auto",
    model: Optional[VisionModel] = None,
    **kwargs,
) -> VisionEngine:
    """
    Create a vision engine.

    Args:
        engine_type: Type of engine ("clip", "llava", "api", "mock", "auto")
        model: Vision model to use
        **kwargs: Additional engine-specific arguments

    Returns:
        VisionEngine instance
    """
    if engine_type == "mock":
        return MockVisionEngine(model or VisionModel.CLIP_VIT_B32)

    if engine_type == "clip":
        return CLIPEncoder(model or VisionModel.CLIP_VIT_B32, **kwargs)

    if engine_type == "llava":
        return LLaVAVision(model or VisionModel.LLAVA_7B, **kwargs)

    if engine_type == "api":
        return APIVisionEngine(model or VisionModel.GPT4_VISION, **kwargs)

    # Auto-detect best available engine
    if engine_type == "auto":
        # Try CLIP first (local)
        clip = CLIPEncoder()
        if clip.is_available():
            logger.info("Using CLIP for vision")
            return clip

        # Try LLaVA via Ollama
        llava = LLaVAVision()
        if llava.is_available():
            logger.info("Using LLaVA for vision")
            return llava

        # Try API
        api = APIVisionEngine()
        if api.is_available():
            logger.info("Using API for vision")
            return api

        # Fall back to mock
        logger.warning("No vision engine available, using mock")
        return MockVisionEngine()

    raise ValueError(f"Unknown vision engine type: {engine_type}")
