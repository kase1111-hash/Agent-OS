"""
Image Generation Routes

API endpoints for AI image generation using local LLMs.
Supports Stable Diffusion, SDXL, and other image generation models.
"""

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Security Utilities
# =============================================================================

import ipaddress
import re
from urllib.parse import urlparse

from src.core.exceptions import PathValidationError, SSRFProtectionError

# Backward compatibility alias
PathTraversalError = PathValidationError


def validate_filename_in_directory(filename: str, base_directory: Path) -> Path:
    """
    Validate that a filename resolves to a path within the base directory.

    Prevents path traversal attacks by ensuring the resolved path
    doesn't escape the intended directory.

    Args:
        filename: The filename to validate
        base_directory: The directory the file must remain within

    Returns:
        The validated absolute path

    Raises:
        PathTraversalError: If the path escapes the base directory
    """
    # Reject obviously malicious filenames
    if not filename:
        raise PathTraversalError("Empty filename")

    # Check for null bytes
    if "\x00" in filename:
        raise PathTraversalError("Filename contains null byte")

    # Check for path traversal patterns
    if ".." in filename:
        raise PathTraversalError("Filename contains path traversal sequence")

    # Check for absolute paths
    if filename.startswith("/") or filename.startswith("\\"):
        raise PathTraversalError("Filename is an absolute path")

    # On Windows, also check for drive letters
    if len(filename) >= 2 and filename[1] == ":":
        raise PathTraversalError("Filename contains drive letter")

    # Construct the full path
    filepath = base_directory / filename

    # Resolve to absolute path and ensure it stays within base directory
    try:
        resolved_base = base_directory.resolve()
        resolved_path = filepath.resolve()
    except (OSError, RuntimeError) as e:
        raise PathTraversalError(f"Cannot resolve path: {e}")

    # Verify the resolved path is within the base directory
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError:
        raise PathTraversalError(
            f"Path escapes base directory: {resolved_path} is not within {resolved_base}"
        )

    return resolved_path


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private, loopback, or otherwise internal."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False


def validate_api_endpoint(url: str, allow_localhost: bool = True) -> str:
    """
    Validate an API endpoint URL to prevent SSRF attacks.

    Args:
        url: The URL to validate
        allow_localhost: Whether to allow localhost URLs (for local services)

    Returns:
        The validated URL

    Raises:
        SSRFProtectionError: If the URL fails validation
    """
    if not url:
        raise SSRFProtectionError("Empty URL")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise SSRFProtectionError(f"Invalid URL format: {e}")

    # Validate scheme
    if parsed.scheme not in ("http", "https"):
        raise SSRFProtectionError(f"Invalid URL scheme: {parsed.scheme}")

    # Validate host exists
    if not parsed.hostname:
        raise SSRFProtectionError("URL has no hostname")

    hostname = parsed.hostname.lower()

    # Check for suspicious internal hostnames
    suspicious_patterns = [
        r"^metadata\.",  # Cloud metadata services
        r"^169\.254\.",  # AWS metadata IP range
        r"^internal\.",  # Internal services
        r"\.internal$",
        r"\.local$",
        r"\.corp$",
        r"\.lan$",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, hostname):
            raise SSRFProtectionError(f"Suspicious hostname pattern: {hostname}")

    # Check for IP addresses
    try:
        ip = ipaddress.ip_address(hostname)
        if not allow_localhost and ip.is_loopback:
            raise SSRFProtectionError("Localhost not allowed")
        if ip.is_private and not ip.is_loopback:
            raise SSRFProtectionError(f"Private IP address not allowed: {hostname}")
        if ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise SSRFProtectionError(f"Reserved IP address not allowed: {hostname}")
    except ValueError:
        # Not an IP address, it's a hostname - check for localhost
        if not allow_localhost and hostname in ("localhost", "127.0.0.1", "::1"):
            raise SSRFProtectionError("Localhost not allowed")

    return url


# =============================================================================
# Models
# =============================================================================


class ImageGenerationRequest(BaseModel):
    """Request for image generation."""

    prompt: str = Field(
        ..., min_length=1, max_length=2000, description="Text prompt for image generation"
    )
    negative_prompt: Optional[str] = Field(
        None, max_length=1000, description="Negative prompt to avoid certain features"
    )
    model: Optional[str] = Field(None, description="Model to use for generation")
    width: int = Field(512, ge=64, le=2048, description="Image width in pixels")
    height: int = Field(512, ge=64, le=2048, description="Image height in pixels")
    steps: int = Field(20, ge=1, le=150, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0, description="Guidance scale (CFG)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    scheduler: Optional[str] = Field(None, description="Scheduler/sampler to use")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")


class ImageGenerationResponse(BaseModel):
    """Response from image generation."""

    id: str
    status: str  # "pending", "processing", "completed", "failed"
    prompt: str
    model: str
    created_at: str
    completed_at: Optional[str] = None
    images: List[Dict[str, Any]] = []
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ImageModelInfo(BaseModel):
    """Information about an image generation model."""

    id: str
    name: str
    type: str  # "stable-diffusion", "sdxl", "flux", etc.
    description: str
    capabilities: List[str]
    default_resolution: Dict[str, int]
    max_resolution: Dict[str, int]
    available: bool = True


class GalleryImage(BaseModel):
    """Image in the gallery."""

    id: str
    prompt: str
    negative_prompt: Optional[str]
    model: str
    width: int
    height: int
    seed: int
    steps: int
    guidance_scale: float
    created_at: str
    thumbnail_url: str
    full_url: str
    metadata: Dict[str, Any] = {}


# =============================================================================
# In-Memory Storage (for demo - replace with persistent storage in production)
# =============================================================================


@dataclass
class ImageGenerationJob:
    """Represents an image generation job."""

    id: str
    prompt: str
    negative_prompt: Optional[str]
    model: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    num_images: int
    scheduler: Optional[str]
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    images: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class ImageStore:
    """In-memory store for image generation jobs and gallery."""

    def __init__(self):
        self.jobs: Dict[str, ImageGenerationJob] = {}
        self.gallery: List[GalleryImage] = []
        self.output_dir = Path(os.environ.get("IMAGE_OUTPUT_DIR", "./generated_images"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, request: ImageGenerationRequest) -> ImageGenerationJob:
        """Create a new image generation job."""
        import random

        job = ImageGenerationJob(
            id=str(uuid.uuid4()),
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model=request.model or "stable-diffusion-v1-5",
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed if request.seed is not None else random.randint(0, 2**32 - 1),
            num_images=request.num_images,
            scheduler=request.scheduler,
        )
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Optional[ImageGenerationJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> Optional[ImageGenerationJob]:
        """Update a job."""
        job = self.jobs.get(job_id)
        if job:
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
        return job

    def list_jobs(self, limit: int = 50) -> List[ImageGenerationJob]:
        """List recent jobs."""
        jobs = list(self.jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def add_to_gallery(self, image: GalleryImage) -> None:
        """Add an image to the gallery."""
        self.gallery.insert(0, image)
        # Keep gallery limited to last 100 images
        if len(self.gallery) > 100:
            self.gallery = self.gallery[:100]

    def get_gallery(self, limit: int = 50, offset: int = 0) -> List[GalleryImage]:
        """Get gallery images."""
        return self.gallery[offset : offset + limit]

    def delete_from_gallery(self, image_id: str) -> bool:
        """Delete an image from gallery."""
        for i, img in enumerate(self.gallery):
            if img.id == image_id:
                del self.gallery[i]
                return True
        return False


# Global store instance
_image_store = ImageStore()


def get_image_store() -> ImageStore:
    """Get the image store instance."""
    return _image_store


# =============================================================================
# Model Registry
# =============================================================================

# Mapping from short model IDs to full Hugging Face model paths
MODEL_HF_PATHS: Dict[str, str] = {
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "stable-diffusion-xl-turbo": "stabilityai/sdxl-turbo",
    "flux-1-schnell": "black-forest-labs/FLUX.1-schnell",
    "playground-v2.5": "playgroundai/playground-v2.5-1024px-aesthetic",
}

# Available image generation models (can be expanded)
AVAILABLE_MODELS: List[ImageModelInfo] = [
    ImageModelInfo(
        id="stable-diffusion-v1-5",
        name="Stable Diffusion v1.5",
        type="stable-diffusion",
        description="Classic Stable Diffusion model, good balance of speed and quality",
        capabilities=["txt2img", "img2img", "inpainting"],
        default_resolution={"width": 512, "height": 512},
        max_resolution={"width": 1024, "height": 1024},
        available=True,
    ),
    ImageModelInfo(
        id="stable-diffusion-xl-base-1.0",
        name="Stable Diffusion XL Base 1.0",
        type="sdxl",
        description="High-quality SDXL base model with improved aesthetics",
        capabilities=["txt2img", "img2img", "inpainting"],
        default_resolution={"width": 1024, "height": 1024},
        max_resolution={"width": 2048, "height": 2048},
        available=True,
    ),
    ImageModelInfo(
        id="stable-diffusion-xl-turbo",
        name="SDXL Turbo",
        type="sdxl-turbo",
        description="Fast SDXL variant, generates in 1-4 steps",
        capabilities=["txt2img"],
        default_resolution={"width": 512, "height": 512},
        max_resolution={"width": 1024, "height": 1024},
        available=True,
    ),
    ImageModelInfo(
        id="flux-1-schnell",
        name="FLUX.1 [schnell]",
        type="flux",
        description="Fast, high-quality image generation from Black Forest Labs",
        capabilities=["txt2img"],
        default_resolution={"width": 1024, "height": 1024},
        max_resolution={"width": 2048, "height": 2048},
        available=True,
    ),
    ImageModelInfo(
        id="playground-v2.5",
        name="Playground v2.5",
        type="playground",
        description="Aesthetic-focused model with excellent prompt following",
        capabilities=["txt2img"],
        default_resolution={"width": 1024, "height": 1024},
        max_resolution={"width": 2048, "height": 2048},
        available=True,
    ),
]


# =============================================================================
# Image Generation Backend
# =============================================================================


class ImageGenerator:
    """
    Image generation backend.

    Supports multiple backends:
    - diffusers (Hugging Face)
    - ComfyUI API
    - Automatic1111 API
    - Custom llama.cpp multimodal
    """

    def __init__(self):
        self._diffusers_available = False
        self._comfyui_endpoint = None
        self._a1111_endpoint = None

        # Validate and set ComfyUI endpoint if configured
        comfyui_env = os.environ.get("COMFYUI_ENDPOINT")
        if comfyui_env:
            try:
                # Allow localhost since these are typically local services
                self._comfyui_endpoint = validate_api_endpoint(comfyui_env, allow_localhost=True)
            except SSRFProtectionError as e:
                logger.error(f"Invalid COMFYUI_ENDPOINT configuration: {e}")
                # Don't set the endpoint - it fails validation

        # Validate and set A1111 endpoint if configured
        a1111_env = os.environ.get("A1111_ENDPOINT")
        if a1111_env:
            try:
                # Allow localhost since these are typically local services
                self._a1111_endpoint = validate_api_endpoint(a1111_env, allow_localhost=True)
            except SSRFProtectionError as e:
                logger.error(f"Invalid A1111_ENDPOINT configuration: {e}")
                # Don't set the endpoint - it fails validation

        # Check for diffusers
        try:
            import diffusers

            self._diffusers_available = True
        except ImportError:
            pass

    async def generate(
        self,
        job: ImageGenerationJob,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate images for a job.

        Returns list of generated image data.
        """
        # Try different backends
        if self._comfyui_endpoint:
            return await self._generate_comfyui(job, progress_callback)
        elif self._a1111_endpoint:
            return await self._generate_a1111(job, progress_callback)
        elif self._diffusers_available:
            return await self._generate_diffusers(job, progress_callback)
        else:
            # Placeholder - generate a simple gradient image
            return await self._generate_placeholder(job, progress_callback)

    async def _generate_placeholder(
        self,
        job: ImageGenerationJob,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate placeholder images (for demo without actual model)."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            # Return a simple base64 encoded placeholder
            return self._generate_simple_placeholder(job)

        images = []
        store = get_image_store()

        for i in range(job.num_images):
            if progress_callback:
                await progress_callback(i + 1, job.num_images)

            # Create a gradient image with prompt text
            img = Image.new("RGB", (job.width, job.height))
            draw = ImageDraw.Draw(img)

            # Create gradient
            for y in range(job.height):
                # Color based on seed and position
                r = int((job.seed + y) % 256)
                g = int((job.seed * 2 + y) % 256)
                b = int((job.seed * 3 + job.height - y) % 256)
                for x in range(job.width):
                    # Add some variation
                    rr = (r + x // 4) % 256
                    gg = (g + x // 8) % 256
                    bb = (b + x // 6) % 256
                    draw.point((x, y), fill=(rr, gg, bb))

            # Add prompt text overlay
            text = job.prompt[:50] + "..." if len(job.prompt) > 50 else job.prompt
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except Exception:
                font = ImageFont.load_default()

            # Draw text with background
            text_bbox = draw.textbbox((10, 10), text, font=font)
            draw.rectangle(
                [text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5],
                fill=(0, 0, 0, 180),
            )
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)

            # Add model info
            info_text = f"Model: {job.model} | Seed: {job.seed + i}"
            draw.text((10, job.height - 30), info_text, fill=(200, 200, 200), font=font)

            # Save image
            image_id = str(uuid.uuid4())
            filename = f"{image_id}.png"
            filepath = store.output_dir / filename

            img.save(filepath, "PNG")

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode()

            images.append(
                {
                    "id": image_id,
                    "filename": filename,
                    "base64": base64_image,
                    "width": job.width,
                    "height": job.height,
                    "seed": job.seed + i,
                }
            )

            # Small delay to simulate processing
            await asyncio.sleep(0.5)

        return images

    def _generate_simple_placeholder(self, job: ImageGenerationJob) -> List[Dict[str, Any]]:
        """Generate very simple placeholder without PIL."""
        images = []
        for i in range(job.num_images):
            # Create a simple 1x1 pixel PNG (placeholder)
            images.append(
                {
                    "id": str(uuid.uuid4()),
                    "filename": "placeholder.png",
                    "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "width": job.width,
                    "height": job.height,
                    "seed": job.seed + i,
                    "placeholder": True,
                }
            )
        return images

    async def _generate_diffusers(
        self,
        job: ImageGenerationJob,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate using Hugging Face diffusers."""
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        # Load pipeline - resolve short model ID to full HuggingFace path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = MODEL_HF_PATHS.get(job.model, job.model)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        pipe = pipe.to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Generate
        generator = torch.Generator(device).manual_seed(job.seed)
        images_pil = pipe(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            width=job.width,
            height=job.height,
            num_inference_steps=job.steps,
            guidance_scale=job.guidance_scale,
            num_images_per_prompt=job.num_images,
            generator=generator,
        ).images

        # Convert to output format
        images = []
        store = get_image_store()

        for i, img in enumerate(images_pil):
            image_id = str(uuid.uuid4())
            filename = f"{image_id}.png"
            filepath = store.output_dir / filename

            img.save(filepath, "PNG")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode()

            images.append(
                {
                    "id": image_id,
                    "filename": filename,
                    "base64": base64_image,
                    "width": job.width,
                    "height": job.height,
                    "seed": job.seed + i,
                }
            )

        return images

    async def _generate_comfyui(
        self,
        job: ImageGenerationJob,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate using ComfyUI API."""
        import httpx

        # ComfyUI workflow for txt2img
        workflow = {
            "prompt": {
                "3": {
                    "inputs": {
                        "seed": job.seed,
                        "steps": job.steps,
                        "cfg": job.guidance_scale,
                        "sampler_name": job.scheduler or "euler",
                        "scheduler": "normal",
                        "denoise": 1,
                        "model": ["4", 0],
                        "positive": ["6", 0],
                        "negative": ["7", 0],
                        "latent_image": ["5", 0],
                    },
                    "class_type": "KSampler",
                },
                # Add more workflow nodes...
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._comfyui_endpoint}/prompt",
                json=workflow,
                timeout=300,
            )
            result = response.json()

        # Process ComfyUI response
        images = []
        # ... parse ComfyUI output
        return images

    async def _generate_a1111(
        self,
        job: ImageGenerationJob,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate using Automatic1111 API."""
        import httpx

        payload = {
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt or "",
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "cfg_scale": job.guidance_scale,
            "seed": job.seed,
            "batch_size": job.num_images,
            "sampler_index": job.scheduler or "Euler a",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._a1111_endpoint}/sdapi/v1/txt2img",
                json=payload,
                timeout=300,
            )
            result = response.json()

        images = []
        store = get_image_store()

        for i, img_base64 in enumerate(result.get("images", [])):
            image_id = str(uuid.uuid4())
            filename = f"{image_id}.png"

            # Decode and save
            img_data = base64.b64decode(img_base64)
            filepath = store.output_dir / filename
            with open(filepath, "wb") as f:
                f.write(img_data)

            images.append(
                {
                    "id": image_id,
                    "filename": filename,
                    "base64": img_base64,
                    "width": job.width,
                    "height": job.height,
                    "seed": job.seed + i,
                }
            )

        return images


# Global generator instance
_generator = ImageGenerator()


def get_generator() -> ImageGenerator:
    """Get the image generator instance."""
    return _generator


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/models", response_model=List[ImageModelInfo])
async def list_models():
    """List available image generation models."""
    return AVAILABLE_MODELS


@router.get("/models/{model_id}", response_model=ImageModelInfo)
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=404, detail="Model not found")


@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate images from a text prompt.

    Starts an async generation job and returns immediately.
    Poll the status endpoint to check completion.
    """
    store = get_image_store()
    job = store.create_job(request)

    # Start generation in background
    background_tasks.add_task(run_generation_job, job.id)

    return ImageGenerationResponse(
        id=job.id,
        status=job.status,
        prompt=job.prompt,
        model=job.model,
        created_at=job.created_at.isoformat(),
        metadata={
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "guidance_scale": job.guidance_scale,
            "seed": job.seed,
        },
    )


@router.get("/generate/{job_id}", response_model=ImageGenerationResponse)
async def get_generation_status(job_id: str):
    """Get the status of a generation job."""
    store = get_image_store()
    job = store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return ImageGenerationResponse(
        id=job.id,
        status=job.status,
        prompt=job.prompt,
        model=job.model,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        images=job.images,
        error=job.error,
        metadata={
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "guidance_scale": job.guidance_scale,
            "seed": job.seed,
        },
    )


@router.get("/jobs", response_model=List[ImageGenerationResponse])
async def list_jobs(limit: int = 50):
    """List recent generation jobs."""
    store = get_image_store()
    jobs = store.list_jobs(limit)

    return [
        ImageGenerationResponse(
            id=job.id,
            status=job.status,
            prompt=job.prompt,
            model=job.model,
            created_at=job.created_at.isoformat(),
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            images=job.images,
            error=job.error,
        )
        for job in jobs
    ]


@router.get("/gallery", response_model=List[GalleryImage])
async def get_gallery(limit: int = 50, offset: int = 0):
    """Get gallery of generated images."""
    store = get_image_store()
    return store.get_gallery(limit, offset)


@router.delete("/gallery/{image_id}")
async def delete_gallery_image(image_id: str):
    """Delete an image from the gallery."""
    store = get_image_store()
    if store.delete_from_gallery(image_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """Get a generated image by ID."""
    store = get_image_store()

    # Check in jobs
    for job in store.jobs.values():
        for img in job.images:
            if img.get("id") == image_id:
                # Return the image file - validate filename to prevent path traversal
                try:
                    filepath = validate_filename_in_directory(img["filename"], store.output_dir)
                except PathTraversalError as e:
                    logger.warning(f"Path traversal attempt blocked: {e}")
                    raise HTTPException(status_code=400, detail="Invalid filename")

                if filepath.exists():
                    return StreamingResponse(
                        open(filepath, "rb"),
                        media_type="image/png",
                    )
                # Return base64 as fallback
                if "base64" in img:
                    return StreamingResponse(
                        io.BytesIO(base64.b64decode(img["base64"])),
                        media_type="image/png",
                    )

    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/stats")
async def get_stats():
    """Get image generation statistics."""
    store = get_image_store()
    jobs = list(store.jobs.values())

    completed = [j for j in jobs if j.status == "completed"]
    failed = [j for j in jobs if j.status == "failed"]
    pending = [j for j in jobs if j.status in ("pending", "processing")]

    total_images = sum(len(j.images) for j in completed)

    return {
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
        "pending_jobs": len(pending),
        "total_images": total_images,
        "gallery_size": len(store.gallery),
    }


# =============================================================================
# WebSocket for Real-time Updates
# =============================================================================


@router.websocket("/ws")
async def image_websocket(websocket: WebSocket):
    """WebSocket for real-time generation updates."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "generate":
                # Start generation
                request = ImageGenerationRequest(**data.get("params", {}))
                store = get_image_store()
                job = store.create_job(request)

                await websocket.send_json(
                    {
                        "type": "job_created",
                        "job_id": job.id,
                    }
                )

                # Run generation with progress updates
                await run_generation_with_websocket(job.id, websocket)

            elif data.get("type") == "status":
                # Get job status
                job_id = data.get("job_id")
                store = get_image_store()
                job = store.get_job(job_id)

                if job:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "job_id": job.id,
                            "status": job.status,
                            "images": job.images,
                        }
                    )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


# =============================================================================
# Background Tasks
# =============================================================================


async def run_generation_job(job_id: str):
    """Run image generation job in background."""
    store = get_image_store()
    generator = get_generator()

    job = store.get_job(job_id)
    if not job:
        return

    try:
        store.update_job(job_id, status="processing")

        images = await generator.generate(job)

        store.update_job(
            job_id,
            status="completed",
            completed_at=datetime.now(),
            images=images,
        )

        # Add to gallery
        for img in images:
            gallery_image = GalleryImage(
                id=img["id"],
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                model=job.model,
                width=img["width"],
                height=img["height"],
                seed=img["seed"],
                steps=job.steps,
                guidance_scale=job.guidance_scale,
                created_at=datetime.now().isoformat(),
                thumbnail_url=f"/api/images/image/{img['id']}",
                full_url=f"/api/images/image/{img['id']}",
            )
            store.add_to_gallery(gallery_image)

    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}")
        store.update_job(
            job_id,
            status="failed",
            completed_at=datetime.now(),
            error=str(e),
        )


async def run_generation_with_websocket(job_id: str, websocket: WebSocket):
    """Run generation with WebSocket progress updates."""
    store = get_image_store()
    generator = get_generator()

    job = store.get_job(job_id)
    if not job:
        return

    try:
        store.update_job(job_id, status="processing")
        await websocket.send_json(
            {
                "type": "progress",
                "job_id": job_id,
                "status": "processing",
                "progress": 0,
            }
        )

        async def progress_callback(current: int, total: int):
            progress = int((current / total) * 100)
            await websocket.send_json(
                {
                    "type": "progress",
                    "job_id": job_id,
                    "status": "processing",
                    "progress": progress,
                }
            )

        images = await generator.generate(job, progress_callback)

        store.update_job(
            job_id,
            status="completed",
            completed_at=datetime.now(),
            images=images,
        )

        await websocket.send_json(
            {
                "type": "completed",
                "job_id": job_id,
                "images": images,
            }
        )

        # Add to gallery
        for img in images:
            gallery_image = GalleryImage(
                id=img["id"],
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                model=job.model,
                width=img["width"],
                height=img["height"],
                seed=img["seed"],
                steps=job.steps,
                guidance_scale=job.guidance_scale,
                created_at=datetime.now().isoformat(),
                thumbnail_url=f"/api/images/image/{img['id']}",
                full_url=f"/api/images/image/{img['id']}",
            )
            store.add_to_gallery(gallery_image)

    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}")
        store.update_job(
            job_id,
            status="failed",
            completed_at=datetime.now(),
            error=str(e),
        )
        await websocket.send_json(
            {
                "type": "error",
                "job_id": job_id,
                "error": str(e),
            }
        )
