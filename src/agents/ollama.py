"""
Agent OS Ollama Integration Layer

Provides integration with Ollama for local LLM inference.
Supports model loading, prompt formatting, and streaming responses.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Iterator, Callable
import logging
import threading
from urllib.parse import urljoin

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


# =============================================================================
# Security: Endpoint Validation
# =============================================================================

import ipaddress
from urllib.parse import urlparse


class SSRFProtectionError(Exception):
    """Raised when a URL fails SSRF protection validation."""
    pass


def validate_ollama_endpoint(url: str) -> str:
    """
    Validate an Ollama API endpoint URL to prevent SSRF attacks.

    Args:
        url: The URL to validate

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
    if parsed.scheme not in ('http', 'https'):
        raise SSRFProtectionError(f"Invalid URL scheme: {parsed.scheme}")

    # Validate host exists
    if not parsed.hostname:
        raise SSRFProtectionError("URL has no hostname")

    hostname = parsed.hostname.lower()

    # Check for suspicious internal hostnames
    suspicious_patterns = [
        'metadata.',           # Cloud metadata services
        '169.254.',            # AWS metadata IP range
        'internal.',           # Internal services
    ]

    for pattern in suspicious_patterns:
        if pattern in hostname:
            raise SSRFProtectionError(f"Suspicious hostname pattern: {hostname}")

    # Reject .internal, .local, .corp, .lan TLDs (except localhost)
    if hostname not in ('localhost', '127.0.0.1', '::1'):
        suspicious_tlds = ['.internal', '.local', '.corp', '.lan']
        for tld in suspicious_tlds:
            if hostname.endswith(tld):
                raise SSRFProtectionError(f"Suspicious hostname pattern: {hostname}")

    # Check for IP addresses
    try:
        ip = ipaddress.ip_address(hostname)
        # Allow localhost/loopback for local Ollama
        if not ip.is_loopback:
            if ip.is_link_local or ip.is_multicast or ip.is_reserved:
                raise SSRFProtectionError(f"Reserved IP address not allowed: {hostname}")
    except ValueError:
        # Not an IP address, it's a hostname - that's fine
        pass

    return url


logger = logging.getLogger(__name__)


@dataclass
class OllamaMessage:
    """A message in an Ollama chat conversation."""
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class OllamaResponse:
    """Response from Ollama API."""
    model: str
    content: str
    done: bool = True
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0
    context: Optional[List[int]] = None

    @property
    def total_duration_ms(self) -> int:
        return self.total_duration_ns // 1_000_000

    @property
    def tokens_generated(self) -> int:
        return self.eval_count

    @property
    def tokens_per_second(self) -> float:
        if self.eval_duration_ns == 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1_000_000_000)


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""
    name: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.size / (1024 ** 3)

    @property
    def family(self) -> Optional[str]:
        return self.details.get("family")

    @property
    def parameter_size(self) -> Optional[str]:
        return self.details.get("parameter_size")

    @property
    def quantization_level(self) -> Optional[str]:
        return self.details.get("quantization_level")


class OllamaError(Exception):
    """Base exception for Ollama errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Connection to Ollama failed."""
    pass


class OllamaModelError(OllamaError):
    """Model-related error."""
    pass


class OllamaClient:
    """
    Client for interacting with Ollama API.

    Provides:
    - Model management (list, pull, info)
    - Text generation (generate, chat)
    - Embedding generation
    - Streaming responses
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        timeout: float = 120.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Ollama client.

        Args:
            endpoint: Ollama API endpoint
            timeout: Request timeout in seconds
            retry_count: Number of retries for failed requests
            retry_delay: Delay between retries in seconds

        Raises:
            SSRFProtectionError: If the endpoint URL fails SSRF validation
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx package required for Ollama integration")

        # SECURITY: Validate endpoint URL to prevent SSRF attacks
        validated_endpoint = validate_ollama_endpoint(endpoint)
        self.endpoint = validated_endpoint.rstrip("/")
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._client = httpx.Client(timeout=timeout)
        self._lock = threading.Lock()

    def close(self) -> None:
        """Close the client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==========================================================================
    # Model Management
    # ==========================================================================

    def list_models(self) -> List[OllamaModelInfo]:
        """
        List available models.

        Returns:
            List of model information
        """
        response = self._request("GET", "/api/tags")
        models = []
        for model_data in response.get("models", []):
            models.append(OllamaModelInfo(
                name=model_data["name"],
                modified_at=model_data.get("modified_at", ""),
                size=model_data.get("size", 0),
                digest=model_data.get("digest", ""),
                details=model_data.get("details", {}),
            ))
        return models

    def model_exists(self, model: str) -> bool:
        """Check if a model exists locally."""
        models = self.list_models()
        return any(m.name == model or m.name.startswith(model + ":") for m in models)

    def get_model_info(self, model: str) -> Optional[OllamaModelInfo]:
        """Get information about a specific model."""
        models = self.list_models()
        for m in models:
            if m.name == model or m.name.startswith(model + ":"):
                return m
        return None

    def pull_model(
        self,
        model: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model: Model name (e.g., "llama3:8b")
            progress_callback: Optional callback(status, completed, total)

        Returns:
            True if successful
        """
        try:
            with self._client.stream(
                "POST",
                urljoin(self.endpoint, "/api/pull"),
                json={"name": model},
                timeout=None,  # Pulling can take a long time
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)

                        if progress_callback:
                            progress_callback(status, completed, total)

                        logger.debug(f"Pull progress: {status}")

            return True

        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            raise OllamaModelError(f"Failed to pull model: {e}")

    # ==========================================================================
    # Text Generation
    # ==========================================================================

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
        raw: bool = False,
    ) -> OllamaResponse:
        """
        Generate text from a prompt.

        Args:
            model: Model name
            prompt: User prompt
            system: System prompt
            template: Custom prompt template
            context: Previous context for continuation
            options: Generation options (temperature, top_p, etc.)
            format: Response format ("json" for JSON mode)
            raw: If True, don't apply prompt template

        Returns:
            OllamaResponse with generated text
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        if system:
            payload["system"] = system
        if template:
            payload["template"] = template
        if context:
            payload["context"] = context
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format
        if raw:
            payload["raw"] = raw

        response = self._request("POST", "/api/generate", json=payload)

        return OllamaResponse(
            model=response.get("model", model),
            content=response.get("response", ""),
            done=response.get("done", True),
            total_duration_ns=response.get("total_duration", 0),
            load_duration_ns=response.get("load_duration", 0),
            prompt_eval_count=response.get("prompt_eval_count", 0),
            prompt_eval_duration_ns=response.get("prompt_eval_duration", 0),
            eval_count=response.get("eval_count", 0),
            eval_duration_ns=response.get("eval_duration", 0),
            context=response.get("context"),
        )

    def generate_stream(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Generate text with streaming response.

        Args:
            model: Model name
            prompt: User prompt
            system: System prompt
            options: Generation options

        Yields:
            Text chunks as they are generated
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }

        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        try:
            with self._client.stream(
                "POST",
                urljoin(self.endpoint, "/api/generate"),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done"):
                            break

        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            raise OllamaError(f"Stream generation failed: {e}")

    def chat(
        self,
        model: str,
        messages: List[OllamaMessage],
        options: Optional[Dict[str, Any]] = None,
        format: Optional[str] = None,
    ) -> OllamaResponse:
        """
        Chat completion with message history.

        Args:
            model: Model name
            messages: List of messages (conversation history)
            options: Generation options
            format: Response format

        Returns:
            OllamaResponse with assistant's reply
        """
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
        }

        if options:
            payload["options"] = options
        if format:
            payload["format"] = format

        response = self._request("POST", "/api/chat", json=payload)

        message = response.get("message", {})

        return OllamaResponse(
            model=response.get("model", model),
            content=message.get("content", ""),
            done=response.get("done", True),
            total_duration_ns=response.get("total_duration", 0),
            load_duration_ns=response.get("load_duration", 0),
            prompt_eval_count=response.get("prompt_eval_count", 0),
            prompt_eval_duration_ns=response.get("prompt_eval_duration", 0),
            eval_count=response.get("eval_count", 0),
            eval_duration_ns=response.get("eval_duration", 0),
        )

    def chat_stream(
        self,
        model: str,
        messages: List[OllamaMessage],
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Chat completion with streaming response.

        Args:
            model: Model name
            messages: List of messages
            options: Generation options

        Yields:
            Text chunks as they are generated
        """
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }

        if options:
            payload["options"] = options

        try:
            with self._client.stream(
                "POST",
                urljoin(self.endpoint, "/api/chat"),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        message = data.get("message", {})
                        if "content" in message:
                            yield message["content"]
                        if data.get("done"):
                            break

        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            raise OllamaError(f"Stream chat failed: {e}")

    # ==========================================================================
    # Embeddings
    # ==========================================================================

    def embed(
        self,
        model: str,
        prompt: str,
    ) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            model: Embedding model name
            prompt: Text to embed

        Returns:
            Embedding vector
        """
        response = self._request(
            "POST",
            "/api/embeddings",
            json={"model": model, "prompt": prompt},
        )
        return response.get("embedding", [])

    # ==========================================================================
    # Health Check
    # ==========================================================================

    def is_healthy(self) -> bool:
        """Check if Ollama is running and healthy."""
        try:
            self._request("GET", "/")
            return True
        except Exception:
            return False

    def wait_for_ready(
        self,
        timeout: float = 30.0,
        interval: float = 1.0,
    ) -> bool:
        """
        Wait for Ollama to become ready.

        Args:
            timeout: Maximum wait time
            interval: Check interval

        Returns:
            True if ready, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.is_healthy():
                return True
            time.sleep(interval)
        return False

    # ==========================================================================
    # Internal
    # ==========================================================================

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        url = urljoin(self.endpoint, path)

        last_error = None
        for attempt in range(self.retry_count):
            try:
                response = self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json() if response.content else {}

            except httpx.ConnectError as e:
                last_error = OllamaConnectionError(f"Cannot connect to Ollama: {e}")
                logger.warning(f"Connection failed (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay)

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = OllamaError(f"Server error: {e}")
                    time.sleep(self.retry_delay)
                else:
                    raise OllamaError(f"Request failed: {e}")

            except Exception as e:
                raise OllamaError(f"Request error: {e}")

        raise last_error or OllamaError("Request failed after retries")


class OllamaModelManager:
    """
    Manages Ollama models for Agent OS.

    Provides:
    - Model caching and preloading
    - Model switching
    - Resource management
    """

    def __init__(self, client: OllamaClient):
        """
        Initialize model manager.

        Args:
            client: Ollama client
        """
        self.client = client
        self._loaded_models: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def ensure_model(
        self,
        model: str,
        auto_pull: bool = True,
    ) -> bool:
        """
        Ensure a model is available.

        Args:
            model: Model name
            auto_pull: Pull if not available

        Returns:
            True if model is available
        """
        with self._lock:
            if self.client.model_exists(model):
                return True

            if auto_pull:
                logger.info(f"Pulling model: {model}")
                try:
                    self.client.pull_model(model)
                    return True
                except OllamaModelError as e:
                    logger.error(f"Failed to pull model: {e}")
                    return False

            return False

    def preload_model(self, model: str) -> bool:
        """
        Preload a model into memory.

        Args:
            model: Model name

        Returns:
            True if loaded successfully
        """
        if not self.ensure_model(model):
            return False

        # Generate empty prompt to load model
        try:
            self.client.generate(model, "", options={"num_predict": 1})
            self._loaded_models[model] = datetime.now()
            logger.info(f"Preloaded model: {model}")
            return True
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
            return False

    def get_loaded_models(self) -> List[str]:
        """Get list of models that have been loaded."""
        return list(self._loaded_models.keys())


def create_ollama_client(
    endpoint: Optional[str] = None,
    timeout: float = 120.0,
) -> OllamaClient:
    """
    Create an Ollama client.

    Args:
        endpoint: Ollama endpoint (default: localhost:11434)
        timeout: Request timeout

    Returns:
        OllamaClient instance
    """
    import os
    endpoint = endpoint or os.environ.get(
        "OLLAMA_ENDPOINT",
        "http://localhost:11434"
    )
    return OllamaClient(endpoint=endpoint, timeout=timeout)
