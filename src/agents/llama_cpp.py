"""
Agent OS Llama.cpp Integration Layer

Provides integration with llama.cpp for local LLM inference.
Supports both the llama-cpp-python library and llama.cpp server API.

llama.cpp GitHub: https://github.com/ggerganov/llama.cpp
llama-cpp-python: https://github.com/abetlen/llama-cpp-python
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


# Check for llama-cpp-python availability
try:
    from llama_cpp import Llama

    LLAMA_CPP_PYTHON_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_PYTHON_AVAILABLE = False

# Check for httpx (for server mode)
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False


# =============================================================================
# Security: Endpoint Validation
# =============================================================================

import ipaddress
from urllib.parse import urlparse


class SSRFProtectionError(Exception):
    """Raised when a URL fails SSRF protection validation."""

    pass


def validate_llama_endpoint(url: str) -> str:
    """
    Validate a Llama.cpp server endpoint URL to prevent SSRF attacks.

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
    if parsed.scheme not in ("http", "https"):
        raise SSRFProtectionError(f"Invalid URL scheme: {parsed.scheme}")

    # Validate host exists
    if not parsed.hostname:
        raise SSRFProtectionError("URL has no hostname")

    hostname = parsed.hostname.lower()

    # Check for suspicious internal hostnames
    suspicious_patterns = [
        "metadata.",  # Cloud metadata services
        "169.254.",  # AWS metadata IP range
        "internal.",  # Internal services
    ]

    for pattern in suspicious_patterns:
        if pattern in hostname:
            raise SSRFProtectionError(f"Suspicious hostname pattern: {hostname}")

    # Reject .internal, .local, .corp, .lan TLDs (except localhost)
    if hostname not in ("localhost", "127.0.0.1", "::1"):
        suspicious_tlds = [".internal", ".local", ".corp", ".lan"]
        for tld in suspicious_tlds:
            if hostname.endswith(tld):
                raise SSRFProtectionError(f"Suspicious hostname pattern: {hostname}")

    # Check for IP addresses
    try:
        ip = ipaddress.ip_address(hostname)
        # Allow localhost/loopback for local llama.cpp server
        if not ip.is_loopback:
            if ip.is_link_local or ip.is_multicast or ip.is_reserved:
                raise SSRFProtectionError(f"Reserved IP address not allowed: {hostname}")
    except ValueError:
        # Not an IP address, it's a hostname - that's fine
        pass

    return url


@dataclass
class LlamaCppMessage:
    """A message in a Llama.cpp chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LlamaCppResponse:
    """Response from Llama.cpp."""

    model: str
    content: str
    done: bool = True
    total_duration_ns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    finish_reason: str = "stop"

    @property
    def total_duration_ms(self) -> int:
        return self.total_duration_ns // 1_000_000


@dataclass
class LlamaCppModelInfo:
    """Information about a Llama.cpp model."""

    name: str
    path: str
    size: int
    context_length: int = 4096
    n_params: Optional[int] = None
    quantization: Optional[str] = None
    loaded: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.size / (1024**3)


class LlamaCppError(Exception):
    """Base exception for Llama.cpp errors."""

    pass


class LlamaCppConnectionError(LlamaCppError):
    """Connection to Llama.cpp server failed."""

    pass


class LlamaCppModelError(LlamaCppError):
    """Model-related error."""

    pass


class LlamaCppClient:
    """
    Client for interacting with Llama.cpp.

    Supports two modes:
    1. Direct mode: Uses llama-cpp-python library for in-process inference
    2. Server mode: Uses llama.cpp server API (compatible with OpenAI API)

    Provides:
    - Model loading and management
    - Text generation (generate, chat)
    - Embedding generation
    - Streaming responses
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        endpoint: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        use_mmap: bool = True,
        use_mlock: bool = False,
        timeout: float = 120.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        verbose: bool = False,
    ):
        """
        Initialize Llama.cpp client.

        For direct mode, provide model_path.
        For server mode, provide endpoint.

        Args:
            model_path: Path to GGUF model file (for direct mode)
            endpoint: Llama.cpp server endpoint (for server mode, e.g., http://localhost:8080)
            n_ctx: Context window size
            n_threads: Number of threads (None = auto)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            use_mmap: Use memory mapping
            use_mlock: Lock model in memory
            timeout: Request timeout in seconds
            retry_count: Number of retries for failed requests
            retry_delay: Delay between retries in seconds
            verbose: Enable verbose logging

        Raises:
            SSRFProtectionError: If the endpoint URL fails SSRF validation
        """
        self.model_path = model_path

        # SECURITY: Validate endpoint URL to prevent SSRF attacks
        if endpoint:
            validated_endpoint = validate_llama_endpoint(endpoint)
            self.endpoint = validated_endpoint.rstrip("/")
        else:
            self.endpoint = None

        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.verbose = verbose

        self._model: Optional[Any] = None  # Llama instance
        self._http_client: Optional[Any] = None  # httpx client
        self._lock = threading.Lock()
        self._model_info: Optional[LlamaCppModelInfo] = None

        # Determine mode
        if self.endpoint:
            self._mode = "server"
            if not HTTPX_AVAILABLE:
                raise ImportError("httpx package required for server mode")
            self._http_client = httpx.Client(timeout=timeout)
        elif model_path:
            self._mode = "direct"
            if not LLAMA_CPP_PYTHON_AVAILABLE:
                raise ImportError(
                    "llama-cpp-python package required for direct mode. "
                    "Install with: pip install llama-cpp-python"
                )
        else:
            raise ValueError("Either model_path or endpoint must be provided")

    def close(self) -> None:
        """Close the client and release resources."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        if self._model:
            del self._model
            self._model = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    @property
    def is_server_mode(self) -> bool:
        """Check if client is in server mode."""
        return self._mode == "server"

    @property
    def is_direct_mode(self) -> bool:
        """Check if client is in direct mode."""
        return self._mode == "direct"

    # ==========================================================================
    # Model Management
    # ==========================================================================

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a model (direct mode only).

        Args:
            model_path: Path to GGUF model file (uses initial path if not provided)

        Returns:
            True if model loaded successfully
        """
        if self._mode == "server":
            logger.warning("load_model() is not needed in server mode")
            return True

        path = model_path or self.model_path
        if not path:
            raise LlamaCppModelError("No model path provided")

        if not Path(path).exists():
            raise LlamaCppModelError(f"Model file not found: {path}")

        try:
            with self._lock:
                # Unload existing model
                if self._model:
                    del self._model
                    self._model = None

                logger.info(f"Loading model: {path}")
                self._model = Llama(
                    model_path=path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    use_mmap=self.use_mmap,
                    use_mlock=self.use_mlock,
                    verbose=self.verbose,
                )

                # Store model info
                file_size = Path(path).stat().st_size
                self._model_info = LlamaCppModelInfo(
                    name=Path(path).stem,
                    path=path,
                    size=file_size,
                    context_length=self.n_ctx,
                    loaded=True,
                )

                logger.info(f"Model loaded: {Path(path).stem}")
                return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise LlamaCppModelError(f"Failed to load model: {e}")

    def unload_model(self) -> bool:
        """Unload the current model (direct mode only)."""
        if self._mode == "server":
            logger.warning("unload_model() is not available in server mode")
            return False

        with self._lock:
            if self._model:
                del self._model
                self._model = None
                if self._model_info:
                    self._model_info.loaded = False
                logger.info("Model unloaded")
                return True
            return False

    def get_model_info(self) -> Optional[LlamaCppModelInfo]:
        """Get information about the loaded/connected model."""
        if self._mode == "server":
            try:
                response = self._request("GET", "/v1/models")
                models = response.get("data", [])
                if models:
                    model = models[0]
                    return LlamaCppModelInfo(
                        name=model.get("id", "unknown"),
                        path="",
                        size=0,
                        context_length=self.n_ctx,
                        loaded=True,
                    )
            except Exception:
                pass
            return None
        else:
            return self._model_info

    def list_models(self, model_dir: Optional[str] = None) -> List[LlamaCppModelInfo]:
        """
        List available GGUF models in a directory.

        Args:
            model_dir: Directory to scan for models

        Returns:
            List of model information
        """
        models = []

        if self._mode == "server":
            try:
                response = self._request("GET", "/v1/models")
                for model in response.get("data", []):
                    models.append(
                        LlamaCppModelInfo(
                            name=model.get("id", "unknown"),
                            path="",
                            size=0,
                            loaded=True,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to list models from server: {e}")
        else:
            # Scan directory for GGUF files
            search_dir = model_dir or os.environ.get("LLAMA_MODEL_DIR", ".")
            search_path = Path(search_dir)

            if search_path.exists():
                for gguf_file in search_path.glob("**/*.gguf"):
                    try:
                        file_size = gguf_file.stat().st_size
                        models.append(
                            LlamaCppModelInfo(
                                name=gguf_file.stem,
                                path=str(gguf_file),
                                size=file_size,
                                loaded=(
                                    self._model_info and self._model_info.path == str(gguf_file)
                                ),
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get info for {gguf_file}: {e}")

        return models

    # ==========================================================================
    # Text Generation
    # ==========================================================================

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        echo: bool = False,
    ) -> LlamaCppResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Repetition penalty
            stop: Stop sequences
            echo: Echo prompt in output

        Returns:
            LlamaCppResponse with generated text
        """
        start_time = time.time_ns()

        if self._mode == "server":
            return self._generate_server(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                echo=echo,
            )
        else:
            return self._generate_direct(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                echo=echo,
                start_time=start_time,
            )

    def _generate_direct(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: Optional[List[str]],
        echo: bool,
        start_time: int,
    ) -> LlamaCppResponse:
        """Generate using direct mode (llama-cpp-python)."""
        if not self._model:
            self.load_model()

        if not self._model:
            raise LlamaCppModelError("Model not loaded")

        try:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop,
                echo=echo,
            )

            end_time = time.time_ns()
            duration_ns = end_time - start_time

            choice = output.get("choices", [{}])[0]
            usage = output.get("usage", {})

            return LlamaCppResponse(
                model=self._model_info.name if self._model_info else "unknown",
                content=choice.get("text", ""),
                done=True,
                total_duration_ns=duration_ns,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                tokens_per_second=(
                    usage.get("completion_tokens", 0) / (duration_ns / 1_000_000_000)
                    if duration_ns > 0
                    else 0
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise LlamaCppError(f"Generation failed: {e}")

    def _generate_server(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: Optional[List[str]],
        echo: bool,
    ) -> LlamaCppResponse:
        """Generate using server mode (llama.cpp server API)."""
        start_time = time.time_ns()

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stop": stop or [],
            "echo": echo,
            "stream": False,
        }

        response = self._request("POST", "/completion", json=payload)
        end_time = time.time_ns()

        return LlamaCppResponse(
            model=response.get("model", "unknown"),
            content=response.get("content", ""),
            done=True,
            total_duration_ns=end_time - start_time,
            prompt_tokens=response.get("tokens_evaluated", 0),
            completion_tokens=response.get("tokens_predicted", 0),
            total_tokens=(
                response.get("tokens_evaluated", 0) + response.get("tokens_predicted", 0)
            ),
            tokens_per_second=response.get("tokens_per_second", 0),
            finish_reason=response.get("stop_type", "stop"),
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Generate text with streaming response.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Yields:
            Text chunks as they are generated
        """
        if self._mode == "server":
            yield from self._generate_stream_server(prompt, max_tokens, temperature, top_p, stop)
        else:
            yield from self._generate_stream_direct(prompt, max_tokens, temperature, top_p, stop)

    def _generate_stream_direct(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> Iterator[str]:
        """Stream generate using direct mode."""
        if not self._model:
            self.load_model()

        if not self._model:
            raise LlamaCppModelError("Model not loaded")

        try:
            for output in self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=True,
            ):
                choice = output.get("choices", [{}])[0]
                text = choice.get("text", "")
                if text:
                    yield text

        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            raise LlamaCppError(f"Stream generation failed: {e}")

    def _generate_stream_server(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> Iterator[str]:
        """Stream generate using server mode."""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or [],
            "stream": True,
        }

        try:
            with self._http_client.stream(
                "POST",
                urljoin(self.endpoint, "/completion"),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        content = data.get("content", "")
                        if content:
                            yield content
                        if data.get("stop"):
                            break

        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            raise LlamaCppError(f"Stream generation failed: {e}")

    def chat(
        self,
        messages: List[LlamaCppMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> LlamaCppResponse:
        """
        Chat completion with message history.

        Args:
            messages: List of messages (conversation history)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            LlamaCppResponse with assistant's reply
        """
        if self._mode == "server":
            return self._chat_server(messages, max_tokens, temperature, top_p, stop)
        else:
            return self._chat_direct(messages, max_tokens, temperature, top_p, stop)

    def _chat_direct(
        self,
        messages: List[LlamaCppMessage],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> LlamaCppResponse:
        """Chat using direct mode."""
        if not self._model:
            self.load_model()

        if not self._model:
            raise LlamaCppModelError("Model not loaded")

        start_time = time.time_ns()

        try:
            output = self._model.create_chat_completion(
                messages=[m.to_dict() for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

            end_time = time.time_ns()
            duration_ns = end_time - start_time

            choice = output.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = output.get("usage", {})

            return LlamaCppResponse(
                model=output.get("model", self._model_info.name if self._model_info else "unknown"),
                content=message.get("content", ""),
                done=True,
                total_duration_ns=duration_ns,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                tokens_per_second=(
                    usage.get("completion_tokens", 0) / (duration_ns / 1_000_000_000)
                    if duration_ns > 0
                    else 0
                ),
                finish_reason=choice.get("finish_reason", "stop"),
            )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise LlamaCppError(f"Chat failed: {e}")

    def _chat_server(
        self,
        messages: List[LlamaCppMessage],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> LlamaCppResponse:
        """Chat using server mode (OpenAI-compatible API)."""
        start_time = time.time_ns()

        payload = {
            "messages": [m.to_dict() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "stream": False,
        }

        response = self._request("POST", "/v1/chat/completions", json=payload)
        end_time = time.time_ns()

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})

        return LlamaCppResponse(
            model=response.get("model", "unknown"),
            content=message.get("content", ""),
            done=True,
            total_duration_ns=end_time - start_time,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    def chat_stream(
        self,
        messages: List[LlamaCppMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Chat completion with streaming response.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Yields:
            Text chunks as they are generated
        """
        if self._mode == "server":
            yield from self._chat_stream_server(messages, max_tokens, temperature, stop)
        else:
            yield from self._chat_stream_direct(messages, max_tokens, temperature, stop)

    def _chat_stream_direct(
        self,
        messages: List[LlamaCppMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
    ) -> Iterator[str]:
        """Chat stream using direct mode."""
        if not self._model:
            self.load_model()

        if not self._model:
            raise LlamaCppModelError("Model not loaded")

        try:
            for output in self._model.create_chat_completion(
                messages=[m.to_dict() for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True,
            ):
                choice = output.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            raise LlamaCppError(f"Chat stream failed: {e}")

    def _chat_stream_server(
        self,
        messages: List[LlamaCppMessage],
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]],
    ) -> Iterator[str]:
        """Chat stream using server mode."""
        payload = {
            "messages": [m.to_dict() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "stream": True,
        }

        try:
            with self._http_client.stream(
                "POST",
                urljoin(self.endpoint, "/v1/chat/completions"),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break
                        data = json.loads(line[6:])
                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content

        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            raise LlamaCppError(f"Chat stream failed: {e}")

    # ==========================================================================
    # Embeddings
    # ==========================================================================

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self._mode == "server":
            response = self._request(
                "POST",
                "/v1/embeddings",
                json={"input": text, "model": "default"},
            )
            data = response.get("data", [{}])[0]
            return data.get("embedding", [])
        else:
            if not self._model:
                self.load_model()

            if not self._model:
                raise LlamaCppModelError("Model not loaded")

            try:
                embedding = self._model.embed(text)
                return embedding if isinstance(embedding, list) else list(embedding)
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                raise LlamaCppError(f"Embedding failed: {e}")

    # ==========================================================================
    # Health Check
    # ==========================================================================

    def is_healthy(self) -> bool:
        """Check if Llama.cpp is running and healthy."""
        if self._mode == "server":
            try:
                self._request("GET", "/health")
                return True
            except Exception:
                return False
        else:
            return self._model is not None

    def wait_for_ready(
        self,
        timeout: float = 30.0,
        interval: float = 1.0,
    ) -> bool:
        """
        Wait for Llama.cpp to become ready.

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
        """Make an HTTP request with retry logic (server mode only)."""
        if not self._http_client:
            raise LlamaCppError("HTTP client not initialized")

        url = urljoin(self.endpoint, path)

        last_error = None
        for attempt in range(self.retry_count):
            try:
                response = self._http_client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json() if response.content else {}

            except httpx.ConnectError as e:
                last_error = LlamaCppConnectionError(f"Cannot connect to llama.cpp server: {e}")
                logger.warning(f"Connection failed (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay)

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = LlamaCppError(f"Server error: {e}")
                    time.sleep(self.retry_delay)
                else:
                    raise LlamaCppError(f"Request failed: {e}")

            except Exception as e:
                raise LlamaCppError(f"Request error: {e}")

        raise last_error or LlamaCppError("Request failed after retries")


class LlamaCppModelManager:
    """
    Manages Llama.cpp models for Agent OS.

    Provides:
    - Model discovery and listing
    - Model loading/unloading
    - Resource management
    """

    def __init__(
        self,
        client: Optional[LlamaCppClient] = None,
        model_dir: Optional[str] = None,
    ):
        """
        Initialize model manager.

        Args:
            client: Llama.cpp client
            model_dir: Directory containing GGUF models
        """
        self.client = client
        self.model_dir = model_dir or os.environ.get("LLAMA_MODEL_DIR", ".")
        self._loaded_models: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def list_available_models(self) -> List[LlamaCppModelInfo]:
        """List all available GGUF models."""
        if self.client:
            return self.client.list_models(self.model_dir)
        return []

    def ensure_model(self, model_path: str) -> bool:
        """
        Ensure a model is loaded.

        Args:
            model_path: Path to model file

        Returns:
            True if model is loaded/available
        """
        if not self.client:
            return False

        with self._lock:
            try:
                self.client.load_model(model_path)
                self._loaded_models[model_path] = datetime.now()
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False

    def get_loaded_models(self) -> List[str]:
        """Get list of models that have been loaded."""
        return list(self._loaded_models.keys())


def create_llama_cpp_client(
    model_path: Optional[str] = None,
    endpoint: Optional[str] = None,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    timeout: float = 120.0,
) -> LlamaCppClient:
    """
    Create a Llama.cpp client.

    For direct mode: provide model_path or set LLAMA_MODEL_PATH env var.
    For server mode: provide endpoint or set LLAMA_CPP_ENDPOINT env var.

    Args:
        model_path: Path to GGUF model file
        endpoint: Llama.cpp server endpoint
        n_ctx: Context window size
        n_gpu_layers: GPU layers (-1 = all)
        timeout: Request timeout

    Returns:
        LlamaCppClient instance
    """
    # Try environment variables if not provided
    if not model_path and not endpoint:
        endpoint = os.environ.get("LLAMA_CPP_ENDPOINT")
        model_path = os.environ.get("LLAMA_MODEL_PATH")

    if not model_path and not endpoint:
        raise ValueError(
            "Provide model_path or endpoint, or set LLAMA_MODEL_PATH or LLAMA_CPP_ENDPOINT"
        )

    return LlamaCppClient(
        model_path=model_path,
        endpoint=endpoint,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        timeout=timeout,
    )
