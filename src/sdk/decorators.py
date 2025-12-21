"""
Agent Decorators

Provides decorators for enhancing agent functionality including:
- Request/response logging
- Performance monitoring
- Error handling
- Rate limiting
- Caching
"""

import functools
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from src.messaging.models import FlowRequest, FlowResponse, MessageStatus


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def log_requests(
    level: int = logging.INFO,
    include_content: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to log all requests to an agent.

    Args:
        level: Logging level
        include_content: Include request content in logs

    Example:
        class MyAgent(BaseAgent):
            @log_requests()
            def process(self, request: FlowRequest) -> FlowResponse:
                ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, request: FlowRequest, *args, **kwargs) -> FlowResponse:
            start = time.time()

            msg = f"Request received: intent={request.intent}, source={request.source}"
            if include_content:
                msg += f", prompt={request.content.prompt[:100]}..."

            logger.log(level, msg)

            response = func(self, request, *args, **kwargs)

            duration = (time.time() - start) * 1000
            logger.log(
                level,
                f"Request completed: status={response.status.name}, duration={duration:.1f}ms"
            )

            return response

        return cast(F, wrapper)
    return decorator


def log_responses(
    level: int = logging.INFO,
    include_output: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to log all responses from an agent.

    Args:
        level: Logging level
        include_output: Include response output in logs
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, request: FlowRequest, *args, **kwargs) -> FlowResponse:
            response = func(self, request, *args, **kwargs)

            msg = f"Response: status={response.status.name}"
            if include_output:
                output = response.content.output[:100]
                msg += f", output={output}..."

            logger.log(level, msg)
            return response

        return cast(F, wrapper)
    return decorator


def measure_time(
    metric_name: Optional[str] = None,
    callback: Optional[Callable[[str, float], None]] = None,
) -> Callable[[F], F]:
    """
    Decorator to measure execution time.

    Args:
        metric_name: Name for the metric
        callback: Callback to receive timing data

    Example:
        @measure_time("process_request")
        def process(self, request):
            ...
    """
    def decorator(func: F) -> F:
        name = metric_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000

            logger.debug(f"{name} took {duration:.1f}ms")

            if callback:
                callback(name, duration)

            return result

        return cast(F, wrapper)
    return decorator


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to retry function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Initial delay between attempts
        backoff_factor: Multiply delay by this each attempt
        exceptions: Exception types to catch

    Example:
        @retry(max_attempts=3, delay_seconds=0.5)
        def call_api(self, ...):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = delay_seconds

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

        return cast(F, wrapper)
    return decorator


def catch_errors(
    default_status: MessageStatus = MessageStatus.ERROR,
    log_level: int = logging.ERROR,
) -> Callable[[F], F]:
    """
    Decorator to catch and handle errors gracefully.

    Args:
        default_status: Status to return on error
        log_level: Level to log errors at

    Example:
        @catch_errors()
        def process(self, request: FlowRequest) -> FlowResponse:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, request: FlowRequest, *args, **kwargs) -> FlowResponse:
            try:
                return func(self, request, *args, **kwargs)
            except Exception as e:
                logger.log(log_level, f"{func.__name__} error: {e}", exc_info=True)
                return request.create_response(
                    source=getattr(self, 'name', 'unknown'),
                    status=default_status,
                    output="",
                    errors=[str(e)],
                )

        return cast(F, wrapper)
    return decorator


class RateLimiter:
    """Rate limiter for tracking request rates."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Check if request is allowed."""
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)

            if key not in self._requests:
                self._requests[key] = []

            # Clean old entries
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]

            # Check limit
            if len(self._requests[key]) >= self.max_requests:
                return False

            # Record request
            self._requests[key].append(now)
            return True


_rate_limiters: Dict[str, RateLimiter] = {}


def rate_limit(
    max_requests: int,
    window_seconds: int = 60,
    key_fn: Optional[Callable[[FlowRequest], str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to rate limit requests.

    Args:
        max_requests: Max requests per window
        window_seconds: Window size in seconds
        key_fn: Function to extract rate limit key from request

    Example:
        @rate_limit(max_requests=10, window_seconds=60)
        def process(self, request: FlowRequest) -> FlowResponse:
            ...
    """
    def decorator(func: F) -> F:
        limiter_key = f"{func.__module__}.{func.__qualname__}"

        if limiter_key not in _rate_limiters:
            _rate_limiters[limiter_key] = RateLimiter(max_requests, window_seconds)

        limiter = _rate_limiters[limiter_key]

        @functools.wraps(func)
        def wrapper(self, request: FlowRequest, *args, **kwargs) -> FlowResponse:
            # Get rate limit key
            if key_fn:
                limit_key = key_fn(request)
            else:
                limit_key = request.source

            if not limiter.check(limit_key):
                return request.create_response(
                    source=getattr(self, 'name', 'unknown'),
                    status=MessageStatus.REFUSED,
                    output="Rate limit exceeded. Please try again later.",
                )

            return func(self, request, *args, **kwargs)

        return cast(F, wrapper)
    return decorator


class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                value, expires = self._cache[key]
                if datetime.now() < expires:
                    return value
                del self._cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            expires = datetime.now() + timedelta(seconds=self.ttl_seconds)
            self._cache[key] = (value, expires)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_caches: Dict[str, SimpleCache] = {}


def cache_response(
    ttl_seconds: int = 300,
    key_fn: Optional[Callable[[FlowRequest], str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to cache responses.

    Args:
        ttl_seconds: Time to live for cache entries
        key_fn: Function to generate cache key from request

    Example:
        @cache_response(ttl_seconds=60)
        def process(self, request: FlowRequest) -> FlowResponse:
            ...
    """
    def decorator(func: F) -> F:
        cache_key = f"{func.__module__}.{func.__qualname__}"

        if cache_key not in _caches:
            _caches[cache_key] = SimpleCache(ttl_seconds)

        cache = _caches[cache_key]

        @functools.wraps(func)
        def wrapper(self, request: FlowRequest, *args, **kwargs) -> FlowResponse:
            # Generate cache key
            if key_fn:
                entry_key = key_fn(request)
            else:
                entry_key = f"{request.intent}:{request.content.prompt}"

            # Check cache
            cached = cache.get(entry_key)
            if cached is not None:
                logger.debug(f"Cache hit for {entry_key}")
                return cached

            # Call function and cache result
            response = func(self, request, *args, **kwargs)

            if response.is_success():
                cache.set(entry_key, response)

            return response

        return cast(F, wrapper)
    return decorator


def validate_request(
    required_fields: Optional[List[str]] = None,
    min_prompt_length: int = 0,
    max_prompt_length: int = 10000,
) -> Callable[[F], F]:
    """
    Decorator to validate incoming requests.

    Args:
        required_fields: Required metadata fields
        min_prompt_length: Minimum prompt length
        max_prompt_length: Maximum prompt length

    Example:
        @validate_request(min_prompt_length=5)
        def process(self, request: FlowRequest) -> FlowResponse:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, request: FlowRequest, *args, **kwargs) -> FlowResponse:
            prompt = request.content.prompt
            errors = []

            # Check prompt length
            if len(prompt) < min_prompt_length:
                errors.append(f"Prompt too short (min {min_prompt_length} chars)")

            if len(prompt) > max_prompt_length:
                errors.append(f"Prompt too long (max {max_prompt_length} chars)")

            # Check required fields
            if required_fields:
                for field in required_fields:
                    if field not in request.metadata:
                        errors.append(f"Missing required field: {field}")

            if errors:
                return request.create_response(
                    source=getattr(self, 'name', 'unknown'),
                    status=MessageStatus.REFUSED,
                    output="Request validation failed",
                    errors=errors,
                )

            return func(self, request, *args, **kwargs)

        return cast(F, wrapper)
    return decorator


def require_capability(
    *capabilities,
) -> Callable[[F], F]:
    """
    Decorator to assert agent has required capabilities.

    Raises RuntimeError if agent lacks capabilities.

    Example:
        @require_capability(CapabilityType.REASONING)
        def analyze(self, ...):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            agent_caps = self.get_capabilities()

            missing = set(capabilities) - agent_caps.capabilities
            if missing:
                raise RuntimeError(
                    f"Agent missing required capabilities: {[c.value for c in missing]}"
                )

            return func(self, *args, **kwargs)

        return cast(F, wrapper)
    return decorator
