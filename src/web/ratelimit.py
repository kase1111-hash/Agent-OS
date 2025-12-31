"""
Rate Limiting System

Provides configurable rate limiting for API endpoints with multiple
strategies and storage backends.

Strategies:
- Fixed Window: Simple counter reset at fixed intervals
- Sliding Window: Smoother rate limiting with sliding time windows
- Token Bucket: Allows bursting with steady refill rate

Example:
    from src.web.ratelimit import RateLimiter, RateLimitMiddleware

    # Create limiter
    limiter = RateLimiter(
        requests_per_minute=60,
        strategy="sliding_window",
    )

    # Use as middleware
    app.add_middleware(RateLimitMiddleware, limiter=limiter)

    # Or use decorator for specific endpoints
    @limiter.limit("10/minute")
    async def my_endpoint():
        pass
"""

import asyncio
import functools
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: float  # Unix timestamp
    retry_after: Optional[float] = None  # Seconds until retry

    @property
    def reset_at_datetime(self) -> datetime:
        """Get reset time as datetime."""
        return datetime.fromtimestamp(self.reset_at)


@dataclass
class RateLimitRule:
    """A rate limit rule."""

    requests: int
    window_seconds: int
    key_prefix: str = ""
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    @classmethod
    def parse(cls, spec: str, key_prefix: str = "") -> "RateLimitRule":
        """
        Parse a rate limit specification.

        Format: "requests/period" where period is:
        - second, minute, hour, day
        - or number of seconds

        Examples:
        - "10/second"
        - "60/minute"
        - "1000/hour"
        - "100/30" (100 requests per 30 seconds)
        """
        parts = spec.lower().split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid rate limit spec: {spec}")

        requests = int(parts[0])
        period = parts[1].strip()

        period_map = {
            "second": 1,
            "sec": 1,
            "s": 1,
            "minute": 60,
            "min": 60,
            "m": 60,
            "hour": 3600,
            "hr": 3600,
            "h": 3600,
            "day": 86400,
            "d": 86400,
        }

        if period in period_map:
            window_seconds = period_map[period]
        else:
            window_seconds = int(period)

        return cls(
            requests=requests,
            window_seconds=window_seconds,
            key_prefix=key_prefix,
        )


class RateLimitStorage(ABC):
    """Abstract base for rate limit storage backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get rate limit data for a key."""
        pass

    @abstractmethod
    async def set(self, key: str, data: Dict[str, Any], ttl: int) -> None:
        """Set rate limit data with TTL."""
        pass

    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter and return new value."""
        pass


class InMemoryStorage(RateLimitStorage):
    """In-memory rate limit storage."""

    def __init__(self):
        self._data: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get rate limit data for a key."""
        async with self._lock:
            if key not in self._data:
                return None
            data, expires_at = self._data[key]
            if time.time() > expires_at:
                del self._data[key]
                return None
            return data

    async def set(self, key: str, data: Dict[str, Any], ttl: int) -> None:
        """Set rate limit data with TTL."""
        async with self._lock:
            expires_at = time.time() + ttl
            self._data[key] = (data, expires_at)

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter and return new value."""
        async with self._lock:
            if key not in self._data:
                return amount
            data, expires_at = self._data[key]
            if time.time() > expires_at:
                return amount
            count = data.get("count", 0) + amount
            data["count"] = count
            self._data[key] = (data, expires_at)
            return count

    async def cleanup(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        async with self._lock:
            now = time.time()
            expired = [k for k, (_, exp) in self._data.items() if now > exp]
            for key in expired:
                del self._data[key]
            return len(expired)


class RedisStorage(RateLimitStorage):
    """Redis-based rate limit storage."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis

                self._client = redis.from_url(self._redis_url)
            except ImportError:
                raise ImportError("redis package required for Redis storage")
        return self._client

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get rate limit data for a key."""
        import json

        client = await self._get_client()
        data = await client.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def set(self, key: str, data: Dict[str, Any], ttl: int) -> None:
        """Set rate limit data with TTL."""
        import json

        client = await self._get_client()
        await client.setex(key, ttl, json.dumps(data))

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter and return new value."""
        client = await self._get_client()
        return await client.incrby(key, amount)


class RateLimiter:
    """
    Rate limiter with configurable strategies and storage.

    Example:
        limiter = RateLimiter(
            requests_per_minute=60,
            requests_per_hour=1000,
        )

        # Check if request is allowed
        result = await limiter.check("user:123")
        if not result.allowed:
            return Response(status_code=429)
    """

    def __init__(
        self,
        requests_per_second: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        strategy: Union[str, RateLimitStrategy] = RateLimitStrategy.SLIDING_WINDOW,
        storage: Optional[RateLimitStorage] = None,
        key_prefix: str = "ratelimit:",
        burst_multiplier: float = 1.5,
    ):
        self.rules: List[RateLimitRule] = []
        self.strategy = (
            RateLimitStrategy(strategy) if isinstance(strategy, str) else strategy
        )
        self.storage = storage or InMemoryStorage()
        self.key_prefix = key_prefix
        self.burst_multiplier = burst_multiplier

        # Add configured limits
        if requests_per_second:
            self.rules.append(
                RateLimitRule(requests_per_second, 1, f"{key_prefix}sec:", self.strategy)
            )
        if requests_per_minute:
            self.rules.append(
                RateLimitRule(requests_per_minute, 60, f"{key_prefix}min:", self.strategy)
            )
        if requests_per_hour:
            self.rules.append(
                RateLimitRule(requests_per_hour, 3600, f"{key_prefix}hr:", self.strategy)
            )
        if requests_per_day:
            self.rules.append(
                RateLimitRule(requests_per_day, 86400, f"{key_prefix}day:", self.strategy)
            )

    def add_rule(self, rule: Union[RateLimitRule, str]) -> None:
        """Add a rate limit rule."""
        if isinstance(rule, str):
            rule = RateLimitRule.parse(rule, self.key_prefix)
        self.rules.append(rule)

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """
        Check if a request is allowed under rate limits.

        Args:
            key: Unique identifier for the rate limit subject (e.g., user ID, IP)
            cost: Cost of this request (default 1)

        Returns:
            RateLimitResult indicating if request is allowed
        """
        if not self.rules:
            # No rules configured, allow all
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                limit=999999,
                reset_at=time.time() + 60,
            )

        results: List[RateLimitResult] = []

        for rule in self.rules:
            if self.strategy == RateLimitStrategy.FIXED_WINDOW:
                result = await self._check_fixed_window(key, rule, cost)
            elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
                result = await self._check_sliding_window(key, rule, cost)
            elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                result = await self._check_token_bucket(key, rule, cost)
            else:
                result = await self._check_fixed_window(key, rule, cost)

            results.append(result)

            # If any rule denies, stop checking
            if not result.allowed:
                return result

        # Return the most restrictive result
        if results:
            return min(results, key=lambda r: r.remaining)

        return RateLimitResult(
            allowed=True,
            remaining=999999,
            limit=999999,
            reset_at=time.time() + 60,
        )

    async def _check_fixed_window(
        self, key: str, rule: RateLimitRule, cost: int
    ) -> RateLimitResult:
        """Fixed window rate limiting."""
        now = time.time()
        window_start = int(now / rule.window_seconds) * rule.window_seconds
        window_key = f"{rule.key_prefix}{key}:{window_start}"

        data = await self.storage.get(window_key)
        count = data.get("count", 0) if data else 0

        reset_at = window_start + rule.window_seconds
        remaining = max(0, rule.requests - count - cost)

        if count + cost > rule.requests:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=rule.requests,
                reset_at=reset_at,
                retry_after=reset_at - now,
            )

        # Increment counter
        new_count = count + cost
        await self.storage.set(
            window_key,
            {"count": new_count, "window_start": window_start},
            rule.window_seconds,
        )

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=rule.requests,
            reset_at=reset_at,
        )

    async def _check_sliding_window(
        self, key: str, rule: RateLimitRule, cost: int
    ) -> RateLimitResult:
        """Sliding window rate limiting."""
        now = time.time()
        window_start = now - rule.window_seconds
        window_key = f"{rule.key_prefix}{key}:sliding"

        data = await self.storage.get(window_key)
        if data is None:
            data = {"requests": [], "count": 0}

        # Filter out old requests
        requests = [ts for ts in data.get("requests", []) if ts > window_start]
        count = len(requests)

        reset_at = now + rule.window_seconds
        remaining = max(0, rule.requests - count - cost)

        if count + cost > rule.requests:
            # Find when the oldest request will expire
            if requests:
                oldest = min(requests)
                retry_after = oldest + rule.window_seconds - now
            else:
                retry_after = rule.window_seconds

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=rule.requests,
                reset_at=reset_at,
                retry_after=max(0, retry_after),
            )

        # Add current request(s)
        for _ in range(cost):
            requests.append(now)

        await self.storage.set(
            window_key,
            {"requests": requests[-rule.requests:], "count": len(requests)},
            rule.window_seconds * 2,  # Keep data a bit longer
        )

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=rule.requests,
            reset_at=reset_at,
        )

    async def _check_token_bucket(
        self, key: str, rule: RateLimitRule, cost: int
    ) -> RateLimitResult:
        """Token bucket rate limiting."""
        now = time.time()
        bucket_key = f"{rule.key_prefix}{key}:bucket"

        # Calculate refill rate (tokens per second)
        refill_rate = rule.requests / rule.window_seconds
        max_tokens = int(rule.requests * self.burst_multiplier)

        data = await self.storage.get(bucket_key)
        if data is None:
            tokens = max_tokens
            last_refill = now
        else:
            tokens = data.get("tokens", max_tokens)
            last_refill = data.get("last_refill", now)

            # Refill tokens based on time elapsed
            elapsed = now - last_refill
            tokens = min(max_tokens, tokens + elapsed * refill_rate)

        reset_at = now + rule.window_seconds

        if tokens < cost:
            # Calculate wait time for enough tokens
            needed = cost - tokens
            retry_after = needed / refill_rate

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=rule.requests,
                reset_at=reset_at,
                retry_after=retry_after,
            )

        # Consume tokens
        tokens -= cost

        await self.storage.set(
            bucket_key,
            {"tokens": tokens, "last_refill": now},
            rule.window_seconds * 2,
        )

        return RateLimitResult(
            allowed=True,
            remaining=int(tokens),
            limit=rule.requests,
            reset_at=reset_at,
        )

    def limit(
        self, rule: Union[str, RateLimitRule]
    ) -> Callable[[Callable], Callable]:
        """
        Decorator for rate limiting specific endpoints.

        Usage:
            @limiter.limit("10/minute")
            async def my_endpoint():
                pass
        """
        if isinstance(rule, str):
            rule = RateLimitRule.parse(rule, self.key_prefix)

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Try to get request from args
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                if request is None:
                    # No request found, just call function
                    return await func(*args, **kwargs)

                key = self._get_key(request)
                result = await self._check_single_rule(key, rule, 1)

                if not result.allowed:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "rate_limit_exceeded",
                            "detail": "Too many requests",
                            "retry_after": result.retry_after,
                        },
                        headers={
                            "X-RateLimit-Limit": str(result.limit),
                            "X-RateLimit-Remaining": str(result.remaining),
                            "X-RateLimit-Reset": str(int(result.reset_at)),
                            "Retry-After": str(int(result.retry_after or 1)),
                        },
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    async def _check_single_rule(
        self, key: str, rule: RateLimitRule, cost: int
    ) -> RateLimitResult:
        """Check a single rule."""
        if self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(key, rule, cost)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(key, rule, cost)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(key, rule, cost)
        return await self._check_fixed_window(key, rule, cost)

    def _get_key(self, request: Request) -> str:
        """Get rate limit key from request."""
        # Try authenticated user first
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Fall back to IP address
        client = request.client
        if client:
            return f"ip:{client.host}"

        # Last resort: use a hash of headers
        headers_str = str(sorted(request.headers.items()))
        return f"anon:{hashlib.md5(headers_str.encode()).hexdigest()[:16]}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app: ASGIApp,
        limiter: Optional[RateLimiter] = None,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        exclude_paths: Optional[List[str]] = None,
        key_func: Optional[Callable[[Request], str]] = None,
    ):
        super().__init__(app)
        self.limiter = limiter or RateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
        )
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
        ]
        self.key_func = key_func

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiter."""
        path = request.url.path

        # Skip excluded paths
        if any(path.startswith(exc) for exc in self.exclude_paths):
            return await call_next(request)

        # Get rate limit key
        if self.key_func:
            key = self.key_func(request)
        else:
            key = self._default_key(request)

        # Check rate limit
        result = await self.limiter.check(key)

        if not result.allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": "Too many requests. Please slow down.",
                    "retry_after": result.retry_after,
                    "limit": result.limit,
                    "remaining": result.remaining,
                },
                headers={
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(result.reset_at)),
                    "Retry-After": str(int(result.retry_after or 1)),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))

        return response

    def _default_key(self, request: Request) -> str:
        """Default key extraction from request."""
        # Try to get user from state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Use IP address
        client = request.client
        if client:
            return f"ip:{client.host}"

        return "unknown"


# Convenience function for creating configured limiter
def create_limiter(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    strategy: str = "sliding_window",
    use_redis: bool = False,
    redis_url: str = "redis://localhost:6379",
) -> RateLimiter:
    """
    Create a configured rate limiter.

    Args:
        requests_per_minute: Max requests per minute
        requests_per_hour: Max requests per hour
        strategy: Rate limiting strategy (fixed_window, sliding_window, token_bucket)
        use_redis: Use Redis for storage (for distributed deployments)
        redis_url: Redis connection URL

    Returns:
        Configured RateLimiter instance
    """
    storage: RateLimitStorage
    if use_redis:
        storage = RedisStorage(redis_url)
    else:
        storage = InMemoryStorage()

    return RateLimiter(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        strategy=RateLimitStrategy(strategy),
        storage=storage,
    )


# Global limiter instance
_limiter: Optional[RateLimiter] = None


def get_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(
            requests_per_minute=60,
            requests_per_hour=1000,
        )
    return _limiter


def set_limiter(limiter: RateLimiter) -> None:
    """Set the global rate limiter."""
    global _limiter
    _limiter = limiter
