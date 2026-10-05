"""Tests for rate limit storage backends (src/web/ratelimit.py)."""

import asyncio
import json

import pytest

from src.web.ratelimit import RateLimiter, RateLimitMiddleware, RedisStorage

UNREACHABLE_REDIS = "redis://127.0.0.1:1"  # nothing listens on port 1


class FakeRedis:
    """Minimal async stand-in for redis.asyncio.Redis."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0
        self.data = {}

    def _maybe_fail(self):
        self.calls += 1
        if self.fail:
            raise ConnectionError("redis down")

    async def get(self, key):
        self._maybe_fail()
        return self.data.get(key)

    async def setex(self, key, ttl, value):
        self._maybe_fail()
        self.data[key] = value

    async def incrby(self, key, amount):
        self._maybe_fail()
        return amount


class TestRedisStorageFallback:
    def test_limits_still_enforced_when_redis_unreachable(self):
        limiter = RateLimiter(requests_per_minute=2, storage=RedisStorage(UNREACHABLE_REDIS))

        async def run():
            return [(await limiter.check("ip:1")).allowed for _ in range(3)]

        assert asyncio.run(run()) == [True, True, False]

    def test_uses_redis_when_available(self):
        storage = RedisStorage(UNREACHABLE_REDIS)
        storage._client = FakeRedis()

        asyncio.run(storage.set("k", {"count": 1}, 60))

        assert json.loads(storage._client.data["k"]) == {"count": 1}
        assert asyncio.run(storage.get("k")) == {"count": 1}

    def test_backs_off_then_retries_redis(self):
        storage = RedisStorage(UNREACHABLE_REDIS)
        client = storage._client = FakeRedis(fail=True)

        asyncio.run(storage.set("k", {"count": 1}, 60))
        assert asyncio.run(storage.get("k")) == {"count": 1}  # served from fallback
        assert client.calls == 1  # no Redis retry inside the back-off window

        storage._retry_at = 0.0
        client.fail = False
        asyncio.run(storage.set("k", {"count": 2}, 60))
        assert client.calls == 2
        assert json.loads(client.data["k"]) == {"count": 2}


class TestRateLimitMiddlewareWithRedisDown:
    def test_requests_succeed_when_redis_unreachable(self):
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        app = fastapi.FastAPI()

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        limiter = RateLimiter(requests_per_minute=2, storage=RedisStorage(UNREACHABLE_REDIS))
        app.add_middleware(RateLimitMiddleware, limiter=limiter)

        with TestClient(app) as client:
            statuses = [client.get("/ping").status_code for _ in range(3)]

        assert statuses == [200, 200, 429]
