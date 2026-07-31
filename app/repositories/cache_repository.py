"""Cache with Redis primary + in-memory TTL fallback.

The Redis connection is attempted once at construction. If Redis is
unavailable (e.g. no Docker running), every operation silently falls back to a
process-local dict with the same TTL semantics — so the API works with *zero*
infrastructure. This fallback is exactly what keeps the app functional on
deployments without managed Redis.
"""

import threading
import time

import redis


class CacheRepository:
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self._redis_url = redis_url
        self._ttl_seconds = max(1, ttl_seconds)
        self._memory: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._client: redis.Redis | None = None
        try:
            client = redis.Redis.from_url(
                redis_url, socket_connect_timeout=1.0, socket_timeout=1.0
            )
            client.ping()
            self._client = client
        except Exception:
            self._client = None  # Redis unavailable → in-memory fallback

    @property
    def connected(self) -> bool:
        return self._client is not None

    def get(self, key: str) -> str | None:
        if self._client is not None:
            try:
                raw = self._client.get(key)
                return raw.decode("utf-8") if raw else None
            except Exception:
                pass  # transient Redis failure → fall back to memory
        with self._lock:
            item = self._memory.get(key)
            if item is None:
                return None
            expires_at, value = item
            if time.monotonic() >= expires_at:
                self._memory.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str) -> None:
        if self._client is not None:
            try:
                self._client.setex(key, self._ttl_seconds, value)
                return
            except Exception:
                pass
        with self._lock:
            self._memory[key] = (time.monotonic() + self._ttl_seconds, value)

    def delete(self, key: str) -> None:
        if self._client is not None:
            try:
                self._client.delete(key)
                return
            except Exception:
                pass
        with self._lock:
            self._memory.pop(key, None)
