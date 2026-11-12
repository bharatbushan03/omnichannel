"""In-process sliding-window rate limit for webhook endpoints.

One process, one window. A multi-worker deployment needs a shared store in
front of this (the limit is per worker otherwise). The default is 60 requests
per minute per client address, override with ``WEBHOOK_RATE_LIMIT``.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int = 60, window_s: float = 60.0):
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_s
        with self._lock:
            hits = [stamp for stamp in self._hits.get(key, []) if stamp > cutoff]
            if len(hits) >= self.max_requests:
                self._hits[key] = hits
                return False
            hits.append(current)
            self._hits[key] = hits
            return True


def limiter_from_env() -> SlidingWindowRateLimiter:
    raw = os.environ.get("WEBHOOK_RATE_LIMIT", "60")
    try:
        max_requests = int(raw)
    except ValueError:
        max_requests = 60
    return SlidingWindowRateLimiter(max_requests=max(1, max_requests))


def client_key(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_rate_limit(request: Request, limiter: SlidingWindowRateLimiter) -> None:
    if not limiter.allow(client_key(request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
