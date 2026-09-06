"""Throttling for the two places a credential can be guessed.

Classroom codes are short by design and passwords are chosen by people, so
both the join route and the login route count failures and refuse a client
that has had too many recently. In-process and in-memory: the backend is one
uvicorn process on one VM, so there is no second instance to share state with,
and a restart forgetting the counters costs nothing.

Failures, not attempts, are what count. A class of forty students joining in
the same minute from behind one NAT address must not be locked out by each
other's successes.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Dict

from fastapi import Request

Clock = Callable[[], float]


class RateLimiter:
    """Sliding window: at most `limit` failures per key per `window_seconds`."""

    def __init__(self, *, limit: int, window_seconds: float, clock: Clock = time.monotonic) -> None:
        self.limit = limit
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._failures: Dict[str, Deque[float]] = {}

    def _prune(self, key: str, now: float) -> Deque[float]:
        bucket = self._failures.get(key)
        if bucket is None:
            bucket = deque()
            self._failures[key] = bucket
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        return bucket

    def retry_after(self, key: str) -> float | None:
        """Seconds until `key` may try again, or None when it may try now."""
        now = self._clock()
        bucket = self._prune(key, now)
        if len(bucket) < self.limit:
            if not bucket:
                self._failures.pop(key, None)
            return None
        return max(1.0, bucket[0] + self.window_seconds - now)

    def record_failure(self, key: str) -> None:
        now = self._clock()
        self._prune(key, now).append(now)
        if len(self._failures) > 10_000:
            self._sweep(now)

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)

    def clear(self) -> None:
        self._failures.clear()

    def _sweep(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for key in list(self._failures):
            bucket = self._failures[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                del self._failures[key]


#: Join-code guessing. Ten wrong codes from one client in ten minutes is far
#: more than a typo budget; three hundred across everyone is far more than a
#: lecture hall's worth of typos.
JOIN_CODE_PER_CLIENT = RateLimiter(limit=10, window_seconds=600)
JOIN_CODE_GLOBAL = RateLimiter(limit=300, window_seconds=600)

#: Password guessing, per account and per client.
LOGIN_PER_ACCOUNT = RateLimiter(limit=10, window_seconds=900)
LOGIN_PER_CLIENT = RateLimiter(limit=30, window_seconds=900)

#: Invitation-token guessing. A 256-bit token cannot be guessed, but a client
#: hammering the accept route is still not a client worth serving.
INVITE_PER_CLIENT = RateLimiter(limit=20, window_seconds=600)

GLOBAL_KEY = "*"

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def client_address(request: Request) -> str:
    """The address to throttle on.

    Uvicorn sits behind Nginx on the same host, so `request.client.host` is
    the loopback address for every browser. When that is the case and Nginx
    forwarded the real client in `X-Forwarded-For`, the last entry — the one
    Nginx itself appended — is used. A direct connection from anywhere else is
    taken at face value, and a header from a non-loopback client is ignored
    because anyone can send one.

    If Nginx does not forward the header at all, every client shares one key,
    which is stricter than intended rather than looser.
    """
    direct = request.client.host if request.client else ""
    if direct in _LOOPBACK:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            candidate = forwarded.split(",")[-1].strip()
            if candidate:
                return candidate
    return direct or "unknown"


def reset_all_limiters() -> None:
    for limiter in (
        JOIN_CODE_PER_CLIENT,
        JOIN_CODE_GLOBAL,
        LOGIN_PER_ACCOUNT,
        LOGIN_PER_CLIENT,
        INVITE_PER_CLIENT,
    ):
        limiter.clear()
