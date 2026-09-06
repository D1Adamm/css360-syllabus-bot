"""The failure throttle behind join codes and logins."""

from __future__ import annotations

import unittest

from starlette.requests import Request

from app.auth.rate_limit import RateLimiter, client_address


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class RateLimiterTests(unittest.TestCase):
    def test_allows_up_to_the_limit_then_refuses(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(limit=3, window_seconds=60, clock=clock)

        for _ in range(3):
            self.assertIsNone(limiter.retry_after("k"))
            limiter.record_failure("k")

        retry = limiter.retry_after("k")
        self.assertIsNotNone(retry)
        assert retry is not None
        self.assertGreater(retry, 0)
        self.assertLessEqual(retry, 60)

    def test_the_window_slides(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
        limiter.record_failure("k")
        clock.now += 30
        limiter.record_failure("k")
        self.assertIsNotNone(limiter.retry_after("k"))

        clock.now += 31  # the first failure has aged out
        self.assertIsNone(limiter.retry_after("k"))

    def test_keys_are_independent(self) -> None:
        limiter = RateLimiter(limit=1, window_seconds=60, clock=FakeClock())
        limiter.record_failure("a")
        self.assertIsNotNone(limiter.retry_after("a"))
        self.assertIsNone(limiter.retry_after("b"))

    def test_reset_forgives_a_key(self) -> None:
        limiter = RateLimiter(limit=1, window_seconds=60, clock=FakeClock())
        limiter.record_failure("a")
        limiter.reset("a")
        self.assertIsNone(limiter.retry_after("a"))

    def test_successes_are_not_counted(self) -> None:
        """Forty students behind one NAT joining correctly must not lock it."""
        limiter = RateLimiter(limit=2, window_seconds=60, clock=FakeClock())
        for _ in range(100):
            self.assertIsNone(limiter.retry_after("nat"))


def _request(client_host: str, headers: dict[str, str] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client_host, 12345),
    }
    return Request(scope)


class ClientAddressTests(unittest.TestCase):
    def test_loopback_with_forwarded_header_uses_the_last_hop(self) -> None:
        request = _request("127.0.0.1", {"X-Forwarded-For": "10.0.0.5, 203.0.113.9"})
        self.assertEqual(client_address(request), "203.0.113.9")

    def test_loopback_without_header_is_loopback(self) -> None:
        self.assertEqual(client_address(_request("127.0.0.1")), "127.0.0.1")

    def test_a_direct_client_cannot_spoof_with_the_header(self) -> None:
        request = _request("198.51.100.7", {"X-Forwarded-For": "1.2.3.4"})
        self.assertEqual(client_address(request), "198.51.100.7")


if __name__ == "__main__":
    unittest.main()
