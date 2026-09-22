#!/usr/bin/env python3
"""Shared pieces for the production verification and latency tools on the UWB VM.

Standard library only, so both tools run under the VM's `python3` as well as
the backend virtualenv. Nothing here stores a credential: the administrator
password is read from the environment or an interactive prompt, exchanged for
a session cookie that lives in memory for the length of one run, and the
session is ended with `/api/auth/logout` before the process exits.

What is deliberately not done: no HTTP cookie jar. The backend marks its
cookies `Secure`, which `http.cookiejar` honours by refusing to send them
over the plain-HTTP loopback these tools use (the same `curl 127.0.0.1:8001`
path every runbook uses). The `Set-Cookie` values are kept by name instead
and sent back verbatim.
"""

from __future__ import annotations

import getpass
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

#: `backend/app/auth/settings.py`; the header a cross-site page cannot set.
CSRF_HEADER_NAME = "x-requested-with"
CSRF_HEADER_VALUE = "SyllabusModelLab"

DEFAULT_BACKEND_URL = "http://127.0.0.1:8001"
DEFAULT_SERVICE_URL = "http://127.0.0.1:9001"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_ENV_FILE = "~/.config/aiswe/finetuned.env"
KNOWN_COURSES = ("css-350-spring-2026-n3h9", "css-360-winter-2026-a7rp")
#: Generic, contains nothing about a person, and every course syllabus answers it.
PROBE_QUESTION = "When does the course meet?"

PASSWORD_ENV = "AISWE_VERIFY_PASSWORD"
EMAIL_ENV = "AISWE_VERIFY_EMAIL"

MODE_ROUTES = {
    "base": "/api/base-model/generate",
    "rag": "/api/rag/generate",
    "fineTuned": "/api/fine-tuned/generate",
    "fineTunedRag": "/api/fine-tuned-rag/generate",
}


@dataclass
class HttpResult:
    status: int
    body: Any
    seconds: float
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300


class Transport:
    """One HTTP call. Replaced by a fake in tests."""

    def request(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> HttpResult:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request_headers = {"Accept": "application/json"}
        if data is not None:
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback
                raw = response.read()
                status = response.status
                response_headers = {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
            response_headers = {k.lower(): v for k, v in exc.headers.items()}
        except TimeoutError:
            return HttpResult(0, None, time.perf_counter() - started, error="timeout", timed_out=True)
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            timed_out = isinstance(reason, TimeoutError) or "timed out" in str(reason).lower()
            return HttpResult(0, None, time.perf_counter() - started, error=str(reason), timed_out=timed_out)
        elapsed = time.perf_counter() - started
        parsed: Any = None
        if raw:
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except ValueError:
                parsed = None
        return HttpResult(status, parsed, elapsed, headers=response_headers)


class BackendSession:
    """A signed-in administrator session against the backend, or an anonymous one."""

    def __init__(self, base_url: str, transport: Transport | None = None, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or Transport()
        self.timeout = timeout
        self._cookies: dict[str, str] = {}
        self.signed_in = False

    def _headers(self, *, csrf: bool) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._cookies:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in self._cookies.items())
        if csrf:
            headers[CSRF_HEADER_NAME] = CSRF_HEADER_VALUE
        return headers

    def _absorb_cookies(self, headers: dict[str, str]) -> None:
        # urllib folds repeated Set-Cookie headers into one comma-joined value.
        raw = headers.get("set-cookie")
        if not raw:
            return
        for part in raw.split(","):
            pair = part.strip().split(";", 1)[0]
            if "=" in pair:
                name, value = pair.split("=", 1)
                name = name.strip()
                if name and (value or name in self._cookies):
                    if value:
                        self._cookies[name] = value
                    else:
                        self._cookies.pop(name, None)

    def get(self, path: str, *, timeout: float | None = None) -> HttpResult:
        return self.transport.request("GET", self.base_url + path, headers=self._headers(csrf=False), timeout=timeout or self.timeout)

    def post(self, path: str, body: dict[str, Any], *, timeout: float | None = None) -> HttpResult:
        return self.transport.request("POST", self.base_url + path, body=body, headers=self._headers(csrf=True), timeout=timeout or self.timeout)

    def login(self, email: str, password: str) -> HttpResult:
        result = self.transport.request(
            "POST",
            self.base_url + "/api/auth/login",
            body={"email": email, "password": password},
            headers=self._headers(csrf=True),
            timeout=self.timeout,
        )
        if result.ok:
            self._absorb_cookies(result.headers)
            self.signed_in = bool(self._cookies)
        return result

    def logout(self) -> None:
        if not self.signed_in:
            return
        try:
            self.transport.request("POST", self.base_url + "/api/auth/logout", body={}, headers=self._headers(csrf=True), timeout=self.timeout)
        finally:
            self._cookies.clear()
            self.signed_in = False


def resolve_admin_credentials(email_arg: str | None, *, env: dict[str, str] | None = None, prompt: bool = True) -> tuple[str, str] | None:
    """`(email, password)` from the argument/environment, or None meaning SKIP.

    The password is never an argument (it would land in shell history and
    `ps`). It comes from `AISWE_VERIFY_PASSWORD`, or, when stdin is a
    terminal, from a prompt that echoes nothing.
    """
    environment = os.environ if env is None else env
    email = (email_arg or environment.get(EMAIL_ENV) or "").strip()
    if not email:
        return None
    password = environment.get(PASSWORD_ENV) or ""
    if not password and prompt and sys.stdin.isatty():
        password = getpass.getpass(f"Password for {email} (not stored): ")
    if not password:
        return None
    return email, password


# --------------------------------------------------------------------------- #
# Generation checks
# --------------------------------------------------------------------------- #


def generate(session: BackendSession, mode: str, course_id: str, *, question: str = PROBE_QUESTION, timeout: float | None = None) -> HttpResult:
    return session.post(MODE_ROUTES[mode], {"courseId": course_id, "question": question}, timeout=timeout)


def judge_generation(result: HttpResult, *, course_id: str, mode: str, allowed_versions: Iterable[str] | None = None) -> tuple[bool, str]:
    """PASS/FAIL for one classroom-route answer, without quoting the answer.

    The course on the response must be the one asked for; a fine-tuned mode
    must report `adapterLoaded` and a version, and, when the mapping is known,
    a version this host maps for the course.
    """
    if result.timed_out:
        return False, f"timed out after {result.seconds:.0f}s"
    if not result.ok:
        detail = ""
        if isinstance(result.body, dict):
            detail = str(result.body.get("detail") or "")[:160]
        return False, f"HTTP {result.status} {result.error or detail}".strip()
    body = result.body if isinstance(result.body, dict) else {}
    answer = body.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return False, "empty answer"
    if body.get("courseId") != course_id:
        return False, f"courseId {body.get('courseId')!r} != {course_id!r}"
    detail = f"{len(answer.strip())} chars, {result.seconds:.1f}s"
    if mode in ("fineTuned", "fineTunedRag"):
        if body.get("adapterLoaded") is not True:
            return False, "adapterLoaded is not true"
        version = body.get("modelVersion")
        if not isinstance(version, str) or not version:
            return False, "no modelVersion on the response"
        if allowed_versions is not None and version not in set(allowed_versions):
            return False, f"modelVersion {version} is not mapped for {course_id} (mapped: {', '.join(allowed_versions) or 'none'})"
        detail = f"{version}, " + detail
    return True, detail


def judge_direct_generation(result: HttpResult, *, course_id: str, version: str) -> tuple[bool, str]:
    """The same for the service's own `/generate`, which echoes course and version."""
    if result.timed_out:
        return False, f"timed out after {result.seconds:.0f}s"
    if not result.ok:
        detail = ""
        if isinstance(result.body, dict):
            detail = str(result.body.get("detail") or "")[:160]
        return False, f"HTTP {result.status} {result.error or detail}".strip()
    body = result.body if isinstance(result.body, dict) else {}
    answer = body.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return False, "empty answer"
    if body.get("courseId") != course_id:
        return False, f"courseId {body.get('courseId')!r} != {course_id!r}"
    if body.get("modelVersion") != version:
        return False, f"modelVersion {body.get('modelVersion')!r} != {version!r}"
    if body.get("adapterLoaded") is not True:
        return False, "adapterLoaded is not true"
    return True, f"{body.get('model')}, {len(answer.strip())} chars, {result.seconds:.1f}s"


# --------------------------------------------------------------------------- #
# Latency arithmetic
# --------------------------------------------------------------------------- #


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; `fraction` in [0, 1]. None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def summarize_latencies(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and quantiles over `{ok, seconds, timedOut}` samples.

    Quantiles are over successful requests only: a timed-out request has no
    latency, only a timeout, and mixing the two would make p95 a function of
    the timeout setting.
    """
    successes = [s["seconds"] for s in samples if s.get("ok")]
    return {
        "requests": len(samples),
        "ok": len(successes),
        "failed": sum(1 for s in samples if not s.get("ok")),
        "timeouts": sum(1 for s in samples if s.get("timedOut")),
        "median": statistics.median(successes) if successes else None,
        "p95": percentile(successes, 0.95),
        "max": max(successes) if successes else None,
        "min": min(successes) if successes else None,
    }


def format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}s"


Check = tuple[str, str, str]  # status, name, detail


def print_checks(checks: list[Check], out: Callable[[str], None] = print) -> bool:
    """Compact PASS/FAIL/SKIP lines. Returns True when nothing failed."""
    for status, name, detail in checks:
        out(f"{status:<4} {name}" + (f"  ({detail})" if detail else ""))
    counts = {s: sum(1 for c in checks if c[0] == s) for s in ("PASS", "FAIL", "SKIP", "WARN")}
    out("summary: " + ", ".join(f"{k.lower()} {v}" for k, v in counts.items() if v))
    return counts["FAIL"] == 0
