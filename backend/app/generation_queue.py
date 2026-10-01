"""One bounded queue in front of every generation this backend asks for.

Why it exists
-------------
The VM has one CPU-only Ollama doing all generation, and the four Compare
conditions used to reach it through two locks that did not know about each
other: the backend's (Base, RAG, starter jobs) and the fine-tuned service's
(Fine-Tuned, Fine-Tuned + RAG). So two generations were always in flight
during a class, one for the base model and one for the course adapter. Those
two models share one weights file, and Ollama keeps at most one runner per
weights file, so every hand-off between them unloaded one model and loaded the
other (see docs/classroom-capacity.md). Nothing bounded the wait either: a
Base request waited for the backend lock for as long as Nginx allowed, and a
fine-tuned request spent its 120 s timeout queueing behind the service's lock
and came back 503 having done nothing.

What it does
------------
- At most `GENERATION_MAX_CONCURRENCY` generations run at once (default 1),
  counting the fine-tuned service's calls, which the backend makes. Raising it
  is a measured decision (`scripts/classroom_load_test.py`), not a default.
- A request that cannot start is queued. It is refused at once, with 503 and
  `{"code": "generation_busy"}`, when the queue is full or when the expected
  wait is clearly beyond the limit, and refused after
  `GENERATION_QUEUE_TIMEOUT_SECONDS` if its turn has not come. A refused
  request never reaches a model, so an abandoned request costs nothing.
- The order is by comparison, then arrival: the four requests one student's
  Compare makes share the server-side time their first request arrived, so a
  student who asked first finishes first rather than every student getting
  three answers out of four. Without a comparison id a request is its own
  comparison, which is plain arrival order.
- Interactive requests always go before background work (starter seed
  generation), which waits without a time limit as it always has.
- A request whose client has gone (tab closed, page refreshed) leaves the
  queue instead of running.

Nothing about a prompt, an option or a model is decided here; this module
only decides when a generation may start.
"""

from __future__ import annotations

import asyncio
import bisect
import contextvars
import itertools
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _ensure_visible_logs() -> None:
    """Make this logger's INFO lines reach the journal.

    Nothing in the backend configures logging, so an `app.*` INFO record falls
    through to Python's last-resort handler, which prints WARNING and above
    only. The per-generation timing line is INFO and is the point of this
    module's logging, so this one logger gets its own stderr handler (systemd
    sends stderr to the journal). `GENERATION_LOG_LEVEL=WARNING` quiets it.
    """
    level_name = (os.getenv("GENERATION_LOG_LEVEL") or "INFO").strip().upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)


_ensure_visible_logs()

Priority =Literal["interactive", "background"]

DEFAULT_MAX_CONCURRENCY = 1
#: A backstop on held requests, not the time limit (the queue timeout and the
#: wait estimate are). Thirty students' first wave is about ninety requests
#: (Base and both fine-tuned conditions at once); 64 refused a third of a
#: burst that then drained in two minutes. 160 is forty students' four.
DEFAULT_MAX_WAITING = 160
#: Plus the 120 s generation timeout, this stays under Nginx's 300 s
#: `proxy_read_timeout`, so a student is told "busy" by us rather than shown
#: a gateway timeout by the proxy.
DEFAULT_QUEUE_TIMEOUT_SECONDS = 150.0
#: How often a waiting request checks whether its client is still there.
DISCONNECT_POLL_SECONDS = 1.0
#: Comparison ids are remembered this long; long enough for the slowest
#: comparison, short enough that an old id cannot buy a place in line.
COMPARISON_TTL_SECONDS = 600.0
#: Weight of the newest service time in the running estimate.
EWMA_ALPHA = 0.3
#: Refuse up front only when the estimated wait exceeds the limit by this
#: much. The estimate is built from recent service times, which swing with the
#: mix of conditions and with Ollama's prompt cache (a repeated question is
#: far cheaper). Refusing on a borderline estimate turned away requests that
#: would have been served in time; a borderline request waits instead, and
#: the queue timeout still bounds it.
ESTIMATE_REFUSAL_MARGIN = 1.5

BUSY_CODE = "generation_busy"
BUSY_MESSAGE = (
    "Many people are asking at the same moment and this answer could not start "
    "in time. Please try again in a minute."
)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default
    return value if value >= 1 else default


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def configured_max_concurrency() -> int:
    return _positive_int_env("GENERATION_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)


def configured_max_waiting() -> int:
    return _positive_int_env("GENERATION_MAX_WAITING", DEFAULT_MAX_WAITING)


def configured_queue_timeout() -> float:
    return _positive_float_env("GENERATION_QUEUE_TIMEOUT_SECONDS", DEFAULT_QUEUE_TIMEOUT_SECONDS)


# --------------------------------------------------------------------------- #
# Request context: which comparison, and is the client still there
# --------------------------------------------------------------------------- #

DisconnectCheck = Callable[[], Awaitable[bool]]

_comparison_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "generation_comparison_id", default=None
)
_disconnect_check: contextvars.ContextVar[DisconnectCheck | None] = contextvars.ContextVar(
    "generation_disconnect_check", default=None
)

#: The header the Compare page sends with each of a comparison's four requests.
COMPARISON_HEADER = "x-comparison-id"
MAX_COMPARISON_ID_LENGTH = 64


def bind_request(request: Any) -> None:
    """Attach the current HTTP request's comparison id and disconnect check.

    Called at the top of a generate route, in the route's own task, so the
    values are visible to every await beneath it. A malformed or oversized id
    is ignored rather than refused: ordering is a courtesy, not a contract.
    """
    raw = (request.headers.get(COMPARISON_HEADER) or "").strip()
    if raw and len(raw) <= MAX_COMPARISON_ID_LENGTH and raw.isprintable():
        _comparison_id.set(raw)
    else:
        _comparison_id.set(None)
    _disconnect_check.set(request.is_disconnected)


class ClientDisconnected(HTTPException):
    """The client left while the request waited. Nobody reads this response."""

    def __init__(self) -> None:
        # 499 is Nginx's "client closed request"; it only ever reaches a log.
        super().__init__(status_code=499, detail="Client closed the request.")


def busy_exception(reason: str, retry_after: float) -> HTTPException:
    seconds = int(min(300, max(5, math.ceil(retry_after))))
    return HTTPException(
        status_code=503,
        detail={
            "code": BUSY_CODE,
            "reason": reason,
            "message": BUSY_MESSAGE,
            "retryAfterSeconds": seconds,
        },
        headers={"Retry-After": str(seconds)},
    )


def is_busy_exception(exc: BaseException) -> bool:
    return (
        isinstance(exc, HTTPException)
        and isinstance(exc.detail, dict)
        and exc.detail.get("code") == BUSY_CODE
    )


# --------------------------------------------------------------------------- #
# The queue
# --------------------------------------------------------------------------- #


@dataclass
class SlotInfo:
    """What a caller learns about its own turn, for its log line."""

    label: str
    queue_wait_seconds: float = 0.0
    ahead_at_entry: int = 0
    waiting_at_entry: int = 0


@dataclass(order=True)
class _Waiter:
    key: tuple[int, float, int]
    future: asyncio.Future = field(compare=False)
    priority: Priority = field(compare=False, default="interactive")


class GenerationQueue:
    def __init__(
        self,
        *,
        max_concurrency: int,
        max_waiting: int,
        queue_timeout: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_waiting = max(1, int(max_waiting))
        self.queue_timeout = float(queue_timeout)
        self._clock = clock
        self._active = 0
        self._waiters: list[_Waiter] = []
        self._sequence = itertools.count()
        self._comparisons: dict[str, float] = {}
        self._service_ewma: float | None = None
        self.counters = {
            "started": 0,
            "completed": 0,
            "rejectedFull": 0,
            "rejectedEstimate": 0,
            "timedOut": 0,
            "clientGone": 0,
        }

    # -- state ---------------------------------------------------------------

    @property
    def active(self) -> int:
        return self._active

    def _interactive_waiting(self) -> int:
        return sum(1 for w in self._waiters if w.priority == "interactive")

    def snapshot(self) -> dict[str, Any]:
        return {
            "maxConcurrency": self.max_concurrency,
            "maxWaiting": self.max_waiting,
            "queueTimeoutSeconds": self.queue_timeout,
            "active": self._active,
            "waiting": self._interactive_waiting(),
            "backgroundWaiting": len(self._waiters) - self._interactive_waiting(),
            "serviceSecondsEstimate": (
                round(self._service_ewma, 2) if self._service_ewma is not None else None
            ),
            **self.counters,
        }

    def estimated_wait(self, ahead: int) -> float | None:
        """Seconds until a request with `ahead` interactive waiters in front starts."""
        if self._service_ewma is None:
            return None
        return (ahead + self._active) / self.max_concurrency * self._service_ewma

    def _record_service(self, seconds: float) -> None:
        if self._service_ewma is None:
            self._service_ewma = seconds
        else:
            self._service_ewma = EWMA_ALPHA * seconds + (1 - EWMA_ALPHA) * self._service_ewma

    def _group_time(self, comparison_id: str | None, now: float) -> float:
        if comparison_id is None:
            return now
        for stale in [cid for cid, seen in self._comparisons.items() if now - seen > COMPARISON_TTL_SECONDS]:
            del self._comparisons[stale]
        return self._comparisons.setdefault(comparison_id, now)

    # -- hand-off ------------------------------------------------------------

    def _release(self) -> None:
        while self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.future.done():
                # The slot passes straight to the next waiter; `_active` is unchanged.
                waiter.future.set_result(None)
                return
        self._active -= 1

    def _remove(self, waiter: _Waiter) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            pass

    async def _wait_turn(
        self,
        waiter: _Waiter,
        *,
        deadline: float | None,
        disconnected: DisconnectCheck | None,
    ) -> str | None:
        """Wait for the slot. Returns None when granted, else why not."""
        while True:
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is not None and remaining <= 0:
                return "queue_timeout"
            step = DISCONNECT_POLL_SECONDS if disconnected is not None else remaining
            if remaining is not None and step is not None:
                step = min(step, remaining)
            done, _ = await asyncio.wait({waiter.future}, timeout=step)
            if done:
                return None
            if disconnected is not None and await disconnected():
                return "client_gone"

    @asynccontextmanager
    async def slot(
        self,
        label: str,
        *,
        priority: Priority = "interactive",
    ) -> AsyncIterator[SlotInfo]:
        info = SlotInfo(label=label)
        comparison_id = _comparison_id.get() if priority == "interactive" else None
        disconnected = _disconnect_check.get() if priority == "interactive" else None
        now = self._clock()
        waited = False
        # Recorded even when this request starts at once, so the comparison's
        # later requests (RAG after Base) keep the place it began with.
        group = self._group_time(comparison_id, now) if priority == "interactive" else now

        if self._active < self.max_concurrency and not self._waiters:
            self._active += 1
        else:
            waited = True
            rank = 0 if priority == "interactive" else 1
            waiter = _Waiter(
                key=(rank, group, next(self._sequence)),
                future=asyncio.get_running_loop().create_future(),
                priority=priority,
            )
            index = bisect.bisect(self._waiters, waiter)
            ahead = sum(1 for w in self._waiters[:index] if w.priority == "interactive")
            info.ahead_at_entry = ahead
            info.waiting_at_entry = self._interactive_waiting()

            if priority == "interactive":
                if self._interactive_waiting() >= self.max_waiting:
                    self.counters["rejectedFull"] += 1
                    estimate = self.estimated_wait(ahead) or self.queue_timeout
                    _log_rejection(label, "queue_full", info)
                    raise busy_exception("queue_full", estimate)
                estimate = self.estimated_wait(ahead)
                if estimate is not None and estimate > self.queue_timeout * ESTIMATE_REFUSAL_MARGIN:
                    self.counters["rejectedEstimate"] += 1
                    _log_rejection(label, "estimated_wait", info, estimate=estimate)
                    raise busy_exception("estimated_wait", estimate)

            self._waiters.insert(index, waiter)
            deadline = now + self.queue_timeout if priority == "interactive" else None
            try:
                outcome = await self._wait_turn(waiter, deadline=deadline, disconnected=disconnected)
            except BaseException:
                self._abandon(waiter)
                raise
            if outcome is not None:
                self._abandon(waiter)
                info.queue_wait_seconds = self._clock() - now
                if outcome == "client_gone":
                    self.counters["clientGone"] += 1
                    _log_rejection(label, "client_gone", info)
                    raise ClientDisconnected()
                self.counters["timedOut"] += 1
                _log_rejection(label, "queue_timeout", info)
                raise busy_exception(
                    "queue_timeout", self.estimated_wait(info.waiting_at_entry) or self.queue_timeout
                )

        info.queue_wait_seconds = self._clock() - now
        # The slot is held from here on; every path below must release it.
        try:
            if waited and disconnected is not None and await disconnected():
                self.counters["clientGone"] += 1
                _log_rejection(label, "client_gone", info)
                raise ClientDisconnected()
        except BaseException:
            self._release()
            raise

        self.counters["started"] += 1
        started = self._clock()
        try:
            yield info
        finally:
            if priority == "interactive":
                self._record_service(self._clock() - started)
            self.counters["completed"] += 1
            self._release()

    def _abandon(self, waiter: _Waiter) -> None:
        """A waiter that gave up. If the slot reached it in the meantime, pass it on."""
        self._remove(waiter)
        if waiter.future.done() and not waiter.future.cancelled():
            self._release()
        else:
            waiter.future.cancel()


def _log_rejection(label: str, reason: str, info: SlotInfo, *, estimate: float | None = None) -> None:
    logger.warning(
        "generation condition=%s outcome=%s reason=%s queue_wait_ms=%d ahead=%d waiting=%d%s",
        label,
        "client_gone" if reason == "client_gone" else "busy",
        reason,
        int(info.queue_wait_seconds * 1000),
        info.ahead_at_entry,
        info.waiting_at_entry,
        f" estimated_wait_s={estimate:.0f}" if estimate is not None else "",
    )


_queue: GenerationQueue | None = None


def get_generation_queue() -> GenerationQueue:
    """The process-wide queue, built from the environment on first use."""
    global _queue
    if _queue is None:
        _queue = GenerationQueue(
            max_concurrency=configured_max_concurrency(),
            max_waiting=configured_max_waiting(),
            queue_timeout=configured_queue_timeout(),
        )
    return _queue


def generation_slot(label: str, *, priority: Priority = "interactive"):
    return get_generation_queue().slot(label, priority=priority)


def reset_generation_queue_for_tests(queue: GenerationQueue | None = None) -> None:
    global _queue
    _queue = queue


# --------------------------------------------------------------------------- #
# One log line per generation
# --------------------------------------------------------------------------- #

#: Ollama's response durations are nanoseconds; these are the ones worth a log.
_OLLAMA_DURATIONS = (
    ("load_duration", "load_ms"),
    ("prompt_eval_duration", "prompt_eval_ms"),
    ("eval_duration", "eval_ms"),
    ("total_duration", "ollama_total_ms"),
)
_OLLAMA_COUNTS = (("prompt_eval_count", "prompt_tokens"), ("eval_count", "output_tokens"))
TIMING_KEYS = tuple(k for _, k in _OLLAMA_DURATIONS) + tuple(k for _, k in _OLLAMA_COUNTS)


def ollama_timings(data: Any) -> dict[str, int]:
    """Milliseconds and token counts from an Ollama `/api/chat` or `/api/generate` body.

    `load_ms` is the number that shows a model swap: near zero when the runner
    was resident, seconds when Ollama had to (re)load it for this request.
    """
    if not isinstance(data, dict):
        return {}
    timings: dict[str, int] = {}
    for source, target in _OLLAMA_DURATIONS:
        value = data.get(source)
        if isinstance(value, (int, float)) and value >= 0:
            timings[target] = int(value / 1_000_000)
    for source, target in _OLLAMA_COUNTS:
        value = data.get(source)
        if isinstance(value, int) and value >= 0:
            timings[target] = value
    return timings


def clean_timings(raw: Any) -> dict[str, int]:
    """Timings another service reported, restricted to the known integer keys."""
    if not isinstance(raw, dict):
        return {}
    return {
        key: int(raw[key])
        for key in TIMING_KEYS
        if isinstance(raw.get(key), (int, float)) and not isinstance(raw.get(key), bool) and raw[key] >= 0
    }


def log_generation(
    info: SlotInfo | None,
    *,
    model: str,
    outcome: str,
    elapsed_seconds: float,
    timings: dict[str, int] | None = None,
    reason: str | None = None,
) -> None:
    """`generation condition=… model=… outcome=… queue_wait_ms=… load_ms=…`.

    No question text, prompt or answer: the line is for capacity, and a
    student's question does not belong in a service log.
    """
    parts = [
        f"condition={info.label if info else '-'}",
        f"model={model}",
        f"outcome={outcome}",
    ]
    if reason:
        parts.append(f"reason={reason}")
    if info is not None:
        parts.append(f"queue_wait_ms={int(info.queue_wait_seconds * 1000)}")
        parts.append(f"ahead={info.ahead_at_entry}")
    parts.append(f"elapsed_ms={int(elapsed_seconds * 1000)}")
    for key in TIMING_KEYS:
        if timings and key in timings:
            parts.append(f"{key}={timings[key]}")
    log = logger.info if outcome == "ok" else logger.warning
    log("generation %s", " ".join(parts))
