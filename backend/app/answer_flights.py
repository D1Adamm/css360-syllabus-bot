"""One generation for everyone asking the same thing at the same moment.

When several requests need the identical answer (same course, approach,
question and configuration) while it is still being generated, they share one
generation instead of each taking a turn in the queue. Twenty students asking
one uncached question cost one generation per approach, not twenty.

This is not a queue and does not replace one. The shared generation takes one
ordinary turn in `app.generation_queue`, with the priority and limits of the
request that started it; the others simply wait for its result and never enter
the queue, so they add nothing to its length or its wait estimate.

Rules
-----
- A shared generation is abandoned only when *every* request waiting on it has
  gone. One student closing a tab changes nothing for the rest.
- Whatever it ends with, an answer or a failure, is what everyone waiting
  receives. Nothing is remembered afterwards: the next request starts afresh,
  so a failure is never replayed and a retry is a real retry.
- A student request never waits on background work that has not started. If
  classroom preparation is still waiting for its turn on the same answer, the
  student's request starts its own generation ahead of it.
- Nothing here stores anything. Sharing lasts exactly as long as the
  generation does.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Hashable, TypeVar

from app.generation_queue import (
    DisconnectCheck,
    busy_exception,
    current_disconnect_check,
    is_background_generation,
    shared_request,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Flight:
    """One generation in progress and the requests waiting on it."""

    key: Hashable
    background: bool
    started: bool = False
    #: One entry per interactive request waiting; None means "cannot tell",
    #: which counts as still there.
    checks: list[DisconnectCheck | None] = field(default_factory=list)
    task: "asyncio.Task[Any] | None" = None

    def mark_started(self) -> None:
        self.started = True

    async def everyone_gone(self) -> bool:
        """True only when every waiting request's client has disconnected."""
        if not self.checks:
            return False
        for check in list(self.checks):
            if check is None:
                return False
            try:
                if not await check():
                    return False
            except Exception:  # noqa: BLE001 - unknown is "still there"
                return False
        return True

    def joinable_by(self, *, background: bool) -> bool:
        if self.task is None or self.task.done():
            return False
        # Preparation may wait on anything; a student only on work that is
        # already running or that another student started.
        return background or not self.background or self.started


_flights: dict[Hashable, Flight] = {}


def active_flight(key: Hashable) -> Flight | None:
    flight = _flights.get(key)
    return flight if flight is not None and flight.task is not None and not flight.task.done() else None


def active_flight_count() -> int:
    return sum(1 for key in list(_flights) if active_flight(key) is not None)


def cancel_background(*, include_started: bool = False) -> int:
    """Cancel background generations, by default only ones still waiting for a turn.

    Stopping classroom preparation uses this: work that has not reached a model
    is dropped at once, work that is already generating is left to finish (its
    answer is kept). `include_started` is the last resort for a generation
    that never returns; the model may go on computing after it, which is why
    it is not the ordinary way to stop.
    """
    cancelled = 0
    for flight in list(_flights.values()):
        if not flight.background or flight.task is None or flight.task.done():
            continue
        if flight.started and not include_started:
            continue
        flight.task.cancel()
        cancelled += 1
    return cancelled


def background_generation_started() -> bool:
    """Whether a background generation currently holds a turn at a model."""
    return any(
        flight.background and flight.started and flight.task is not None and not flight.task.done()
        for flight in _flights.values()
    )


def reset_flights_for_tests() -> None:
    _flights.clear()


def _finished(flight: Flight, task: "asyncio.Task[Any]") -> None:
    if _flights.get(flight.key) is flight:
        del _flights[flight.key]
    # Mark the result as seen: when every waiter has gone, nobody awaits the
    # task, and asyncio would otherwise log an unretrieved exception.
    if not task.cancelled():
        task.exception()


async def share(key: Hashable, work: Callable[[], Awaitable[T]]) -> T:
    """Run `work` once for every concurrent caller with the same `key`.

    The first caller starts it; later callers wait for the same result. Each
    caller is shielded from the others: cancelling one caller does not cancel
    the generation.
    """
    background = is_background_generation()
    flight = _flights.get(key)
    if flight is None or not flight.joinable_by(background=background):
        flight = Flight(key=key, background=background)

        async def run() -> T:
            with shared_request(flight.everyone_gone, on_started=flight.mark_started):
                return await work()

        # The task copies this caller's context: its comparison id (the place
        # in line) and, for preparation, its background priority.
        flight.task = asyncio.create_task(run())
        flight.task.add_done_callback(lambda task, flight=flight: _finished(flight, task))
        _flights[key] = flight

    if not background:
        flight.checks.append(current_disconnect_check())
    assert flight.task is not None
    try:
        return await asyncio.shield(flight.task)
    except asyncio.CancelledError:
        # The shared generation itself was cancelled (preparation was forced to
        # stop) rather than this caller: a student who had joined it is told
        # to try again instead of having the request die.
        if flight.task.cancelled() and not background:
            raise busy_exception("generation_cancelled", 5) from None
        raise
