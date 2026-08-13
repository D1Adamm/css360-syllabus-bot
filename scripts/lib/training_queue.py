"""Durable training queue over the Firebase Realtime Database (stdlib only).

Runs are stored course-scoped at:

    courses/{courseId}/trainingRuns/{runId}

This module is the cluster side of that queue: it discovers queued work, and
claims exactly one run with a time-limited lease. It performs no training, does
not know what a scheduler is, and never shells out — everything about actually
submitting a job stays in `training/start_qlora_training.sh`, which already owns
versioned output directories and the live-adapter guard.

Why REST and not a client library
---------------------------------
Tillicum has outbound HTTPS to the database and nothing else is needed:
`urllib` with the same URL and token handling `scripts/register_course_model.py`
already uses. Adding an SDK to a login node for four HTTP calls would be a new
dependency to install, pin, and keep working on a machine we do not control.

How two runners are kept apart
------------------------------
The database has no transactions over REST, but it does have conditional
requests. A claim is:

    GET  .../trainingRuns/{runId}.json   with `X-Firebase-ETag: true`
    PUT  .../trainingRuns/{runId}.json   with `if-match: <that etag>`

The PUT succeeds only if nothing changed in between. A second runner racing for
the same run gets HTTP 412 and takes nothing — it does not retry the same run,
because by then someone else legitimately holds it.

The lease expires. A runner that dies mid-run — a dropped login session, a
rebooted node — would otherwise strand a run as permanently claimed. Once
`expiresAt` has passed the run becomes claimable again and the attempt count
goes up, so a run that keeps failing this way is visible as one that keeps
being retried rather than one that quietly vanished.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

COURSE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

VALID_MODES = ("smoke", "full")

RUN_STATES = ("queued", "claimed", "submitted", "training", "succeeded", "failed")
TERMINAL_RUN_STATES = ("succeeded", "failed")

DEFAULT_LEASE_SECONDS = 900
REQUEST_TIMEOUT_SECONDS = 30


class TrainingQueueError(Exception):
    """Anything that stops the queue being read or written."""


class ClaimConflict(TrainingQueueError):
    """Another runner changed the run first; it is not ours."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def load_env_file(root: Path = PROJECT_ROOT) -> None:
    """Read `.env.local` / `.env` the way the other scripts in this repo do."""
    for name in (".env.local", ".env"):
        path = root / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def firebase_database_url() -> str:
    url = (
        os.environ.get("FIREBASE_DATABASE_URL")
        or os.environ.get("VITE_FIREBASE_DATABASE_URL")
        or ""
    ).strip().rstrip("/")
    if not url:
        raise TrainingQueueError(
            "Missing Firebase database URL. Set FIREBASE_DATABASE_URL or "
            "VITE_FIREBASE_DATABASE_URL in the environment or .env.local."
        )
    return url


def firebase_auth_token() -> Optional[str]:
    token = (os.environ.get("FIREBASE_AUTH_TOKEN") or "").strip()
    return token or None


def validate_course_id(course_id: str) -> str:
    """Mirror the frontend and backend rule so a bad id cannot reach a path."""
    candidate = (course_id or "").strip()
    if (
        not candidate
        or not COURSE_ID_PATTERN.match(candidate)
        or ".." in candidate
        or "/" in candidate
    ):
        raise TrainingQueueError(f"Invalid courseId: {course_id!r}")
    return candidate


def validate_run_id(run_id: str) -> str:
    candidate = (run_id or "").strip()
    if not candidate or not RUN_ID_PATTERN.match(candidate) or "/" in candidate:
        raise TrainingQueueError(f"Invalid runId: {run_id!r}")
    return candidate


def course_training_runs_path(course_id: str) -> str:
    return f"courses/{validate_course_id(course_id)}/trainingRuns"


def course_training_run_path(course_id: str, run_id: str) -> str:
    return f"{course_training_runs_path(course_id)}/{validate_run_id(run_id)}"


def course_model_request_path(course_id: str) -> str:
    return f"courses/{validate_course_id(course_id)}/modelRequest"


def validate_slurm_job_id(job_id: str) -> str:
    """A real scheduler id is digits only. Placeholders are refused."""
    value = (job_id or "").strip()
    if not value.isdigit():
        raise TrainingQueueError(f"Invalid Slurm job id: {job_id!r}")
    return value


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: str


#: (method, url, body, headers) -> HttpResponse. Injected in tests so nothing
#: here reaches the network on its own.
Transport = Callable[[str, str, Optional[bytes], dict[str, str]], HttpResponse]


def urllib_transport(
    method: str,
    url: str,
    body: Optional[bytes],
    headers: dict[str, str],
) -> HttpResponse:
    request = urllib.request.Request(url, data=body, method=method)
    for name, value in headers.items():
        request.add_header(name, value)

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return HttpResponse(
                status=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read().decode("utf-8"),
            )
    except urllib.error.HTTPError as exc:
        # A 412 is an ordinary outcome of a contested claim, not an error to
        # raise through — the caller decides what a status means.
        return HttpResponse(
            status=exc.code,
            headers={key.lower(): value for key, value in (exc.headers or {}).items()},
            body=exc.read().decode("utf-8", errors="replace"),
        )
    except urllib.error.URLError as exc:
        raise TrainingQueueError(f"Could not reach Firebase: {exc.reason}") from exc


class FirebaseRest:
    """The four database calls this queue needs, and nothing else."""

    def __init__(
        self,
        *,
        database_url: str,
        auth_token: Optional[str] = None,
        transport: Transport = urllib_transport,
    ) -> None:
        self.database_url = database_url.rstrip("/")
        self.auth_token = auth_token
        self.transport = transport

    def url(self, path: str, query: str = "") -> str:
        url = f"{self.database_url}/{path}.json"
        parts = [part for part in (query, f"auth={self.auth_token}" if self.auth_token else "") if part]
        return f"{url}?{'&'.join(parts)}" if parts else url

    def _decode(self, response: HttpResponse, what: str) -> Any:
        if response.status == 401:
            raise TrainingQueueError(
                f"Firebase rejected the request for {what} (401). Provide "
                "FIREBASE_AUTH_TOKEN if the database rules require authentication."
            )
        if response.status >= 400:
            raise TrainingQueueError(
                f"Firebase request for {what} failed with HTTP {response.status}."
            )
        body = response.body.strip()
        if not body or body == "null":
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise TrainingQueueError(
                f"Firebase returned a payload for {what} that is not JSON: {exc.msg}"
            ) from exc

    def get(self, path: str, query: str = "") -> Any:
        response = self.transport("GET", self.url(path, query), None, {"Accept": "application/json"})
        return self._decode(response, path)

    def get_with_etag(self, path: str) -> tuple[Any, str]:
        """Read a node together with the tag a conditional write must match."""
        response = self.transport(
            "GET",
            self.url(path),
            None,
            {"Accept": "application/json", "X-Firebase-ETag": "true"},
        )
        value = self._decode(response, path)
        etag = response.headers.get("etag", "")
        if not etag:
            raise TrainingQueueError(
                f"Firebase did not return an ETag for {path}; a claim cannot be "
                "made safely without one."
            )
        return value, etag

    def put_if_match(self, path: str, value: Any, etag: str) -> Any:
        """Write only if the node still looks exactly as it did when read."""
        response = self.transport(
            "PUT",
            self.url(path),
            json.dumps(value).encode("utf-8"),
            {"Content-Type": "application/json", "if-match": etag},
        )
        if response.status == 412:
            raise ClaimConflict(
                "Another runner changed this run first; it was not claimed here."
            )
        return self._decode(response, path)

    def patch(self, path: str, value: dict[str, Any]) -> Any:
        response = self.transport(
            "PATCH",
            self.url(path),
            json.dumps(value).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        return self._decode(response, path)


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainingRun:
    run_id: str
    course_id: str
    mode: str
    state: str
    enqueued_at: str
    updated_at: str
    dataset_ref: str
    approved_example_count: int
    train_examples: int
    validation_examples: int
    attempt: int
    job_id: str = ""
    claim: Optional[dict[str, str]] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_RUN_STATES


def _as_count(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value if value >= 0 else 0


def parse_run(run_id: str, raw: Any) -> Optional[TrainingRun]:
    """Read one stored run, or None when it is not usable.

    Deliberately strict about `courseId`, `state` and `mode`: a record missing
    any of them cannot be acted on, and guessing a default would mean claiming
    work whose shape nobody has actually described.
    """
    if not isinstance(raw, dict):
        return None

    course_id = raw.get("courseId")
    state = raw.get("state")
    mode = raw.get("mode")
    enqueued_at = raw.get("enqueuedAt")

    if (
        not isinstance(course_id, str)
        or not course_id.strip()
        or state not in RUN_STATES
        or mode not in VALID_MODES
        or not isinstance(enqueued_at, str)
        or not enqueued_at.strip()
    ):
        return None

    claim = raw.get("claim")
    parsed_claim: Optional[dict[str, str]] = None
    if isinstance(claim, dict):
        owner = claim.get("owner")
        claimed_at = claim.get("claimedAt")
        expires_at = claim.get("expiresAt")
        if (
            isinstance(owner, str)
            and owner.strip()
            and isinstance(claimed_at, str)
            and isinstance(expires_at, str)
        ):
            parsed_claim = {
                "owner": owner,
                "claimedAt": claimed_at,
                "expiresAt": expires_at,
            }

    updated_at = raw.get("updatedAt")
    raw_job_id = raw.get("jobId")
    job_id = raw_job_id.strip() if isinstance(raw_job_id, str) else ""

    return TrainingRun(
        run_id=run_id,
        course_id=course_id,
        mode=mode,
        state=state,
        enqueued_at=enqueued_at,
        updated_at=updated_at if isinstance(updated_at, str) else enqueued_at,
        dataset_ref=raw.get("datasetRef") if isinstance(raw.get("datasetRef"), str) else "",
        approved_example_count=_as_count(raw.get("approvedExampleCount")),
        train_examples=_as_count(raw.get("trainExamples")),
        validation_examples=_as_count(raw.get("validationExamples")),
        attempt=_as_count(raw.get("attempt")),
        job_id=job_id,
        claim=parsed_claim,
        raw=dict(raw),
    )


def parse_runs(payload: Any) -> list[TrainingRun]:
    if not isinstance(payload, dict):
        return []
    runs = [parse_run(run_id, raw) for run_id, raw in payload.items()]
    return sorted(
        (run for run in runs if run is not None),
        key=lambda run: run.enqueued_at,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def lease_expired(run: TrainingRun, now: datetime) -> bool:
    """Whether a held run may be taken again.

    An unreadable or missing expiry counts as expired. The alternative is a run
    nothing can ever pick up, which is worse than one picked up twice — the
    second claim still has to win a conditional write.
    """
    if not run.claim:
        return True
    expires_at = parse_iso(run.claim.get("expiresAt", ""))
    return expires_at is None or expires_at <= now


def is_claimable(run: TrainingRun, now: datetime) -> bool:
    # A real job id means a submission already happened. Never take it again.
    if run.job_id:
        return False
    if run.state == "queued":
        return True
    if run.state == "claimed":
        return lease_expired(run, now)
    # submitted / training belong to a job that exists; terminal states are done.
    return False


def select_next_run(runs: list[TrainingRun], now: datetime) -> Optional[TrainingRun]:
    """Oldest claimable run first, so nothing waits behind newer work."""
    claimable = [run for run in runs if is_claimable(run, now)]
    if not claimable:
        return None
    return min(claimable, key=lambda run: run.enqueued_at)


# --------------------------------------------------------------------------- #
# Queue operations
# --------------------------------------------------------------------------- #


class TrainingQueue:
    def __init__(self, client: FirebaseRest) -> None:
        self.client = client

    def list_course_ids(self) -> list[str]:
        """Course ids only — a shallow read, never the whole database."""
        payload = self.client.get("courses", query="shallow=true")
        if not isinstance(payload, dict):
            return []
        return sorted(
            course_id
            for course_id in payload
            if isinstance(course_id, str) and COURSE_ID_PATTERN.match(course_id)
        )

    def list_runs(self, course_id: str) -> list[TrainingRun]:
        return parse_runs(self.client.get(course_training_runs_path(course_id)))

    def discover_claimable(
        self,
        *,
        now: datetime,
        course_ids: Optional[list[str]] = None,
    ) -> list[tuple[str, TrainingRun]]:
        """Claimable runs across the courses asked for, oldest first.

        Passing `course_ids` keeps the runner to exactly those courses; nothing
        else is read.
        """
        courses = course_ids if course_ids is not None else self.list_course_ids()
        found: list[tuple[str, TrainingRun]] = []
        for course_id in courses:
            for run in self.list_runs(course_id):
                if is_claimable(run, now):
                    found.append((course_id, run))
        return sorted(found, key=lambda pair: pair[1].enqueued_at)

    def claim(
        self,
        course_id: str,
        run: TrainingRun,
        *,
        owner: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: Optional[datetime] = None,
    ) -> TrainingRun:
        """Take one run, or raise `ClaimConflict` and take nothing.

        The read immediately before the write is what makes this safe: the ETag
        it returns describes the run as it is right now, and the write is
        refused if anything about it changed — including another runner's claim
        landing microseconds earlier.
        """
        moment = now or utc_now()
        path = course_training_run_path(course_id, run.run_id)

        current_raw, etag = self.client.get_with_etag(path)
        current = parse_run(run.run_id, current_raw)
        if current is None:
            raise ClaimConflict(f"Run {run.run_id} is no longer readable.")
        if current.job_id:
            raise ClaimConflict(
                f"Run {run.run_id} already has job {current.job_id}; it was not claimed."
            )
        if not is_claimable(current, moment):
            raise ClaimConflict(
                f"Run {run.run_id} is {current.state} and held by "
                f"{(current.claim or {}).get('owner', 'another runner')}."
            )

        claimed = dict(current.raw)
        claimed.update(
            {
                "state": "claimed",
                "attempt": current.attempt + 1,
                "updatedAt": iso(moment),
                "claim": {
                    "owner": owner,
                    "claimedAt": iso(moment),
                    "expiresAt": iso(moment + timedelta(seconds=lease_seconds)),
                },
            }
        )

        self.client.put_if_match(path, claimed, etag)

        result = parse_run(run.run_id, claimed)
        if result is None:  # pragma: no cover - we just built this record
            raise TrainingQueueError("Claimed run could not be read back.")
        return result

    def release(
        self,
        course_id: str,
        run: TrainingRun,
        *,
        error: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> None:
        """Put a claimed run back on the queue.

        Used when the runner finishes without submitting anything. Leaving it
        `claimed` would make it wait out a lease for no reason, and marking it
        failed would be untrue — nothing was attempted.
        """
        moment = now or utc_now()
        patch: dict[str, Any] = {
            "state": "queued",
            "updatedAt": iso(moment),
            # Firebase deletes a key written as null; the lease must not linger.
            "claim": None,
        }
        if error is not None:
            patch["error"] = error
        self.client.patch(course_training_run_path(course_id, run.run_id), patch)

    def record_submission(
        self,
        course_id: str,
        run: TrainingRun,
        *,
        job_id: str,
        train_count: int,
        validation_count: int,
        now: Optional[datetime] = None,
    ) -> TrainingRun:
        """Persist a real job id, then point the request at the running job.

        Order matters. The run is marked ``submitted`` first so a crash before
        the request update cannot look like "nothing was queued". The request
        becomes ``training`` only once that id exists — a professor must never
        be told their model is training when no job can be looked up.

        ``failureMessage`` is not written. That field is professor-facing when
        a request has failed; a successful submission is not a failure, and a
        later launch error belongs on ``launchError`` (admin-only).
        """
        moment = now or utc_now()
        safe_job_id = validate_slurm_job_id(job_id)
        submitted_at = iso(moment)

        run_patch: dict[str, Any] = {
            "state": "submitted",
            "updatedAt": submitted_at,
            "jobId": safe_job_id,
            "claim": None,
            "error": None,
        }
        self.client.patch(course_training_run_path(course_id, run.run_id), run_patch)

        request_patch: dict[str, Any] = {
            "status": "training",
            "updatedAt": submitted_at,
            "currentRunId": run.run_id,
            "launchError": None,
            "training": {
                "jobId": safe_job_id,
                "mode": run.mode,
                "submittedAt": submitted_at,
                "datasetRef": run.dataset_ref,
                "trainExamples": int(train_count),
                "validationExamples": int(validation_count),
            },
        }
        self.client.patch(course_model_request_path(course_id), request_patch)

        stored = dict(run.raw)
        stored.update(
            {
                "state": "submitted",
                "updatedAt": submitted_at,
                "jobId": safe_job_id,
            }
        )
        stored.pop("claim", None)
        stored.pop("error", None)
        result = parse_run(run.run_id, stored)
        if result is None:  # pragma: no cover - we just built this record
            raise TrainingQueueError("Submitted run could not be read back.")
        return result

    def record_submission_failure(
        self,
        course_id: str,
        run: TrainingRun,
        *,
        error: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Leave the run retryable and keep professor-facing status unchanged.

        The dataset is still valid, so the run goes back to ``queued`` with an
        operator-visible error. The request stays ``preparing``: nothing was
        submitted, and telling a professor it is training (or that it failed)
        would be untrue. ``launchError`` is admin-only. ``failureMessage`` is
        not written — that field is what a professor-facing failed request
        carries.
        """
        moment = now or utc_now()
        self.release(course_id, run, error=error, now=moment)
        self.client.patch(
            course_model_request_path(course_id),
            {
                "launchError": error,
                "updatedAt": iso(moment),
                "currentRunId": run.run_id,
            },
        )


def build_queue(
    *,
    transport: Transport = urllib_transport,
    database_url: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> TrainingQueue:
    return TrainingQueue(
        FirebaseRest(
            database_url=database_url or firebase_database_url(),
            auth_token=auth_token if auth_token is not None else firebase_auth_token(),
            transport=transport,
        )
    )
