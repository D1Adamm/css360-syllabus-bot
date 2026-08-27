"""Durable training queue client for the cluster runner (stdlib only).

Runs live in PostgreSQL on the application VM, in `training_runs`, scoped to a
course. This module is the cluster side of that queue: it discovers queued work
and claims exactly one run with a time-limited lease. It performs no training,
does not know what a scheduler is, and never shells out — everything about
actually submitting a job stays in `training/start_qlora_training.sh`, which
already owns versioned output directories and the live-adapter guard.

Why HTTP and not a database connection
--------------------------------------
Tillicum is not on the database's network, and the only way to give it a
psycopg connection would be to expose PostgreSQL beyond the VM. That is not
something a queue migration should buy. So the worker keeps the shape it always
had — outbound HTTPS to a single host, `urllib`, no client library to install
and pin on a login node we do not control — and the other end is the backend's
`/api/training-queue` router, which owns the transaction.

The credential is a shared worker token sent as `X-Training-Worker-Token`. It
is not a database credential and cannot be used as one: it reaches exactly the
queue endpoints and nothing else.

How two runners are kept apart
------------------------------
By PostgreSQL, in one statement, rather than by anything negotiated here. A
claim is a single POST; the backend selects one eligible row `FOR UPDATE SKIP
LOCKED` and stamps the lease inside the same transaction. A second runner
posting at the same instant cannot see the locked row and does not wait for it
— it either gets the next eligible run or is told there is none. There is no
compare-and-set to retry and no window between choosing and claiming.

The lease still expires, and for the same reason it always did: a runner that
dies mid-run — a dropped login session, a rebooted node — would otherwise
strand a run as permanently claimed. Once `expiresAt` has passed the run
becomes claimable again and `attempt` goes up, so a run that keeps failing this
way is visible as one that keeps being retried rather than one that quietly
vanished.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

API_BASE_URL_ENV_VARS = ("TRAINING_API_BASE_URL", "VITE_API_BASE_URL")
WORKER_TOKEN_ENV_VAR = "TRAINING_WORKER_TOKEN"
WORKER_TOKEN_HEADER = "X-Training-Worker-Token"

QUEUE_PREFIX = "/api/training-queue"


class TrainingQueueError(Exception):
    """Anything that stops the queue being read or written."""


class ClaimConflict(TrainingQueueError):
    """There was no run to take — none queued, or another runner took it."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


#: Permission bits that would let anyone but the owner read the env file.
#:
#: `.env.local` on the cluster holds TRAINING_WORKER_TOKEN, and it lives on a
#: shared project filesystem. It is read by processes on login nodes *and* by
#: training jobs on compute nodes — the token is deliberately not passed through
#: the scheduler, so the file is the only thing standing between the token and
#: anyone else with access to /gpfs/projects/simswe.
GROUP_OR_WORLD_READABLE = 0o077


def warn_if_env_file_is_readable(path: Path) -> Optional[str]:
    """Return a warning when a secrets file is readable beyond its owner.

    Returned rather than printed so callers decide where it goes, and so this
    stays testable. The value itself is never read, quoted, or logged — only the
    mode bits and the path.
    """
    try:
        mode = path.stat().st_mode
    except OSError:
        return None
    if not mode & GROUP_OR_WORLD_READABLE:
        return None
    return (
        "WARNING: {0} is readable by others (mode {1:04o}). It holds "
        "{2}, which training jobs read from this file on the shared "
        "filesystem. Run: chmod 600 {0}".format(
            path, mode & 0o777, WORKER_TOKEN_ENV_VAR
        )
    )


def load_env_file(root: Path = PROJECT_ROOT, *, warn: bool = True) -> None:
    """Read `.env.local` / `.env` the way the other scripts in this repo do.

    This is also how a training job on a compute node obtains
    TRAINING_WORKER_TOKEN: it is scrubbed from the sbatch environment on
    purpose, so it does not travel through the scheduler, and the job reads it
    here instead. That makes the file's permissions the actual protection, which
    is why an over-readable one is called out.
    """
    for name in (".env.local", ".env"):
        path = root / name
        if not path.exists():
            continue
        if warn:
            message = warn_if_env_file_is_readable(path)
            if message:
                print(message, file=sys.stderr)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def training_api_base_url() -> str:
    for name in API_BASE_URL_ENV_VARS:
        url = (os.environ.get(name) or "").strip().rstrip("/")
        if url:
            return url
    raise TrainingQueueError(
        "Missing backend API URL. Set TRAINING_API_BASE_URL in the environment "
        "or .env.local, e.g. TRAINING_API_BASE_URL=https://aiswe.uwb.edu"
    )


def training_worker_token() -> str:
    token = (os.environ.get(WORKER_TOKEN_ENV_VAR) or "").strip()
    if not token:
        raise TrainingQueueError(
            f"Missing {WORKER_TOKEN_ENV_VAR}. The backend refuses queue requests "
            "without it; set the same value the backend has."
        )
    return token


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


MODEL_VERSION_PATTERN = re.compile(r"^v\d+$")


def validate_model_version(version: str) -> str:
    """`v1`, `v2`, … — the scheme the registry allocates.

    Checked here as well as on the backend because this value becomes a URL path
    segment and a filesystem path segment on the cluster.
    """
    candidate = (version or "").strip()
    if not MODEL_VERSION_PATTERN.match(candidate):
        raise TrainingQueueError(f"Invalid model version: {version!r}")
    return candidate


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
    #: The undecoded response body, when the transport kept it.
    #:
    #: Dataset files are UTF-8 by construction, so `body.encode("utf-8")` is
    #: byte-identical for them and the checksum check would pass either way.
    #: Carrying the original bytes anyway means the digest is computed over what
    #: actually arrived rather than over a re-encoding of it, which is the
    #: difference between verifying a transfer and verifying a round trip.
    raw: Optional[bytes] = None

    def content(self) -> bytes:
        return self.raw if self.raw is not None else self.body.encode("utf-8")


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
            payload = response.read()
            return HttpResponse(
                status=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=payload.decode("utf-8", errors="replace"),
                raw=payload,
            )
    except urllib.error.HTTPError as exc:
        # A 4xx is an ordinary outcome for several of these calls — the caller
        # decides what a status means, so it is returned rather than raised.
        return HttpResponse(
            status=exc.code,
            headers={key.lower(): value for key, value in (exc.headers or {}).items()},
            body=exc.read().decode("utf-8", errors="replace"),
        )
    except urllib.error.URLError as exc:
        raise TrainingQueueError(f"Could not reach the backend API: {exc.reason}") from exc


class TrainingQueueApi:
    """The handful of backend calls this queue needs, and nothing else."""

    def __init__(
        self,
        *,
        base_url: str,
        worker_token: str,
        transport: Transport = urllib_transport,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.worker_token = worker_token
        self.transport = transport

    def url(self, path: str, query: Optional[dict[str, str]] = None) -> str:
        url = f"{self.base_url}{QUEUE_PREFIX}{path}"
        if query:
            return f"{url}?{urllib.parse.urlencode(query)}"
        return url

    def _headers(self, *, json_body: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            WORKER_TOKEN_HEADER: self.worker_token,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _decode(self, response: HttpResponse, what: str) -> Any:
        if response.status in (401, 403):
            raise TrainingQueueError(
                f"The backend rejected the request for {what} "
                f"(HTTP {response.status}). Check {WORKER_TOKEN_ENV_VAR} matches "
                "the backend's."
            )
        if response.status >= 400:
            raise TrainingQueueError(
                f"The backend request for {what} failed with HTTP "
                f"{response.status}: {_detail(response.body)}"
            )
        body = response.body.strip()
        if not body or body == "null":
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise TrainingQueueError(
                f"The backend returned a payload for {what} that is not JSON: {exc.msg}"
            ) from exc

    def get(self, path: str, query: Optional[dict[str, str]] = None) -> Any:
        response = self.transport(
            "GET", self.url(path, query), None, self._headers(json_body=False)
        )
        return self._decode(response, path)

    def post(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        body = json.dumps(payload or {}).encode("utf-8")
        response = self.transport(
            "POST", self.url(path), body, self._headers(json_body=True)
        )
        return self._decode(response, path)

    def put(self, path: str, payload: Optional[dict[str, Any]] = None) -> Any:
        body = json.dumps(payload or {}).encode("utf-8")
        response = self.transport(
            "PUT", self.url(path), body, self._headers(json_body=True)
        )
        return self._decode(response, path)

    def get_bytes(self, path: str) -> tuple[bytes, dict[str, str]]:
        """Fetch a file body, with its headers, without JSON-decoding it.

        Used only for dataset files. The status handling is the same as `get`,
        so an expired token or a missing dataset raises the same way it does for
        every other call rather than writing an HTML error page to disk.
        """
        response = self.transport(
            "GET", self.url(path), None, self._headers(json_body=False)
        )
        if response.status in (401, 403):
            raise TrainingQueueError(
                f"The backend rejected the request for {path} "
                f"(HTTP {response.status}). Check {WORKER_TOKEN_ENV_VAR} matches "
                "the backend's."
            )
        if response.status >= 400:
            raise TrainingQueueError(
                f"The backend request for {path} failed with HTTP "
                f"{response.status}: {_detail(response.body)}"
            )
        return response.content(), dict(response.headers)


def _detail(body: str) -> str:
    """The API's `detail` string when there is one, else the raw body."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return (body or "").strip()[:300]
    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        return payload["detail"]
    return (body or "").strip()[:300]


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
    """Read one run record, or None when it is not usable.

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


def parse_run_record(raw: Any) -> Optional[TrainingRun]:
    """Parse a record that carries its own `runId`, as the API returns them."""
    if not isinstance(raw, dict):
        return None
    run_id = raw.get("runId")
    if not isinstance(run_id, str) or not run_id.strip():
        return None
    return parse_run(run_id.strip(), raw)


def parse_runs(payload: Any) -> list[TrainingRun]:
    """A list of API run records, or a mapping of runId -> record."""
    if isinstance(payload, list):
        runs = [parse_run_record(raw) for raw in payload]
    elif isinstance(payload, dict):
        runs = [parse_run(run_id, raw) for run_id, raw in payload.items()]
    else:
        return []
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
    nothing can ever pick up, which is worse than one picked up twice — and the
    second claim still has to win the row lock.
    """
    if not run.claim:
        return True
    expires_at = parse_iso(run.claim.get("expiresAt", ""))
    return expires_at is None or expires_at <= now


def is_claimable(run: TrainingRun, now: datetime) -> bool:
    """The same rule the backend's `CLAIMABLE_PREDICATE` enforces in SQL.

    Kept here so `--dry-run` can explain its own reasoning locally. The backend
    decides; this only has to agree with it.
    """
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
    def __init__(self, client: TrainingQueueApi) -> None:
        self.client = client

    def _run_path(self, course_id: str, run_id: str, action: str) -> str:
        return (
            f"/courses/{validate_course_id(course_id)}"
            f"/runs/{validate_run_id(run_id)}/{action}"
        )

    def list_runs(self, course_id: str) -> list[TrainingRun]:
        """Claimable runs for one course. Read-only."""
        payload = self.client.get(
            "/pending", {"course_id": validate_course_id(course_id)}
        )
        runs = (payload or {}).get("runs") if isinstance(payload, dict) else None
        return parse_runs(runs or [])

    def discover_claimable(
        self,
        *,
        now: datetime,
        course_ids: Optional[list[str]] = None,
    ) -> list[tuple[str, TrainingRun]]:
        """Claimable runs across the courses asked for, oldest first.

        Passing `course_ids` keeps the runner to exactly those courses. One
        request per course when they are named, one request in total when they
        are not — the backend already filters and orders, so nothing here reads
        a course it was not asked about.
        """
        if course_ids is None:
            payload = self.client.get("/pending")
            runs = (payload or {}).get("runs") if isinstance(payload, dict) else None
            found = [(run.course_id, run) for run in parse_runs(runs or [])]
        else:
            found = []
            for course_id in course_ids:
                for run in self.list_runs(course_id):
                    found.append((course_id, run))

        return sorted(
            (pair for pair in found if is_claimable(pair[1], now)),
            key=lambda pair: pair[1].enqueued_at,
        )

    def claim(
        self,
        course_id: str,
        run: TrainingRun,
        *,
        owner: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: Optional[datetime] = None,
    ) -> TrainingRun:
        """Take one run for this course, or raise `ClaimConflict`.

        The run passed in is the candidate this runner saw a moment ago; the
        backend picks and locks the run it actually hands over, which may be a
        different one if something changed in between. Whatever comes back is
        what this runner holds — the caller uses the returned run, not the
        candidate.
        """
        del now  # the backend stamps the lease from its own clock
        safe_course_id = validate_course_id(course_id)
        validate_run_id(run.run_id)

        payload = self.client.post(
            "/claim",
            {
                "owner": owner,
                "leaseSeconds": int(lease_seconds),
                "courseIds": [safe_course_id],
            },
        )

        if not isinstance(payload, dict) or not payload.get("claimed"):
            raise ClaimConflict(
                f"No claimable run for {safe_course_id}; another runner may have "
                "taken it first."
            )

        claimed = parse_run_record(payload.get("run"))
        if claimed is None:
            raise TrainingQueueError("The claimed run could not be read back.")
        return claimed

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
        del now
        self.client.post(
            self._run_path(course_id, run.run_id, "release"), {"error": error}
        )

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
        """Persist a real job id and point the model request at the running job.

        One call, one transaction on the backend. Previously these were two
        ordered writes, and a crash between them left a submitted run whose
        request still said `preparing`; now a professor either sees a job that
        exists or sees the state from before the submission.

        ``failureMessage`` is not written. That field is professor-facing when a
        request has failed; a successful submission is not a failure, and a
        later launch error belongs on ``launchError`` (admin-only).
        """
        del now
        safe_job_id = validate_slurm_job_id(job_id)

        payload = self.client.post(
            self._run_path(course_id, run.run_id, "submitted"),
            {
                "jobId": safe_job_id,
                "trainExamples": int(train_count),
                "validationExamples": int(validation_count),
            },
        )

        submitted = parse_run_record((payload or {}).get("run"))
        if submitted is None:
            raise TrainingQueueError("The submitted run could not be read back.")
        return submitted

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
        would be untrue. ``launchError`` is admin-only.
        """
        del now
        self.client.post(
            self._run_path(course_id, run.run_id, "submission-failed"),
            {"error": error},
        )

    def record_training_failure(
        self,
        course_id: str,
        run: TrainingRun,
        *,
        error: str,
    ) -> TrainingRun:
        """Training itself failed: terminal for the run and for the request.

        Distinct from a submission failure. A job existed and produced no model,
        so there is nothing to retry automatically and the professor does need
        to be told.
        """
        payload = self.client.post(
            self._run_path(course_id, run.run_id, "failed"), {"error": error}
        )
        failed = parse_run_record(payload)
        if failed is None:
            raise TrainingQueueError("The failed run could not be read back.")
        return failed

    # ---------------------------------------------------------------- #
    # Prepared dataset
    #
    # The replacement for the manual rsync. A worker asks the backend for the
    # dataset belonging to a run it holds; it does not name a path, and there
    # is no shape of these calls that reaches another course's data.
    # ---------------------------------------------------------------- #

    def describe_dataset(self, course_id: str, run_id: str) -> dict[str, Any]:
        """Counts, per-file digests, and one digest for the whole dataset.

        Read-only. `--dry-run` calls this to say whether a run is actually
        trainable without claiming it or moving any bytes.
        """
        payload = self.client.get(self._run_path(course_id, run_id, "dataset"))
        if not isinstance(payload, dict):
            raise TrainingQueueError(
                "The backend returned an unreadable dataset description."
            )
        return payload

    def download_dataset_file(
        self, course_id: str, run_id: str, name: str
    ) -> bytes:
        """One prepared file, as bytes. The caller verifies the digest."""
        path = f"{self._run_path(course_id, run_id, 'dataset')}/files/{name}"
        body, _headers = self.client.get_bytes(path)
        return body

    # ---------------------------------------------------------------- #
    # Completion
    # ---------------------------------------------------------------- #

    def record_completion(
        self, course_id: str, run_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Report how a job ended, and let the backend register the model.

        Safe to call more than once with the same payload. The backend keys
        registration on the run, so a redelivery after a network failure
        refreshes the version the first delivery created rather than creating
        another one — which is what makes the worker's "persist, then send,
        then retry later" behaviour safe.
        """
        result = self.client.post(
            self._run_path(course_id, run_id, "completed"), payload
        )
        if not isinstance(result, dict):
            raise TrainingQueueError("The completion result could not be read back.")
        return result

    # ---------------------------------------------------------------- #
    # Serving sessions
    # ---------------------------------------------------------------- #

    def put_serving_session(
        self, session_id: str, payload: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        result = self.client.put(f"/serving-sessions/{session_id}", payload)
        return (result or {}).get("session") if isinstance(result, dict) else None

    def stop_serving_session(self, session_id: str) -> Optional[dict[str, Any]]:
        result = self.client.post(f"/serving-sessions/{session_id}/stopped")
        return (result or {}).get("session") if isinstance(result, dict) else None

    def current_serving_session(self) -> Optional[dict[str, Any]]:
        result = self.client.get("/serving-session")
        return (result or {}).get("session") if isinstance(result, dict) else None

    def report_model_published(
        self,
        course_id: str,
        version: str,
        *,
        source_ref: Optional[str] = None,
        published_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """Tell the application a version's adapter is now in the serving tree.

        Called only after the copy has landed and been validated. That ordering
        is the failure safety: a copy that fails reports nothing, so the
        application keeps naming the version that really is published rather
        than routing every question at one that is not there.

        Idempotent on the backend, so a rerun of the promote script or a report
        redelivered after a network failure lands in the same state.
        """
        payload: dict[str, Any] = {}
        if source_ref:
            payload["sourceRef"] = source_ref
        if published_at:
            payload["publishedAt"] = published_at

        result = self.client.post(
            "/courses/{0}/model-versions/{1}/published".format(
                validate_course_id(course_id), validate_model_version(version)
            ),
            payload,
        )
        if not isinstance(result, dict):
            raise TrainingQueueError(
                "The published model version could not be read back."
            )
        return result

    def register_model_version(
        self,
        course_id: str,
        *,
        base_model: str,
        training_example_count: int,
        artifact_ref: str,
        status: str = "ready",
        deployment: str = "offline",
        version: Optional[str] = None,
        notes: Optional[str] = None,
        run_id: Optional[str] = None,
        set_current: bool = True,
    ) -> dict[str, Any]:
        """Record a finished model against this course.

        Course isolation is the backend's, not this caller's: the version row is
        keyed by course id, so there is no shape of this request that writes
        another course's registry.
        """
        payload: dict[str, Any] = {
            "baseModel": base_model,
            "trainingExampleCount": int(training_example_count),
            "artifactRef": artifact_ref,
            "status": status,
            "deployment": deployment,
            "setCurrent": bool(set_current),
        }
        if version:
            payload["version"] = version
        if notes:
            payload["notes"] = notes
        if run_id:
            payload["runId"] = validate_run_id(run_id)

        result = self.client.post(
            f"/courses/{validate_course_id(course_id)}/model-versions", payload
        )
        if not isinstance(result, dict):
            raise TrainingQueueError("The registered model version could not be read back.")
        return result


def build_queue(
    *,
    transport: Transport = urllib_transport,
    base_url: Optional[str] = None,
    worker_token: Optional[str] = None,
) -> TrainingQueue:
    return TrainingQueue(
        TrainingQueueApi(
            base_url=base_url or training_api_base_url(),
            worker_token=(
                worker_token if worker_token is not None else training_worker_token()
            ),
            transport=transport,
        )
    )
