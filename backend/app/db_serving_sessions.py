"""PostgreSQL repository for fine-tuned serving sessions.

What a session is, and why it is not a deployment status
--------------------------------------------------------
`course_model_versions.deployment` answers "is this model meant to be served?".
It is a property of one course's artifact and it is durable. A serving session
answers a different question — "is a GPU currently running the inference
service, and until when?" — which belongs to no course in particular: one Slurm
allocation serves every course whose adapter it can load.

Keeping them apart is what lets `status = ready` and `deployment = offline`
remain the honest resting state of a trained model. A model does not stop being
ready because nobody is serving it right now, and a session ending must not
rewrite the training history of every course it happened to serve.

Why sessions expire on their own
--------------------------------
The session's `expires_at` mirrors the Slurm wall clock the job was submitted
with. The allocation ends at that moment whether or not anything reported it,
so a session past its expiry is over as a matter of fact — no heartbeat, no
liveness protocol, and no dependence on the operator's login session surviving.
That is what makes a dropped SSH connection harmless: the record ages out at
exactly the time the GPU does.

`session_id` is derived from the Slurm job id by the start script, so a script
re-run against the same job refreshes one row instead of creating a second.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.db_mapping import as_int, bind_jsonb, optional_string, put_optional, to_iso

SESSION_COLUMNS = """
    session_id, job_id, node, port, state, started_at, expires_at, updated_at,
    detail
"""

SESSION_STATES = ("starting", "ready", "stopped", "expired")

#: States that mean the session might still be serving. `stopped` is reported
#: by the stop script; `expired` is what a read decides for itself.
LIVE_SESSION_STATES = ("starting", "ready")

JSONB_COLUMNS = frozenset({"detail"})


def map_serving_session(
    row: Mapping[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    """One session as the API reports it, with expiry decided at read time.

    A session whose wall clock has passed is reported `expired` regardless of
    what the row says. Nothing writes that state: the allocation ended on the
    cluster, and a reader that trusted the stored value would show a service as
    live for as long as it took someone to notice.
    """
    moment = now or datetime.now(timezone.utc)
    expires_at = row.get("expires_at")
    stored_state = row.get("state") or "starting"

    state = stored_state
    if stored_state in LIVE_SESSION_STATES and isinstance(expires_at, datetime):
        deadline = (
            expires_at
            if expires_at.tzinfo
            else expires_at.replace(tzinfo=timezone.utc)
        )
        if deadline <= moment:
            state = "expired"

    record: dict[str, Any] = {
        "sessionId": row["session_id"],
        "jobId": row["job_id"],
        "node": row["node"],
        "port": as_int(row.get("port")),
        "state": state,
        "storedState": stored_state,
        "startedAt": to_iso(row.get("started_at")),
        "expiresAt": to_iso(row.get("expires_at")),
        "updatedAt": to_iso(row.get("updated_at")),
        "live": state in LIVE_SESSION_STATES,
    }

    detail = row.get("detail")
    if isinstance(detail, dict):
        record["detail"] = detail

    return record


def upsert_serving_session(
    conn: Any,
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Record or refresh one session, keyed by its id.

    An upsert rather than an insert because the start script is safe to re-run:
    submitting nothing new, finding its own job still allocated and reporting the
    same session again must not produce a second row claiming a second service.
    """
    session_id = optional_string(session.get("sessionId"))
    if not session_id:
        raise ValueError("A serving session needs a sessionId.")

    state = optional_string(session.get("state")) or "starting"
    if state not in SESSION_STATES:
        raise ValueError(f"Unknown serving session state: {state!r}")

    now = datetime.now(timezone.utc).isoformat()
    parameters = {
        "session_id": session_id,
        "job_id": session["jobId"],
        "node": session["node"],
        "port": as_int(session.get("port")),
        "state": state,
        "started_at": session.get("startedAt") or now,
        "expires_at": session["expiresAt"],
        "updated_at": now,
        "detail": session.get("detail"),
    }

    columns = list(parameters)
    placeholders = ", ".join(f"%({column})s" for column in columns)
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in columns
        if column != "session_id"
    )

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO serving_sessions ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (session_id) DO UPDATE SET {updates}
            """,
            bind_jsonb(parameters, JSONB_COLUMNS),
        )

    stored = get_serving_session(conn, session_id)
    if stored is None:  # pragma: no cover - defensive
        raise ValueError(f'Serving session "{session_id}" could not be read back.')
    return stored


def get_serving_session(conn: Any, session_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT {SESSION_COLUMNS} FROM serving_sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cursor.fetchone()
    return map_serving_session(row) if row else None


def stop_serving_session(
    conn: Any, session_id: str, *, now: datetime | None = None
) -> dict[str, Any] | None:
    """Mark a session stopped. None when there is no such session.

    Stopping is recorded rather than deleted: an operator asking why the service
    was unavailable at 14:20 needs to see that it was stopped at 14:15, not an
    absence.
    """
    moment = (now or datetime.now(timezone.utc)).isoformat()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE serving_sessions
            SET state = 'stopped', updated_at = %(now)s
            WHERE session_id = %(session_id)s
            """,
            {"now": moment, "session_id": session_id},
        )
        if cursor.rowcount == 0:
            return None
    return get_serving_session(conn, session_id)


def current_serving_session(
    conn: Any, *, now: datetime | None = None
) -> dict[str, Any] | None:
    """The session that is serving right now, or None.

    Newest first among rows that have not been stopped and have not run out of
    wall clock. The expiry is in the SQL as well as in `map_serving_session` so
    that a long-dead row cannot be selected as current in the first place.
    """
    moment = now or datetime.now(timezone.utc)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {SESSION_COLUMNS} FROM serving_sessions
            WHERE state = ANY(%(live_states)s)
              AND expires_at > %(now)s
            ORDER BY started_at DESC
            LIMIT 1
            """,
            {"live_states": list(LIVE_SESSION_STATES), "now": moment},
        )
        row = cursor.fetchone()
    return map_serving_session(row, now=moment) if row else None


def list_serving_sessions(
    conn: Any, *, limit: int = 20, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Recent sessions, newest first. Operator history, not a live view."""
    moment = now or datetime.now(timezone.utc)
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {SESSION_COLUMNS} FROM serving_sessions
            ORDER BY started_at DESC
            LIMIT %s
            """,
            (max(1, int(limit)),),
        )
        rows = cursor.fetchall()
    return [map_serving_session(row, now=moment) for row in rows]


def public_serving_session(session: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The browser-safe view of a session.

    `node` and `port` are dropped. Every `/api/db` route is reachable without a
    credential, and a compute hostname plus a listening port is the one thing in
    this record that describes how to reach a machine rather than what the
    application is doing. The job id stays: `training_runs.jobId` is already
    served on those routes, so removing it here would be inconsistent without
    being protective.
    """
    if session is None:
        return None

    record = {
        "sessionId": session["sessionId"],
        "jobId": session["jobId"],
        "state": session["state"],
        "startedAt": session.get("startedAt"),
        "expiresAt": session.get("expiresAt"),
        "updatedAt": session.get("updatedAt"),
        "live": session.get("live", False),
    }

    detail = session.get("detail")
    if isinstance(detail, dict):
        # Only the part that says what is being served, never how to reach it.
        put_optional(record, "courses", detail.get("courses"))
        put_optional(record, "baseModel", detail.get("baseModel"))
    return record
