#!/usr/bin/env python3
"""Tell the application where the fine-tuned service is, and when it ends.

Stdlib only, Python 3.9 compatible: this runs on a Tillicum login node.

Why the cluster reports this
----------------------------
The compute node hostname changes with every Slurm job. Before this, an operator
read it off Tillicum's output and typed it into a command on the UWB VM — a
copy-and-paste between two machines, each behind its own Duo prompt, and the one
step most likely to be got wrong at five to the hour before a class.

Recording the session against the backend turns that into a lookup: the VM asks
where the service is instead of being told. It also gives the Admin page an
answer to "is anything serving right now, and until when?", which no existing
record could provide — a model version's `deployment` describes the artifact,
not a GPU allocation with a wall clock on it.

The session expires on its own. `expiresAt` is the end of the Slurm allocation,
so a dropped login session, a closed laptop or a forgotten stop command all
resolve themselves at exactly the moment the GPU is released.

Where the backend URL and worker token come from
------------------------------------------------
The process environment first, then the repository's `.env.local` / `.env`,
then `backend/.env`. The last of those is what makes `show` work on the UWB VM
without a shell export: the token there lives in `backend/.env`, because that
is the file the backend service itself loads, and `--from-backend` in
`scripts/start_finetuned_tunnel.sh` used to fail with "Missing
TRAINING_WORKER_TOKEN" until an operator sourced it by hand. On Tillicum there
is no `backend/.env`, and `.env.local` is read exactly as before.

Usage
-----
    python3 training/serving_session.py register \\
        --job-id 264787 --node g014 --port 8001 --seconds-remaining 7200
    python3 training/serving_session.py stop --job-id 264787
    python3 training/serving_session.py show
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_LIB = REPO_ROOT / "scripts" / "lib"

#: Exit status when the backend could not be asked at all — configuration
#: missing or the request failing — as opposed to 1, which `show` uses for
#: "asked, and there is no session". The tunnel script tells the two apart.
EXIT_CANNOT_REACH_BACKEND = 2

sys.path.insert(0, str(SCRIPTS_LIB))

from list_published_adapters import published_courses  # noqa: E402
from training_queue import (  # noqa: E402
    TrainingQueueError,
    build_queue,
    load_env_file,
    validate_slurm_job_id,
)

SESSION_STATES = ("starting", "ready", "stopped")


def load_configuration(root: Path = REPO_ROOT) -> None:
    """Populate the environment from the repo's env files, then `backend/.env`.

    `load_env_file` never overwrites a variable that is already set, so the
    order is the precedence: an exported variable wins over every file, the
    repository's `.env.local` / `.env` win over the backend's, and `backend/.env`
    is consulted last. Values are only ever placed in `os.environ`; nothing
    here prints one.

    The permission warning is suppressed for `backend/.env`. Its wording is
    about the cluster copy on a shared filesystem, which training jobs read; the
    VM's file is the backend service's own configuration and is not on GPFS.
    """
    load_env_file(root)
    load_env_file(root / "backend", warn=False)


def session_id_for_job(job_id: str) -> str:
    """`serve-<jobId>`.

    Derived from the Slurm job rather than random so re-running the start script
    against an allocation that is already up refreshes one row instead of
    claiming a second service exists.
    """
    return "serve-{0}".format(validate_slurm_job_id(job_id))


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_detail(
    *,
    serving_root: Optional[str],
    base_model: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """What is being served, for the operator-facing record.

    The course list is read off the serving root rather than asked of the
    service, so it is available before the model has finished loading — which is
    when an operator most wants to know whether the adapter they expected is
    actually published.
    """
    detail: Dict[str, Any] = {}
    if serving_root:
        courses = published_courses(Path(serving_root))
        detail["courses"] = [
            {
                "courseId": course["courseId"],
                "currentVersion": course["currentVersion"],
            }
            for course in courses
        ]
    if base_model:
        detail["baseModel"] = base_model
    if extra:
        detail.update(extra)
    return detail


def register(args: argparse.Namespace) -> int:
    session_id = session_id_for_job(args.job_id)
    now = datetime.now(timezone.utc)
    expires_at = (
        args.expires_at
        if args.expires_at
        else iso(now + timedelta(seconds=max(60, int(args.seconds_remaining or 7200))))
    )

    payload: Dict[str, Any] = {
        "jobId": validate_slurm_job_id(args.job_id),
        "node": args.node,
        "port": int(args.port),
        "state": args.state,
        "startedAt": args.started_at or iso(now),
        "expiresAt": expires_at,
        "detail": build_detail(
            serving_root=args.serving_root,
            base_model=args.base_model,
        ),
    }

    queue = build_queue()
    session = queue.put_serving_session(session_id, payload)
    print("Serving session recorded: {0}".format(session_id))
    if session:
        print("  state:     {0}".format(session.get("state")))
        print("  node:      {0}:{1}".format(session.get("node"), session.get("port")))
        print("  expires:   {0}".format(session.get("expiresAt")))
        courses = ((session.get("detail") or {}).get("courses")) or []
        if courses:
            print(
                "  serving:   {0}".format(
                    ", ".join(
                        "{0} ({1})".format(item["courseId"], item["currentVersion"])
                        for item in courses
                    )
                )
            )
        else:
            print("  serving:   no published course adapters yet")
    return 0


def stop(args: argparse.Namespace) -> int:
    session_id = session_id_for_job(args.job_id)
    queue = build_queue()
    session = queue.stop_serving_session(session_id)
    if session is None:
        # Not an error. A session that already expired on its own is exactly the
        # case where there is nothing left to stop, and reporting that as a
        # failure would make a successful stop look broken.
        print("No recorded session for {0}; nothing to stop.".format(session_id))
        return 0
    print("Serving session stopped: {0}".format(session_id))
    return 0


def show(args: argparse.Namespace) -> int:
    queue = build_queue()
    session = queue.current_serving_session()
    if session is None:
        print("No fine-tuned serving session is currently recorded.")
        return 1
    if args.json:
        print(json.dumps(session, indent=2))
        return 0
    print("Serving session {0}".format(session.get("sessionId")))
    print("  job:     {0}".format(session.get("jobId")))
    print("  node:    {0}:{1}".format(session.get("node"), session.get("port")))
    print("  state:   {0}".format(session.get("state")))
    print("  started: {0}".format(session.get("startedAt")))
    print("  expires: {0}".format(session.get("expiresAt")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serving_session.py",
        description="Record, stop, or show the fine-tuned serving session.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("register", help="Record or refresh the current session")
    p.add_argument("--job-id", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--state", choices=SESSION_STATES, default="ready")
    p.add_argument("--started-at", default=None)
    p.add_argument("--expires-at", default=None)
    p.add_argument(
        "--seconds-remaining",
        type=int,
        default=None,
        help="Wall clock left, used when --expires-at is not given.",
    )
    p.add_argument("--serving-root", default=None)
    p.add_argument("--base-model", default=None)
    p.set_defaults(func=register)

    p = sub.add_parser("stop", help="Mark the session stopped")
    p.add_argument("--job-id", required=True)
    p.set_defaults(func=stop)

    p = sub.add_parser("show", help="Print the session the backend currently has")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=show)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_configuration()
    try:
        return int(args.func(args))
    except TrainingQueueError as exc:
        # The GPU job is the real service; this record is how other machines
        # find it. A backend that is unreachable is worth reporting, and worth
        # not treating as a failure of the thing that actually matters.
        print("Could not reach the application: {0}".format(exc), file=sys.stderr)
        return EXIT_CANNOT_REACH_BACKEND


if __name__ == "__main__":
    raise SystemExit(main())
