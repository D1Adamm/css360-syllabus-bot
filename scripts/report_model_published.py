#!/usr/bin/env python3
"""Tell the application that a course model version has been published.

Stdlib only. Runs on Tillicum, after `promote_qlora_adapter.sh` has copied an
adapter into the serving tree and validated it.

Why this is a separate step rather than part of registration
------------------------------------------------------------
Registering a model and publishing it are different facts, and conflating them
caused a real outage. A finished training run registers `v2` and moves the
course's `current_version` to it, because the newest model is the honest answer
to a professor asking "what is my model". But `v2` is not on the cluster yet —
publishing is a deliberate act — so an application that routed questions by
`current_version` started asking for an adapter that did not exist, and a course
that `v1` had been answering perfectly began failing every fine-tuned request.

So inference resolves the *published* version, and this is what tells the
application which one that is.

Ordering, and why it is this way round
--------------------------------------
The adapter is copied and validated first. Only then is this called. A copy that
fails reports nothing, so the application goes on naming the version that really
is published — degraded (the new one is not live yet) rather than wrong (every
question routed at an adapter that is not there).

If the report itself fails — the backend is down, the tunnel is closed, the
login session dropped — the publication is real and the application does not
know it. The payload is persisted under `training/state/pending/` and delivered
by the next `./training/run_training_queue.sh --once`, the same mechanism a
finished training job uses. Nothing is lost and nothing has to be remembered.

Refusals are final, not outages
-------------------------------
The backend holds this report to the same check as Admin → Models → Activate:
it asks the fine-tuned service students' questions reach whether it can serve
this course and version, and refuses with 409 when it cannot (or 404 when the
version is not registered). That is a decision, not a network failure, and
replaying it on every queue run would show the same refusal forever while an
operator believed a report was merely pending. So a 404/409 clears the pending
entry, is printed in full, and exits 3 — the adapter is in Tillicum's serving
tree, but inference keeps the previously published version until the version is
installed and mapped on the VM and then activated or re-reported.

Usage:
    python3 scripts/report_model_published.py \\
        --course-id css-350-spring-2026-n3h9 --version v1
    python3 scripts/report_model_published.py ... --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lib"))

from training_queue import (  # noqa: E402  (path set above)
    TrainingQueue,
    TrainingQueueError,
    build_queue,
    load_env_file,
    validate_course_id,
    validate_model_version,
)
from training_state import (  # noqa: E402
    clear_pending_callback,
    queue_pending_callback,
)

#: The synthetic run id a publication is filed under.
#:
#: The pending-callback store is keyed by run id because everything in it used
#: to describe a training run. A publication describes a course and a version
#: instead, so it gets a key built from those — stable, so re-publishing the
#: same version overwrites its own pending entry rather than queueing a second.
PENDING_KIND = "published"

#: Exit status when the backend refused the publication (404/409). Distinct
#: from 0 ("delivered, or queued for redelivery") so the promote script can
#: tell an operator the version is NOT serving rather than "the report is queued".
REFUSED_EXIT = 3

#: Backend answers that mean "decided, do not replay". The queue client puts the
#: status in the message (`training_queue.TrainingQueue._decode`), the same
#: text the completion reporter matches on.
FINAL_REFUSAL_MARKERS = ("HTTP 404", "HTTP 409")


def is_final_refusal(message: str) -> bool:
    return any(marker in message for marker in FINAL_REFUSAL_MARKERS)


def pending_key(course_id: str, version: str) -> str:
    return "publish-{0}-{1}".format(
        validate_course_id(course_id), validate_model_version(version)
    )


def build_payload(
    *,
    course_id: str,
    version: str,
    source_ref: Optional[str] = None,
    published_at: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "courseId": validate_course_id(course_id),
        "version": validate_model_version(version),
        "publishedAt": published_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if source_ref:
        payload["sourceRef"] = source_ref
    return payload


def send_publication(
    queue: TrainingQueue,
    payload: dict[str, Any],
    *,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Deliver one publication report, clearing its pending entry once decided.

    Returns `{"delivered": True, "result": ...}` when the backend accepted it,
    or `{"delivered": False, "refused": True, "detail": ...}` when it refused
    with 404/409 — final either way, so the pending entry is cleared in both
    cases. Anything else (unreachable, 5xx, a bad token) raises and leaves the
    entry queued, which is what the next worker run exists to retry.
    """
    course_id = str(payload["courseId"])
    version = str(payload["version"])

    try:
        result = queue.report_model_published(
            course_id,
            version,
            source_ref=payload.get("sourceRef"),
            published_at=payload.get("publishedAt"),
        )
    except TrainingQueueError as exc:
        message = str(exc)
        if is_final_refusal(message):
            clear_pending_callback(
                repo_root, pending_key(course_id, version), PENDING_KIND
            )
            return {"delivered": False, "refused": True, "detail": message}
        raise
    clear_pending_callback(repo_root, pending_key(course_id, version), PENDING_KIND)
    return {"delivered": True, "result": result}


def report_publication(
    *,
    course_id: str,
    version: str,
    source_ref: Optional[str] = None,
    repo_root: Path = PROJECT_ROOT,
    queue: Optional[TrainingQueue] = None,
) -> dict[str, Any]:
    """Persist first, then send. Never raises for an unreachable backend.

    Persisting before the attempt rather than after a failure is the same rule
    the completion reporter follows: a process that dies between "decided to
    report" and "failed to report" is indistinguishable from one that never
    tried, and only the file on disk makes the difference recoverable.
    """
    payload = build_payload(
        course_id=course_id, version=version, source_ref=source_ref
    )

    queue_pending_callback(
        repo_root,
        run_id=pending_key(course_id, version),
        course_id=payload["courseId"],
        kind=PENDING_KIND,
        payload=payload,
    )

    try:
        client = queue if queue is not None else build_queue()
        sent = send_publication(client, payload, repo_root=repo_root)
    except TrainingQueueError as exc:
        return {"delivered": False, "error": str(exc), "payload": payload}

    if sent.get("refused"):
        return {
            "delivered": False,
            "refused": True,
            "error": str(sent.get("detail") or ""),
            "payload": payload,
        }
    return {"delivered": True, "result": sent.get("result"), "payload": payload}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--source-ref",
        default=None,
        help="Relative reference to the adapter this was published from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report that would be sent and exit.",
    )
    args = parser.parse_args(argv)

    load_env_file(PROJECT_ROOT)

    try:
        payload = build_payload(
            course_id=args.course_id,
            version=args.version,
            source_ref=args.source_ref,
        )
    except TrainingQueueError as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        print("Would report {courseId} {version} as published.".format(**payload))
        return 0

    outcome = report_publication(
        course_id=args.course_id,
        version=args.version,
        source_ref=args.source_ref,
    )

    if outcome.get("refused"):
        # The backend decided, so nothing is queued and nothing will change on
        # its own. Say exactly that: the one message an operator must not read
        # here is "the report is queued".
        print(
            "REFUSED: the application did not switch {courseId} to {version}.".format(
                **payload
            ),
            file=sys.stderr,
        )
        print("  {0}".format(outcome["error"]), file=sys.stderr)
        print(
            "The adapter is in the serving tree, but the fine-tuned service the "
            "backend uses cannot serve this course and version, so inference "
            "keeps the previously published version. Install and map it on the VM "
            "(scripts/install_finetuned_adapter.py, then scripts/aiswe_finetuned.sh "
            "set-mapping and restart), then activate it from Admin → Models or "
            "re-run this report. Nothing was queued for redelivery.",
            file=sys.stderr,
        )
        return REFUSED_EXIT

    if not outcome["delivered"]:
        # Exit 0 deliberately. The adapter really is published; only the report
        # is outstanding, and it is queued. Failing here would make an operator
        # think the publication itself had failed and re-run it.
        print(
            "Published, but the application was not told: {0}".format(
                outcome["error"]
            )
        )
        print(
            "Queued for delivery. It will be sent by the next "
            "./training/run_training_queue.sh --once"
        )
        return 0

    result = outcome["result"] or {}
    if result.get("unchanged"):
        print(
            "{courseId} {version} was already published; nothing changed.".format(
                **payload
            )
        )
    else:
        previous = result.get("previousVersion")
        print(
            "{0} is now serving {1}{2}.".format(
                payload["courseId"],
                payload["version"],
                " (was {0})".format(previous) if previous else "",
            )
        )
    if result.get("currentVersion") and result["currentVersion"] != payload["version"]:
        print(
            "Newest registered version is {0}; it is not published yet.".format(
                result["currentVersion"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
