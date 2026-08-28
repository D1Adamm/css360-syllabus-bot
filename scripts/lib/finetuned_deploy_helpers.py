#!/usr/bin/env python3
"""Pure helpers for fine-tuned deploy shell scripts.

Used by Tillicum/UWB startup helpers. Keep dependencies to the stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,61}[A-Za-z0-9])?$")
SBATCH_JOB_ID_RE = re.compile(r"(?:Submitted batch job|job)\s+(\d+)", re.IGNORECASE)
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_sbatch_job_id(sbatch_output: str) -> str:
    """Extract a Slurm job ID from ``sbatch`` stdout."""
    for line in sbatch_output.splitlines():
        match = SBATCH_JOB_ID_RE.search(line.strip())
        if match:
            return match.group(1)
    raise ValueError(
        "Could not parse job ID from sbatch output: "
        f"{sbatch_output.strip()!r}"
    )


#: The delimiter every squeue call in this repository now asks for.
#:
#: Whitespace-separated output cannot be parsed positionally, and believing it
#: could be produced visibly wrong status for a pending job. `%N` is *empty* for
#: a job that has no node yet — not "(null)", not a placeholder — so
#: `%i %t %N %M %L` collapses from five fields to four, and every field after
#: the node shifts left by one. A job pending on a QOS limit reported
#: `Node: 0:00`, `Elapsed: 2:00:00` (its time *limit*), and `Time left: unknown`.
#:
#: A delimiter that never appears inside a Slurm field keeps empty fields
#: positional, so an absent node stays an absent node.
SQUEUE_DELIMITER = "|"

#: What the shell helpers ask squeue for: id, state, node, elapsed, left, reason.
SQUEUE_FORMAT = "%i|%t|%N|%M|%L|%R"

SQUEUE_FIELDS = ("job_id", "state", "node", "elapsed", "time_left", "reason")


def _clean_node(value: str) -> str:
    """A node list reduced to one hostname, or empty when there is none.

    A parenthesised value is a *reason*, not a node — squeue puts the reason in
    the NODELIST(REASON) column for pending jobs — so it never becomes one here.
    """
    node = value.strip()
    if not node or node in {"(null)", "N/A"} or node.startswith("("):
        return ""
    return node.split(",")[0]


def parse_squeue_job_line(line: str) -> dict[str, str] | None:
    """Parse one ``squeue -h -o '%i|%t|%N|%M|%L|%R'`` line.

    Returns None for blank lines.

    Whitespace-separated input is still accepted, because older callers and
    hand-run commands produce it, but it is parsed conservatively: only the two
    fields that cannot be empty — the job id and the state — are read
    positionally, and everything after them is left blank rather than guessed
    at. Guessing is what produced a pending job's time limit displayed as its
    elapsed time.
    """
    stripped = line.strip()
    if not stripped:
        return None

    if SQUEUE_DELIMITER in stripped:
        parts = [part.strip() for part in stripped.split(SQUEUE_DELIMITER)]
        parts += [""] * (len(SQUEUE_FIELDS) - len(parts))
        if not parts[0] or not parts[1]:
            raise ValueError(
                f"Unexpected squeue line (need jobid and state): {stripped!r}"
            )
        record = dict(zip(SQUEUE_FIELDS, parts))
        record["node"] = _clean_node(record["node"])
        record["reason"] = record["reason"].strip("()")
        return record

    words = stripped.split()
    if len(words) < 2:
        raise ValueError(f"Unexpected squeue line (need jobid state): {stripped!r}")

    # Only the two unambiguous fields. See the docstring: positions past the
    # state are not trustworthy without a delimiter.
    return {
        "job_id": words[0],
        "state": words[1],
        "node": _clean_node(words[2]) if len(words) >= 3 else "",
        "elapsed": "",
        "time_left": "",
        "reason": "",
    }


def is_active_slurm_state(state: str) -> bool:
    """Return True for PENDING / RUNNING (and common abbreviations)."""
    normalized = state.strip().upper()
    return normalized in {"PD", "PENDING", "R", "RUNNING", "CF", "CONFIGURING"}


def is_running_slurm_state(state: str) -> bool:
    normalized = state.strip().upper()
    return normalized in {"R", "RUNNING"}


#: Slurm state codes an operator sees, in words.
#:
#: Both the abbreviation squeue prints with `%t` and the long form sacct prints,
#: because a job that has left the queue is only visible through sacct and an
#: operator should not have to know which command produced the row they are
#: reading.
SLURM_STATE_LABELS = {
    "PD": "pending",
    "PENDING": "pending",
    "CF": "configuring",
    "CONFIGURING": "configuring",
    "R": "running",
    "RUNNING": "running",
    "CG": "completing",
    "COMPLETING": "completing",
    "CD": "completed",
    "COMPLETED": "completed",
    "F": "failed",
    "FAILED": "failed",
    "CA": "cancelled",
    "CANCELLED": "cancelled",
    "TO": "timed out",
    "TIMEOUT": "timed out",
    "NF": "node failure",
    "NODE_FAIL": "node failure",
    "OOM": "out of memory",
    "PR": "preempted",
    "PREEMPTED": "preempted",
    "S": "suspended",
    "SUSPENDED": "suspended",
}


def describe_slurm_state(state: str) -> str:
    """A state code in words, or the code itself when it is not recognised.

    Unknown codes are returned as-is rather than mapped to "unknown": Slurm has
    more states than are worth enumerating, and showing the real code lets an
    operator look it up.
    """
    raw = (state or "").strip()
    if not raw:
        return "unknown"
    # sacct writes "CANCELLED by 12345"; the first word is the state.
    head = raw.split()[0].upper()
    return SLURM_STATE_LABELS.get(head, raw)


def is_pending_slurm_state(state: str) -> bool:
    normalized = (state or "").strip().upper()
    return normalized in {"PD", "PENDING", "CF", "CONFIGURING"}


def describe_pending_reason(reason: str) -> str:
    """Turn a squeue pending reason into something worth reading.

    The reasons that cost real time here get a sentence; everything else is
    passed through. A job pending on `QOSMaxWallDurationPerJobLimit` will never
    start, and an operator watching it "queue" is watching nothing happen.
    """
    value = (reason or "").strip().strip("()")
    if not value:
        return ""
    explanations = {
        "QOSMaxWallDurationPerJobLimit": (
            "the requested wall clock is longer than this QOS allows — this job "
            "will never start; cancel it and request less time"
        ),
        "PartitionTimeLimit": (
            "the requested wall clock is longer than the partition allows — this "
            "job will never start"
        ),
        "QOSMaxJobsPerUserLimit": "you already have the most jobs this QOS allows",
        "Resources": "waiting for a node with the requested resources",
        "Priority": "waiting behind higher-priority work",
        "ReqNodeNotAvail": (
            "a requested node is unavailable — check any --exclude list and any "
            "reservation or maintenance window"
        ),
        "BeginTime": "held until a scheduled start time",
        "Dependency": "waiting on another job",
    }
    detail = explanations.get(value)
    return f"{value} ({detail})" if detail else value


def validate_compute_hostname(hostname: str) -> str:
    """Validate a compute-node hostname conservatively (no shell metacharacters)."""
    value = hostname.strip()
    if not value:
        raise ValueError("Compute node hostname must not be empty.")
    if len(value) > 63:
        raise ValueError(f"Compute node hostname is too long: {value!r}")
    if not HOSTNAME_RE.fullmatch(value):
        raise ValueError(
            "Invalid compute node hostname. Expected a simple hostname "
            f"like g001 (got {value!r})."
        )
    if ".." in value or value.startswith("-") or value.endswith("-"):
        raise ValueError(f"Invalid compute node hostname: {value!r}")
    return value


def validate_exclude_nodes(nodes: str) -> str:
    """Normalise a comma-separated node list for `sbatch --exclude=`.

    Temporary operator troubleshooting only. A node that fails its GPU preflight
    today is usually repaired within a day or two, and Slurm should be free to
    schedule it again the moment it is — so nothing in this repository ever
    names a node, and a list only exists for as long as the operator types one.

    Each entry goes through the same hostname rule the tunnel helper uses, so an
    exclusion cannot smuggle a shell metacharacter into an sbatch argument.
    Returns the cleaned, de-duplicated, comma-joined list. Empty input returns
    empty, which is the default and means no exclusions at all.
    """
    raw = (nodes or "").strip()
    if not raw:
        return ""

    cleaned: list[str] = []
    for entry in raw.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        validate_compute_hostname(candidate)
        if candidate not in cleaned:
            cleaned.append(candidate)

    if not cleaned:
        return ""
    if len(cleaned) > 16:
        raise ValueError(
            "Refusing to exclude more than 16 nodes. Excluding this much of the "
            "cluster is a scheduling problem to raise with Hyak, not a workaround."
        )
    return ",".join(cleaned)


def parse_wait_for_node_stdout(stdout: str) -> str:
    """Require wait_for_running_node stdout to be exactly one hostname line.

    Progress lines such as ``State: PD (...)`` must go to stderr in the shell
    helper; if they leak into stdout, NODE capture would be contaminated.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(
            "wait_for_running_node stdout must be exactly one hostname line; "
            f"pending/status messages must not appear on stdout (got {lines!r})."
        )
    return validate_compute_hostname(lines[0])


def health_payload_is_ready(payload: Any) -> bool:
    """Return True when /health JSON reports ok + adapterLoaded.

    Accepts either camelCase (remote service / FastAPI aliases) or snake_case.
    """
    if not isinstance(payload, dict):
        return False
    adapter_loaded = payload.get("adapterLoaded")
    if adapter_loaded is None:
        adapter_loaded = payload.get("adapter_loaded")
    return payload.get("status") == "ok" and adapter_loaded is True


def parse_health_json(text: str) -> dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Health response must be a JSON object.")
    return data


def update_env_key(contents: str, key: str, value: str) -> str:
    """Set KEY=value in a .env body, preserving other lines and comments."""
    if not ENV_KEY_RE.fullmatch(key):
        raise ValueError(f"Invalid env key: {key!r}")
    if "\n" in value or "\r" in value:
        raise ValueError("Env value must be a single line.")

    lines = contents.splitlines()
    key_prefix = f"{key}="
    replacement = f"{key}={value}"
    found = False
    updated: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#") or not stripped:
            updated.append(line)
            continue
        if stripped.startswith(key_prefix) or stripped == key:
            if not found:
                updated.append(replacement)
                found = True
            # Drop duplicate KEY= lines
            continue
        updated.append(line)

    if not found:
        if updated and updated[-1] != "":
            updated.append("")
        updated.append(replacement)

    result = "\n".join(updated)
    if contents.endswith("\n") or result:
        result += "\n"
    return result


def update_env_file(path: Path, key: str, value: str) -> bool:
    """Update or append KEY in path. Returns True if the file changed."""
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    updated = update_env_key(original, key, value)
    if updated == original:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return True


def _cli_parse_sbatch(args: argparse.Namespace) -> int:
    print(parse_sbatch_job_id(sys.stdin.read()))
    return 0


def _cli_parse_squeue(args: argparse.Namespace) -> int:
    line = sys.stdin.readline()
    parsed = parse_squeue_job_line(line)
    if parsed is None:
        raise ValueError("Empty squeue line.")
    if args.field:
        print(parsed[args.field])
    else:
        print(json.dumps(parsed))
    return 0


def _cli_validate_hostname(args: argparse.Namespace) -> int:
    print(validate_compute_hostname(args.hostname))
    return 0


def _cli_parse_wait_node_stdout(args: argparse.Namespace) -> int:
    print(parse_wait_for_node_stdout(sys.stdin.read()))
    return 0


def _cli_health_ready(args: argparse.Namespace) -> int:
    payload = parse_health_json(sys.stdin.read())
    if health_payload_is_ready(payload):
        print("ready")
        return 0
    print("not-ready")
    return 1


def _cli_update_env(args: argparse.Namespace) -> int:
    changed = update_env_file(Path(args.path), args.key, args.value)
    print("updated" if changed else "unchanged")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sbatch = sub.add_parser("parse-sbatch-job-id", help="Read sbatch stdout on stdin")
    p_sbatch.set_defaults(func=_cli_parse_sbatch)

    p_squeue = sub.add_parser("parse-squeue-line", help="Parse one squeue line from stdin")
    p_squeue.add_argument(
        "--field",
        choices=SQUEUE_FIELDS,
        default=None,
    )
    p_squeue.set_defaults(func=_cli_parse_squeue)

    p_host = sub.add_parser("validate-hostname", help="Validate a compute hostname")
    p_host.add_argument("hostname")
    p_host.set_defaults(func=_cli_validate_hostname)

    p_wait_node = sub.add_parser(
        "parse-wait-node-stdout",
        help="Require stdin to be exactly one compute hostname line",
    )
    p_wait_node.set_defaults(func=_cli_parse_wait_node_stdout)

    p_fmt = sub.add_parser(
        "squeue-format", help="Print the delimited squeue format string to use"
    )
    p_fmt.set_defaults(func=lambda args: (print(SQUEUE_FORMAT), 0)[1])

    p_state = sub.add_parser("describe-state", help="A Slurm state code in words")
    p_state.add_argument("state")
    p_state.set_defaults(func=lambda args: (print(describe_slurm_state(args.state)), 0)[1])

    p_reason = sub.add_parser(
        "describe-pending-reason", help="Explain a squeue pending reason"
    )
    p_reason.add_argument("reason")
    p_reason.set_defaults(
        func=lambda args: (print(describe_pending_reason(args.reason)), 0)[1]
    )

    p_excl = sub.add_parser(
        "validate-exclude-nodes",
        help="Validate a comma-separated node list for sbatch --exclude",
    )
    p_excl.add_argument("nodes")
    p_excl.set_defaults(
        func=lambda args: (print(validate_exclude_nodes(args.nodes)), 0)[1]
    )

    p_health = sub.add_parser("health-ready", help="Exit 0 if stdin JSON health is ready")
    p_health.set_defaults(func=_cli_health_ready)

    p_env = sub.add_parser("update-env-key", help="Set KEY=value in a .env file")
    p_env.add_argument("path")
    p_env.add_argument("key")
    p_env.add_argument("value")
    p_env.set_defaults(func=_cli_update_env)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, json.JSONDecodeError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
