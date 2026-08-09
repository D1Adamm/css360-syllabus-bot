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


def parse_squeue_job_line(line: str) -> dict[str, str] | None:
    """Parse one ``squeue -h -o '%i %t %N %M %L'`` style line.

    Returns None for blank lines. Fields after the first two may be empty when
    the job is still pending (no node yet).
    """
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) < 2:
        raise ValueError(f"Unexpected squeue line (need jobid state): {stripped!r}")
    job_id, state = parts[0], parts[1]
    node = parts[2] if len(parts) >= 3 and parts[2] not in {"", "(null)"} else ""
    elapsed = parts[3] if len(parts) >= 4 else ""
    time_left = parts[4] if len(parts) >= 5 else ""
    # Pending jobs sometimes show node as "(null)" already handled; also
    # collapse multi-token node lists to the first hostname token.
    if node.startswith("("):
        node = ""
    return {
        "job_id": job_id,
        "state": state,
        "node": node.split(",")[0] if node else "",
        "elapsed": elapsed,
        "time_left": time_left,
    }


def is_active_slurm_state(state: str) -> bool:
    """Return True for PENDING / RUNNING (and common abbreviations)."""
    normalized = state.strip().upper()
    return normalized in {"PD", "PENDING", "R", "RUNNING", "CF", "CONFIGURING"}


def is_running_slurm_state(state: str) -> bool:
    normalized = state.strip().upper()
    return normalized in {"R", "RUNNING"}


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
        choices=("job_id", "state", "node", "elapsed", "time_left"),
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
