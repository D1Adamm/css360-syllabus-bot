#!/usr/bin/env python3
"""Pure helpers for operating the VM-local fine-tuned service (`aiswe-finetuned`).

Used by `scripts/aiswe_finetuned.sh` and `scripts/install_finetuned_adapter.py`.
Standard library only, plus `training/inference_service/helpers.py` (also
standard library only) for the one rule that must not be restated: what a
valid `FINETUNED_OLLAMA_MODELS` mapping is. The service and every tool that
edits its mapping parse it with the same function.

Three concerns live here:

- **Who owns port 9001.** The local service and the Tillicum SSH tunnel both
  listen on `127.0.0.1:9001` and must never run at once. `ss -ltnp` output is
  parsed into a listener list and classified as the tunnel (an `ssh` process),
  the local service (a `python` process), or something else. Nothing here kills
  anything; the callers refuse and say what they found.
- **The environment file.** The unit reads its mutable settings from a
  restricted `EnvironmentFile` rather than from lines in the unit. Reading and
  updating that file is done here so an operator never edits a mapping by hand.
- **Names.** The Ollama tag convention (`css360-ft-v2`) and the mapping entry
  a newly installed adapter should be added under.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_HELPERS_PATH = REPO_ROOT / "training" / "inference_service" / "helpers.py"

DEFAULT_PORT = 9001
UNIT_NAME = "aiswe-finetuned"
KEEP_ALIVE_ENV = "FINETUNED_KEEP_ALIVE"
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: `ss -ltnp` prints `users:(("ssh",pid=1234,fd=5))`; several processes can
#: share one socket, and the first is the one that matters.
SS_USERS_RE = re.compile(r'users:\(\("(?P<name>[^"]+)",pid=(?P<pid>\d+)')
COURSE_SHORT_RE = re.compile(r"^([a-z]+)-([0-9]+)(?:-|$)")


def _load_service_helpers():
    spec = importlib.util.spec_from_file_location(
        "finetuned_service_helpers", SERVICE_HELPERS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SERVICE_HELPERS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


service_helpers = _load_service_helpers()
MODEL_MAP_ENV = service_helpers.MODEL_MAP_ENV
CourseAdapterError = service_helpers.CourseAdapterError
parse_model_map = service_helpers.parse_model_map
normalize_ollama_model_name = service_helpers.normalize_ollama_model_name
validate_course_id = service_helpers.validate_course_id
validate_model_version = service_helpers.validate_model_version


# --------------------------------------------------------------------------- #
# Port ownership
# --------------------------------------------------------------------------- #


def parse_ss_listeners(text: str) -> list[dict[str, Any]]:
    """Listeners from `ss -H -ltnp` (or with the header; it is skipped).

    Each row is `State Recv-Q Send-Q Local:Port Peer:Port Process`. Only the
    local address and the first owning process are kept. A row whose port
    cannot be read is skipped rather than guessed at.
    """
    listeners: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("State") or stripped.startswith("Netid"):
            continue
        parts = stripped.split()
        local = None
        for part in parts:
            if ":" in part and part.rsplit(":", 1)[1].isdigit():
                local = part
                break
        if local is None:
            continue
        host, _, port_text = local.rpartition(":")
        match = SS_USERS_RE.search(stripped)
        listeners.append(
            {
                "host": host,
                "port": int(port_text),
                "process": match.group("name") if match else None,
                "pid": int(match.group("pid")) if match else None,
            }
        )
    return listeners


def classify_port_owner(listeners: list[dict[str, Any]], port: int = DEFAULT_PORT) -> dict[str, Any]:
    """What is listening on `port`, in the terms the operator scripts act on.

    `kind` is one of:

    - `free`    nothing listens there
    - `tunnel`  an `ssh` process: the Tillicum fallback forward
    - `service` a `python` process: the local service (or another Python
                listener, which the caller checks with `/health`)
    - `other`   something else, or a listener whose process could not be read
                (typically another user's process, invisible without root)
    """
    for listener in listeners:
        if listener["port"] != port:
            continue
        name = (listener.get("process") or "").lower()
        if name.startswith("ssh"):
            kind = "tunnel"
        elif name.startswith("python"):
            kind = "service"
        else:
            kind = "other"
        return {
            "kind": kind,
            "port": port,
            "host": listener.get("host"),
            "process": listener.get("process"),
            "pid": listener.get("pid"),
        }
    return {"kind": "free", "port": port, "host": None, "process": None, "pid": None}


# --------------------------------------------------------------------------- #
# The environment file
# --------------------------------------------------------------------------- #


def parse_env_file(text: str) -> dict[str, str]:
    """`KEY=value` lines as systemd's `EnvironmentFile` reads them.

    Comments and blank lines are skipped; a value may be wrapped in single or
    double quotes; a later duplicate key wins, as it does for systemd. No
    shell expansion of any kind happens — `$HOME` is four characters.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY_RE.fullmatch(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def upsert_env_line(text: str, key: str, value: str) -> str:
    """Set `KEY=value`, replacing every existing assignment of the key.

    Comments, blank lines and other keys are kept where they are. A value
    that could break the file — a newline, or a quote character — is refused,
    because the mapping grammar needs neither and a malformed file would stop
    the service at its next start.
    """
    if not ENV_KEY_RE.fullmatch(key):
        raise ValueError(f"Invalid environment key: {key!r}")
    if any(char in value for char in "\n\r\"'"):
        raise ValueError("Environment values must be one line with no quote characters.")
    replacement = f"{key}={value}"
    lines = text.splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        stripped = line.strip()
        candidate = stripped[len("export "):].lstrip() if stripped.startswith("export ") else stripped
        if not stripped.startswith("#") and candidate.split("=", 1)[0].strip() == key and "=" in candidate:
            if not found:
                out.append(replacement)
                found = True
            continue
        out.append(line)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append(replacement)
    return "\n".join(out) + "\n"


def mapping_entries(raw: str | None) -> list[tuple[str, str, str]]:
    """The mapping as `(courseId, version, ollamaModel)` rows, validated."""
    parsed = parse_model_map(raw)
    rows: list[tuple[str, str, str]] = []
    for course_id in sorted(parsed):
        versions = parsed[course_id]
        for version in sorted(versions, key=lambda name: int(name[1:])):
            rows.append((course_id, version, versions[version]))
    return rows


def format_mapping(rows: list[tuple[str, str, str]]) -> str:
    return ",".join(f"{course}@{version}={model}" for course, version, model in rows)


def format_mapping_entry(course_id: str, version: str, ollama_model: str) -> str:
    return format_mapping(
        [
            (
                validate_course_id(course_id),
                validate_model_version(version),
                normalize_ollama_model_name(ollama_model),
            )
        ]
    )


def merge_mapping(
    raw: str | None,
    course_id: str,
    version: str,
    ollama_model: str,
    *,
    replace: bool = False,
) -> str:
    """The mapping with one entry added.

    An existing entry for the same course and version that names the same
    model is a no-op. One that names a *different* model is refused unless
    `replace` is set: a version is meant to be built once, and changing what
    it points at silently is how a served model gets swapped without anyone
    deciding to. The whole result is re-parsed before it is returned, so a
    mapping this function produces is one the service will accept.
    """
    safe_course = validate_course_id(course_id)
    safe_version = validate_model_version(version)
    safe_model = normalize_ollama_model_name(ollama_model)
    rows = mapping_entries(raw)
    out: list[tuple[str, str, str]] = []
    seen = False
    for course, ver, model in rows:
        if course == safe_course and ver == safe_version:
            seen = True
            if model != safe_model and not replace:
                raise CourseAdapterError(
                    f"{MODEL_MAP_ENV} already maps {safe_course} {safe_version} to "
                    f"{model}; pass --replace to point it at {safe_model} instead."
                )
            out.append((course, ver, safe_model))
        else:
            out.append((course, ver, model))
    if not seen:
        out.append((safe_course, safe_version, safe_model))
    merged = format_mapping(sorted(out, key=lambda row: (row[0], int(row[1][1:]))))
    parse_model_map(merged)
    return merged


def validate_env_text(text: str) -> dict[str, Any]:
    """Summarise an environment file: the mapping rows, keep-alive, and errors.

    Never raises on a bad mapping; the error is part of the summary so a
    `check` can print it as a FAIL line beside the other checks.
    """
    values = parse_env_file(text)
    summary: dict[str, Any] = {
        "mappingRaw": values.get(MODEL_MAP_ENV, ""),
        "keepAlive": values.get(KEEP_ALIVE_ENV) or None,
        "entries": [],
        "error": None,
    }
    try:
        summary["entries"] = [
            {"courseId": course, "version": version, "ollamaModel": model}
            for course, version, model in mapping_entries(summary["mappingRaw"])
        ]
    except CourseAdapterError as exc:
        summary["error"] = str(exc)
    return summary


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def short_course_label(course_id: str) -> str:
    """`css-360-winter-2026-a7rp` → `css360`, the prefix every served tag uses."""
    safe = validate_course_id(course_id)
    match = COURSE_SHORT_RE.match(safe)
    if not match:
        raise CourseAdapterError(
            f"Cannot derive a short label from course id {safe!r}; pass --tag explicitly."
        )
    return match.group(1) + match.group(2)


def default_ollama_tag(course_id: str, version: str) -> str:
    """`css360-ft-v2`: one Ollama model per course and version, by convention."""
    return f"{short_course_label(course_id)}-ft-{validate_model_version(version)}"


def gguf_filename(course_id: str, version: str) -> str:
    return f"{short_course_label(course_id)}-{validate_model_version(version)}-lora.gguf"


# --------------------------------------------------------------------------- #
# CLI, for the shell script
# --------------------------------------------------------------------------- #


def _cli_port_owner(args: argparse.Namespace) -> int:
    owner = classify_port_owner(parse_ss_listeners(sys.stdin.read()), args.port)
    if args.field:
        print(owner.get(args.field) if owner.get(args.field) is not None else "")
    else:
        print(json.dumps(owner))
    return 0


def _cli_validate_env(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"Environment file not found: {path}", file=sys.stderr)
        return 2
    summary = validate_env_text(path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(summary))
    else:
        for entry in summary["entries"]:
            print(f"{entry['courseId']} {entry['version']} -> {entry['ollamaModel']}")
        if summary["error"]:
            print(summary["error"], file=sys.stderr)
    return 1 if summary["error"] else 0


def _cli_set_mapping(args: argparse.Namespace) -> int:
    path = Path(args.path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    current = parse_env_file(text).get(MODEL_MAP_ENV, "")
    merged = merge_mapping(current, args.course, args.version, args.model, replace=args.replace)
    if merged == current:
        print("unchanged")
        return 0
    if args.dry_run:
        print(f"would set {MODEL_MAP_ENV}={merged}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(upsert_env_line(text, MODEL_MAP_ENV, merged), encoding="utf-8")
    print(f"{MODEL_MAP_ENV}={merged}")
    return 0


def _cli_mapping_entries(args: argparse.Namespace) -> int:
    path = Path(args.path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    raw = parse_env_file(text).get(MODEL_MAP_ENV, "")
    for course, version, model in mapping_entries(raw):
        print(f"{course} {version} {model}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_owner = sub.add_parser("port-owner", help="Classify the listener on a port from `ss -H -ltnp` on stdin")
    p_owner.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_owner.add_argument("--field", choices=("kind", "process", "pid"), default=None)
    p_owner.set_defaults(func=_cli_port_owner)

    p_env = sub.add_parser("validate-env", help="Parse and validate the unit's environment file")
    p_env.add_argument("path")
    p_env.add_argument("--json", action="store_true")
    p_env.set_defaults(func=_cli_validate_env)

    p_set = sub.add_parser("set-mapping", help="Add or update one course@version=model entry")
    p_set.add_argument("path")
    p_set.add_argument("course")
    p_set.add_argument("version")
    p_set.add_argument("model")
    p_set.add_argument("--replace", action="store_true")
    p_set.add_argument("--dry-run", action="store_true")
    p_set.set_defaults(func=_cli_set_mapping)

    p_rows = sub.add_parser("mapping-entries", help="Print `course version model` rows from the env file")
    p_rows.add_argument("path")
    p_rows.set_defaults(func=_cli_mapping_entries)

    p_tag = sub.add_parser("default-tag", help="The conventional Ollama tag for a course and version")
    p_tag.add_argument("course")
    p_tag.add_argument("version")
    p_tag.set_defaults(func=lambda args: (print(default_ollama_tag(args.course, args.version)), 0)[1])

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (CourseAdapterError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
