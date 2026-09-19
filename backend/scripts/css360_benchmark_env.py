#!/usr/bin/env python3
"""Turn the CSS 360 benchmark route on or off in backend/.env, safely and repeatably.

Run on the application VM, from the backend directory:

    cd ~/css360-syllabus-bot/backend
    .venv/bin/python scripts/css360_benchmark_env.py --enable
    .venv/bin/python scripts/css360_benchmark_env.py --disable
    .venv/bin/python scripts/css360_benchmark_env.py --show

Each managed key is set exactly once, whatever the file held before: a key
that is present is replaced in place, a key that is missing is appended, and
duplicates left by an earlier `>>` are collapsed to one. Run it twice and the
file is the same as after once. Every other line and comment is untouched.

The token is generated here, with `secrets.token_hex(32)`, the first time the
route is enabled and kept on every later run; `--rotate-token` replaces it and
`--token-from-stdin` supplies one of at least 32 characters. The token is never
printed and never passed on a command line. `--disable` changes the flag and
nothing else, so re-enabling reuses the token. The file is written atomically
and created with mode 0600 when it did not exist.

Nothing here touches PostgreSQL, the model registry, or the production
fine-tuned service. Restart `aiswe-backend` afterwards for the change to take
effect; while the flag is off the route answers 404.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import secrets
import stat
import sys
import tempfile
from pathlib import Path
from typing import Callable

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_ENV_FILE = BACKEND_ROOT / ".env"
HELPERS_PATH = REPO_ROOT / "scripts" / "lib" / "finetuned_deploy_helpers.py"

ENABLED_KEY = "CSS360_BENCHMARK_ENABLED"
TOKEN_KEY = "CSS360_BENCHMARK_TOKEN"
SERVICE_URL_KEY = "CSS360_BENCHMARK_SERVICE_URL"
MANAGED_KEYS = (ENABLED_KEY, TOKEN_KEY, SERVICE_URL_KEY)

DEFAULT_SERVICE_URL = "http://127.0.0.1:9002"
#: The route refuses to enable itself on a shorter token (`MIN_TOKEN_LENGTH`
#: in app/research_benchmark_routes.py), so a shorter one is refused here too.
MIN_TOKEN_LENGTH = 32
TOKEN_BYTES = 32


def _load_update_env_key() -> Callable[[str, str, str], str]:
    """The repository's one .env editor, shared with the tunnel helper."""
    spec = importlib.util.spec_from_file_location("finetuned_deploy_helpers", HELPERS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.update_env_key


def current_values(contents: str) -> dict[str, str]:
    """The managed keys as a loader would see them: the last occurrence wins."""
    values: dict[str, str] = {}
    for raw in contents.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if key not in MANAGED_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def generate_token() -> str:
    return secrets.token_hex(TOKEN_BYTES)


def validate_token(token: str) -> str:
    value = token.strip()
    if len(value) < MIN_TOKEN_LENGTH or any(ch.isspace() for ch in value):
        raise ValueError(
            f"The token must be at least {MIN_TOKEN_LENGTH} characters with no whitespace."
        )
    return value


def validate_service_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValueError("The service URL must start with http:// or https://.")
    return value


def apply(
    contents: str,
    *,
    enabled: bool,
    service_url: str | None = None,
    token: str | None = None,
    rotate_token: bool = False,
) -> str:
    """The new file body. Idempotent: applying the result again changes nothing."""
    update_env_key = _load_update_env_key()
    existing = current_values(contents)
    updated = update_env_key(contents, ENABLED_KEY, "true" if enabled else "false")
    if not enabled and service_url is None and token is None:
        return updated

    url = validate_service_url(service_url or existing.get(SERVICE_URL_KEY) or DEFAULT_SERVICE_URL)
    updated = update_env_key(updated, SERVICE_URL_KEY, url)

    if token is not None:
        chosen = validate_token(token)
    elif rotate_token or len((existing.get(TOKEN_KEY) or "").strip()) < MIN_TOKEN_LENGTH:
        chosen = generate_token()
    else:
        chosen = existing[TOKEN_KEY].strip()
    return update_env_key(updated, TOKEN_KEY, chosen)


def write_atomic(path: Path, contents: str) -> None:
    """Replace the file in place; new files are private to the owner."""
    existed = path.exists()
    mode = stat.S_IMODE(path.stat().st_mode) if existed else 0o600
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(contents)
        os.chmod(handle.name, mode)
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def describe(contents: str, path: Path) -> str:
    values = current_values(contents)
    token = values.get(TOKEN_KEY, "")
    enabled = values.get(ENABLED_KEY, "").strip().lower() in {"1", "true", "yes", "on"}
    token_state = f"set ({len(token)} characters)" if token else "unset"
    if token and len(token) < MIN_TOKEN_LENGTH:
        token_state += ", too short: the route will stay off"
    url = values.get(SERVICE_URL_KEY) or "unset"
    lines = [
        f"file: {path}",
        f"{ENABLED_KEY}: {'true' if enabled else 'false'}",
        f"{SERVICE_URL_KEY}: {url}",
        f"{TOKEN_KEY}: {token_state}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE,
                        help="the .env file to edit (default: backend/.env)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--enable", action="store_true", help="turn the route on")
    action.add_argument("--disable", action="store_true", help="turn the route off; keeps the token")
    action.add_argument("--show", action="store_true", help="print the current state and change nothing")
    parser.add_argument("--service-url", help=f"the benchmark service URL (default {DEFAULT_SERVICE_URL})")
    parser.add_argument("--rotate-token", action="store_true", help="with --enable: generate a new token")
    parser.add_argument("--token-from-stdin", action="store_true",
                        help="with --enable: read the token from standard input")
    args = parser.parse_args(argv)

    path: Path = args.env_file
    contents = path.read_text(encoding="utf-8") if path.exists() else ""

    if args.show:
        print(describe(contents, path))
        return 0

    token: str | None = None
    if args.token_from_stdin:
        if not args.enable:
            parser.error("--token-from-stdin requires --enable")
        token = sys.stdin.readline().rstrip("\r\n")
    if args.rotate_token and not args.enable:
        parser.error("--rotate-token requires --enable")

    try:
        updated = apply(
            contents,
            enabled=bool(args.enable),
            service_url=args.service_url,
            token=token,
            rotate_token=bool(args.rotate_token),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if updated != contents or not path.exists():
        write_atomic(path, updated)
        print("updated" if contents else "created")
    else:
        print("unchanged")
    print(describe(updated, path))
    print("Restart aiswe-backend for the change to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
