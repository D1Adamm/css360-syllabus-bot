#!/usr/bin/env python3
"""Configure an isolated test instance of the backend on the VM, without secrets on screen.

Used by docs/answer-cache-vm-test.md to try `feat/compare-answer-cache` beside
production: a second checkout, its own PostgreSQL database, its own ports. The
production checkout, its `backend/.env` and its database are only ever read.

    # What production's database is called and who owns it (no password shown),
    # and the exact command that creates the test database:
    python3 scripts/cache_test_env.py describe --production-env ~/css360-syllabus-bot/backend/.env

    # Write this checkout's backend/.env for the test database (mode 600):
    python3 scripts/cache_test_env.py write --production-env ~/css360-syllabus-bot/backend/.env

    # Before any command that changes a database: exits non-zero unless this
    # checkout's backend/.env names a database that is not production's.
    python3 scripts/cache_test_env.py guard --production-env ~/css360-syllabus-bot/backend/.env

The test `backend/.env` reuses production's database server, role and password
(read from production's file, never printed) with a different database name,
copies the few non-secret settings that decide what a model answers, and turns
the answer cache on. Nothing else of production's configuration is copied: no
worker token, no benchmark token, no site origin.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_ENV = REPO_ROOT / "backend" / ".env"
DEFAULT_DATABASE = "syllabus_lab_cachetest"
DEFAULT_BACKEND_PORT = 8011
DEFAULT_FRONTEND_PORT = 5180

DATABASE_NAME = re.compile(r"^[a-z][a-z0-9_]{2,62}$")

#: Production settings that decide what a model answers (which Ollama, which
#: models, which fine-tuned service, the queue limits). None is a secret. A
#: test instance that differed in these would be testing something else.
COPIED_SETTINGS = (
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_TIMEOUT_SECONDS",
    "BASE_MODEL_KEEP_ALIVE",
    "FINETUNED_SERVICE_URL",
    "FINETUNED_SERVICE_TIMEOUT_SECONDS",
    "GENERATION_MAX_CONCURRENCY",
    "GENERATION_MAX_WAITING",
    "GENERATION_QUEUE_TIMEOUT_SECONDS",
    "GENERATION_MODE_FILE",
)


class SetupError(Exception):
    """The request cannot be carried out; the message says why and names no secret."""


def parse_env_file(path: Path) -> dict[str, str]:
    """`KEY=value` lines as python-dotenv and a shell `source` both read them."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SetupError(f"Cannot read {path}: {exc.strerror or exc}.") from exc
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # An unquoted value ends at an inline comment.
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[key] = value
    return values


def database_parts(url: str, *, source: str) -> dict[str, str]:
    """Where a DSN points, without its password."""
    parts = urlsplit(url)
    name = parts.path.lstrip("/")
    if parts.scheme not in ("postgresql", "postgres") or not parts.hostname or not name:
        raise SetupError(f"DATABASE_URL in {source} is not a postgresql:// address with a database name.")
    return {
        "database": name,
        "host": parts.hostname,
        "port": str(parts.port or 5432),
        "role": parts.username or "",
    }


def production_url(production_env: Path) -> str:
    url = parse_env_file(production_env).get("DATABASE_URL", "").strip()
    if not url:
        raise SetupError(f"No DATABASE_URL in {production_env}.")
    return url


def with_database(url: str, database: str) -> str:
    """The same server, role and password, another database."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment))


def check_database_name(database: str, production_database: str) -> None:
    if not DATABASE_NAME.fullmatch(database):
        raise SetupError("The test database name must be lower-case letters, digits and underscores (3-63 characters).")
    if database == production_database:
        raise SetupError("The test database must not be the production database.")


def _line(key: str, value: str) -> str:
    if "'" in value or "\n" in value:
        raise SetupError(f"{key} contains a character this file cannot carry safely.")
    # Single quotes: read the same by python-dotenv and by a shell `source`,
    # whatever is in a password.
    return f"{key}='{value}'"


def render_test_env(
    production: dict[str, str],
    *,
    database: str,
    backend_port: int,
    frontend_port: int,
) -> str:
    url = production.get("DATABASE_URL", "").strip()
    if not url:
        raise SetupError("Production's environment has no DATABASE_URL.")
    check_database_name(database, database_parts(url, source="production's environment")["database"])
    origin = f"http://localhost:{frontend_port}"
    lines = [
        "# Isolated test instance of the backend (scripts/cache_test_env.py).",
        "# NOT production: its own database, ports and data directory.",
        f"# Backend port {backend_port}; frontend {origin} (through an SSH tunnel).",
        "",
        _line("DATABASE_URL", with_database(url, database)),
        "",
        "# Plain HTTP on loopback, reached through an SSH tunnel.",
        _line("AUTH_COOKIE_SECURE", "false"),
        _line("APP_PUBLIC_ORIGIN", origin),
        _line("CORS_ALLOWED_ORIGINS", origin),
        "",
        "# Nothing in the background: no starter generation, no readiness monitor.",
        _line("AUTO_STARTER_SEED_GENERATION", "false"),
        _line("FINETUNED_READINESS_CHECK_SECONDS", "0"),
        "",
        "# The feature under test. The quiet period is short because this",
        "# instance only sees its own requests.",
        _line("ANSWER_CACHE_ENABLED", "true"),
        _line("CLASSROOM_PREP_QUIET_SECONDS", "20"),
        "",
        "# Copied from production so the same models answer.",
    ]
    lines += [_line(key, production[key]) for key in COPIED_SETTINGS if production.get(key, "").strip()]
    return "\n".join(lines) + "\n"


def write_private(path: Path, content: str) -> None:
    """Create the file readable by its owner only, never widening it afterwards."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.chmod(path, 0o600)


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def describe(args: argparse.Namespace) -> int:
    parts = database_parts(production_url(args.production_env), source=str(args.production_env))
    check_database_name(args.database, parts["database"])
    print(f"Production database: {parts['database']} on {parts['host']}:{parts['port']}, role {parts['role']}")
    print(f"Test database:       {args.database} (same server and role)")
    print()
    print("Create the test database with (changes nothing in production):")
    owner = f"-O {parts['role']} " if parts["role"] else ""
    print(f"  sudo -u postgres createdb {owner}{args.database}")
    return 0


def write(args: argparse.Namespace) -> int:
    if same_file(args.out, args.production_env):
        raise SetupError("Refusing to write over production's environment file.")
    if args.out.exists() and not args.force:
        raise SetupError(f"{args.out} already exists; pass --force to replace it.")
    production = parse_env_file(args.production_env)
    content = render_test_env(
        production,
        database=args.database,
        backend_port=args.backend_port,
        frontend_port=args.frontend_port,
    )
    write_private(args.out, content)
    parts = database_parts(with_database(production["DATABASE_URL"], args.database), source="the test environment")
    copied = [key for key in COPIED_SETTINGS if production.get(key, "").strip()]
    print(f"Wrote {args.out} (owner-only).")
    print(f"  database: {parts['database']} on {parts['host']}:{parts['port']}")
    print(f"  backend port {args.backend_port}, frontend http://localhost:{args.frontend_port}")
    print(f"  answer cache: on; copied from production: {', '.join(copied) or 'nothing'}")
    return 0


def guard(args: argparse.Namespace) -> int:
    if same_file(args.env, args.production_env):
        raise SetupError("This checkout's environment file IS production's. Stop.")
    test_url = parse_env_file(args.env).get("DATABASE_URL", "").strip()
    if not test_url:
        raise SetupError(f"No DATABASE_URL in {args.env}; run the write command first.")
    test = database_parts(test_url, source=str(args.env))
    production = database_parts(production_url(args.production_env), source=str(args.production_env))
    if (test["database"], test["host"], test["port"]) == (
        production["database"], production["host"], production["port"],
    ):
        raise SetupError(
            f"{args.env} points at the production database ({production['database']}). Stop: nothing was changed."
        )
    print(f"OK: this checkout uses database {test['database']}; production is {production['database']}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--production-env", type=Path, required=True, help="Production's backend/.env (read only).")

    sub = commands.add_parser("describe", help="Show where production's database is and how to create the test one.")
    common(sub)
    sub.add_argument("--database", default=DEFAULT_DATABASE, help="Test database name (default %(default)s).")
    sub.set_defaults(run=describe)

    sub = commands.add_parser("write", help="Write this checkout's backend/.env for the test database.")
    common(sub)
    sub.add_argument("--database", default=DEFAULT_DATABASE, help="Test database name (default %(default)s).")
    sub.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    sub.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    sub.add_argument("--out", type=Path, default=DEFAULT_TEST_ENV, help="Where to write (default this checkout's backend/.env).")
    sub.add_argument("--force", action="store_true", help="Replace an existing file.")
    sub.set_defaults(run=write)

    sub = commands.add_parser("guard", help="Refuse unless this checkout's database is not production's.")
    common(sub)
    sub.add_argument("--env", type=Path, default=DEFAULT_TEST_ENV, help="This checkout's backend/.env.")
    sub.set_defaults(run=guard)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except SetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
