#!/usr/bin/env python3
"""Run the real backend app on a laptop for `classroom_load_test.py`, without a database.

    COURSE_STORAGE_BACKEND=local FINETUNED_SERVICE_URL=http://127.0.0.1:9011 \\
        backend/.venv/bin/python scripts/classroom_load_harness.py \\
        --course-id css-360-winter-2026-a7rp --version v1 --port 8011

Everything a classroom generate request does runs as written — retrieval from
the course's local index, the prompt builders, the generation queue, the calls
to Ollama and to the fine-tuned service. Two things are replaced, because they
read PostgreSQL:

- who is asking: every request is an administrator (so no sign-in, no cookie);
- which fine-tuned version the course serves: the `--version` given here.

So this is a measuring instrument, not a server. It binds loopback only and
refuses to start when `DATABASE_URL` is set — in the environment or in
`backend/.env`, checked before and again after the app is imported — which
is how a deployment is configured; the VM is measured through its real
backend instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "backend"


REFUSAL = (
    "DATABASE_URL is configured{where}. This harness signs every request in as an "
    "administrator and must never run where a real database is reachable; on the VM, "
    "measure through the real backend with classroom_load_test.py instead."
)


def deployment_refusal(environ, env_file: Path | None) -> str | None:
    """Why this process must not start, or None.

    A deployment is recognised by its database: `DATABASE_URL` in the
    environment, or in `backend/.env`, which the backend loads on import and
    which on the VM holds the production DSN.
    """
    if (environ.get("DATABASE_URL") or "").strip():
        return REFUSAL.format(where=" in the environment")
    if env_file is not None and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == "DATABASE_URL" and value.strip().strip("'\""):
                return REFUSAL.format(where=f" in {env_file}")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--course-id", required=True, help="The course the harness serves fine-tuned answers for.")
    parser.add_argument("--version", default="v1", help="The version the course is treated as serving (default %(default)s).")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args(argv)

    refusal = deployment_refusal(os.environ, BACKEND / ".env")
    if refusal:
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(BACKEND))
    os.chdir(BACKEND)

    import uvicorn

    from app import grounded_rag, main as backend_main

    # Importing the app loads backend/.env (app.config). Check again with
    # whatever that put into the environment, before serving anything.
    refusal = deployment_refusal(os.environ, None)
    if refusal:
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 2

    from app.auth.dependencies import current_principal
    from app.auth.principal import Principal, StaffUser

    admin = Principal(user=StaffUser(user_id="load-harness", email="load-harness@localhost", display_name="Load harness", role="admin"))
    backend_main.app.dependency_overrides[current_principal] = lambda: admin

    def resolve(course_id: str) -> dict[str, str]:
        if course_id != args.course_id:
            raise RuntimeError(f"Harness only serves {args.course_id}")
        return {"courseId": course_id, "version": args.version}

    backend_main.resolve_current_course_model = resolve
    grounded_rag.resolve_current_course_model = resolve

    uvicorn.run(backend_main.app, host="127.0.0.1", port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
