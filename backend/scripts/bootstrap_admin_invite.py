#!/usr/bin/env python3
"""Mint the first administrator invitation.

Run on the application VM, from the backend directory, with the same
`backend/.env` the service uses:

    cd ~/css360-syllabus-bot/backend
    .venv/bin/python scripts/bootstrap_admin_invite.py --origin https://aiswe.uwb.edu

It prints one link. Open it within the hour, choose an email address, a
display name and a password, and you are the first administrator. Every later
administrator or professor is invited from the People page.

Refuses to run while an active administrator exists; `--force` overrides that
for lockout recovery. Either way the invitation is recorded in admin_actions.

Nothing is written to `.env`, and no password is handled here.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_backend_env  # noqa: E402

load_backend_env()

from app.auth.bootstrap import (  # noqa: E402
    DEFAULT_LIFETIME,
    AdminAlreadyExistsError,
    mint_bootstrap_invitation,
)
from app.db import DatabaseConfigurationError, db_connection  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--origin",
        default=os.getenv("APP_PUBLIC_ORIGIN", "").strip() or None,
        help="Site origin used to print a full link, e.g. https://aiswe.uwb.edu "
        "(defaults to APP_PUBLIC_ORIGIN when set).",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_LIFETIME.total_seconds() / 3600,
        help="How long the invitation stays valid (default 1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Mint even though an active administrator exists (lockout recovery).",
    )
    args = parser.parse_args(argv)

    if args.hours <= 0 or args.hours > 24:
        parser.error("--hours must be between 0 and 24.")

    try:
        with db_connection() as conn:
            invitation = mint_bootstrap_invitation(
                conn, force=args.force, lifetime=timedelta(hours=args.hours)
            )
    except DatabaseConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except AdminAlreadyExistsError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3

    origin = (args.origin or "").rstrip("/")
    link = f"{origin}{invitation['path']}" if origin else invitation["path"]
    print("Administrator invitation created.")
    print(f"  expires: {invitation['expiresAt']}")
    print(f"  link:    {link}")
    if not origin:
        print("  (prefix the path with the site origin, e.g. https://aiswe.uwb.edu)")
    print("Open it in a browser, choose an email address, a display name and a password.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
