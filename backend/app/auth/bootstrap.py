"""Minting the first administrator invitation.

There is no administrator until someone is invited, and nobody to send an
invitation until there is an administrator. This breaks the loop the same way
every later administrator is created — with a single-use, short-lived admin
invitation — except that it is minted from the command line on the VM by the
operator who already holds the database credentials, and recorded in the
audit trail with no actor.

Nothing about a password or a long-lived secret goes into `backend/.env`.
The invitation expires in an hour, works once, and the operator sets their
password through the same page every later administrator uses.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app import db_admin_actions, db_invitations, db_users
from app.auth.tokens import generate_token, hash_token

DEFAULT_LIFETIME = timedelta(hours=1)
INVITE_PATH_PREFIX = "/invite/"


class AdminAlreadyExistsError(Exception):
    """An active administrator exists; the bootstrap path is not needed."""


def mint_bootstrap_invitation(
    conn: Any,
    *,
    force: bool = False,
    lifetime: timedelta = DEFAULT_LIFETIME,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create the single-use admin invitation and return it with its token.

    Refuses when an active administrator already exists, because then the
    ordinary admin-issued invitation is the right path and this one would be
    a way around the audit trail. `--force` exists for the case where every
    administrator has locked themselves out.
    """
    moment = now or datetime.now(timezone.utc)
    existing = db_users.count_active_admins(conn)
    if existing > 0 and not force:
        raise AdminAlreadyExistsError(
            f"{existing} active administrator account(s) exist. Sign in as one of "
            "them and create an admin invitation from the People page, or rerun "
            "with --force if every administrator is locked out."
        )

    token = generate_token()
    invitation = db_invitations.create_token_invitation(
        conn,
        kind=db_invitations.ADMIN,
        token_hash=hash_token(token),
        created_by=None,
        expires_at=moment + lifetime,
        now=moment,
        label="bootstrap",
    )
    db_admin_actions.record_action(
        conn,
        actor_user_id=None,
        actor_role="operator",
        action="bootstrap.admin_invite",
        target_kind="invitation",
        target_id=invitation["invitationId"],
        detail={"forced": force, "existingAdmins": existing},
        now=moment,
    )
    return {**invitation, "token": token, "path": INVITE_PATH_PREFIX + token}
