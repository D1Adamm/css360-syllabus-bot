"""Every route in the application, and who may call it.

This table is the authorization contract in one place. Two tests read it:

- `test_route_auth_coverage.py` checks that every route the application mounts
  is listed here with a class, and that the guard the class names is really
  declared on the route. A new route that is not added here fails the suite,
  which is the point: nothing can ship without saying who may call it.
- `test_authorization_matrix.py` drives every route with every kind of
  principal and asserts the 401/403 outcomes the class implies.

Classes
-------
public               no principal needed (health, sign-in, join, invitation preview)
session_optional     any principal including none (session introspection, logout)
signed_in            any principal but none (the course list, scoped in the handler)
course_access_body   the course is named in the request body and authorized in the
                     handler with `authorize_course_access`; expectations are those
                     of `require_course_access`
require_user         any professor or administrator
require_admin        administrators only
require_course_staff administrators, or a professor holding a membership in the
                     course in the path
require_course_access course staff, or a participant who joined the course in the path
require_participant  a participant who joined the course in the path
worker               the Tillicum runner's shared header token, never a browser
"""

from __future__ import annotations

PUBLIC = "public"
SESSION_OPTIONAL = "session_optional"
SIGNED_IN = "signed_in"
COURSE_ACCESS_BODY = "course_access_body"
REQUIRE_USER = "require_user"
REQUIRE_ADMIN = "require_admin"
REQUIRE_COURSE_STAFF = "require_course_staff"
REQUIRE_COURSE_ACCESS = "require_course_access"
REQUIRE_PARTICIPANT = "require_participant"
WORKER = "worker"

#: Which dependency callable must appear in a route's dependency tree for each
#: class. `None` means no guard is required (public); the route may still name
#: `require_csrf` or `current_principal`.
GUARD_FOR_CLASS: dict[str, str | None] = {
    PUBLIC: None,
    SESSION_OPTIONAL: "current_principal",
    SIGNED_IN: "current_principal",
    COURSE_ACCESS_BODY: "current_principal",
    REQUIRE_USER: "require_user",
    REQUIRE_ADMIN: "require_admin",
    REQUIRE_COURSE_STAFF: "require_course_staff",
    REQUIRE_COURSE_ACCESS: "require_course_access",
    REQUIRE_PARTICIPANT: "require_participant",
    WORKER: "require_worker_token",
}

_C = "/api/courses/{course_id}"
_DB = "/api/db/courses/{course_id}"
_Q = "/api/training-queue"

CLASSIFICATION: dict[tuple[str, str], str] = {
    # ---- health and inference (main.py); the root paths are on-VM aliases ----
    ("GET", "/health"): PUBLIC,
    ("GET", "/api/health"): PUBLIC,
    ("POST", "/base-model/generate"): COURSE_ACCESS_BODY,
    ("POST", "/api/base-model/generate"): COURSE_ACCESS_BODY,
    ("POST", "/rag/generate"): COURSE_ACCESS_BODY,
    ("POST", "/api/rag/generate"): COURSE_ACCESS_BODY,
    ("POST", "/fine-tuned/generate"): COURSE_ACCESS_BODY,
    ("POST", "/api/fine-tuned/generate"): COURSE_ACCESS_BODY,
    ("POST", "/fine-tuned-rag/generate"): COURSE_ACCESS_BODY,
    ("POST", "/api/fine-tuned-rag/generate"): COURSE_ACCESS_BODY,
    ("GET", "/fine-tuned/health"): REQUIRE_ADMIN,
    ("GET", "/api/fine-tuned/health"): REQUIRE_ADMIN,
    # ---- syllabus, seeds, training (main.py) ----
    ("POST", f"{_C}/syllabus"): REQUIRE_COURSE_STAFF,
    ("GET", f"{_C}/syllabus/text"): REQUIRE_COURSE_ACCESS,
    ("GET", f"{_C}/chunks"): REQUIRE_ADMIN,
    ("GET", f"{_C}/seeds"): REQUIRE_COURSE_ACCESS,
    ("POST", f"{_C}/seeds/{{seed_id}}/review"): REQUIRE_COURSE_STAFF,
    ("POST", f"{_C}/seeds/generate"): REQUIRE_ADMIN,
    ("POST", f"{_C}/seeds/generate-starter"): REQUIRE_ADMIN,
    ("POST", f"{_C}/seeds/top-up"): REQUIRE_ADMIN,
    ("POST", f"{_C}/seeds/quality-check"): REQUIRE_ADMIN,
    ("POST", f"{_C}/seeds/export-approved"): REQUIRE_ADMIN,
    ("GET", f"{_C}/seeds/approved-export-status"): REQUIRE_ADMIN,
    ("POST", f"{_C}/seeds/prepare-training-split"): REQUIRE_ADMIN,
    ("POST", f"{_C}/facts/inventory"): REQUIRE_ADMIN,
    ("POST", f"{_C}/facts/allocation"): REQUIRE_ADMIN,
    ("POST", f"{_C}/training-runs"): REQUIRE_ADMIN,
    ("POST", f"{_C}/training-runs/retry"): REQUIRE_ADMIN,
    ("POST", f"{_C}/training/launch"): REQUIRE_ADMIN,
    ("GET", "/api/training/launch-capability"): REQUIRE_ADMIN,
    ("GET", "/api/starter-generation/status"): REQUIRE_ADMIN,
    # ---- persistence (db_routes.py) ----
    ("GET", "/api/db/courses"): SIGNED_IN,
    ("POST", "/api/db/courses"): REQUIRE_USER,
    ("GET", _DB): REQUIRE_COURSE_ACCESS,
    ("PATCH", _DB): REQUIRE_COURSE_STAFF,
    ("GET", f"{_DB}/starter-seed-generation"): REQUIRE_COURSE_STAFF,
    ("PATCH", f"{_DB}/starter-seed-generation"): REQUIRE_ADMIN,
    ("GET", f"{_DB}/seeds"): REQUIRE_COURSE_ACCESS,
    ("POST", f"{_DB}/seeds"): REQUIRE_COURSE_ACCESS,
    ("GET", f"{_DB}/seeds/{{seed_id}}"): REQUIRE_COURSE_STAFF,
    ("PATCH", f"{_DB}/seeds/{{seed_id}}"): REQUIRE_COURSE_STAFF,
    ("DELETE", f"{_DB}/seeds/{{seed_id}}"): REQUIRE_COURSE_ACCESS,
    ("POST", f"{_DB}/seeds/{{seed_id}}/review"): REQUIRE_COURSE_STAFF,
    ("GET", f"{_DB}/evaluations"): REQUIRE_COURSE_ACCESS,
    ("POST", f"{_DB}/evaluations"): REQUIRE_PARTICIPANT,
    ("DELETE", f"{_DB}/evaluations"): REQUIRE_ADMIN,
    ("DELETE", f"{_DB}/evaluations/{{evaluation_id}}"): REQUIRE_ADMIN,
    ("GET", f"{_DB}/model"): REQUIRE_COURSE_STAFF,
    ("GET", f"{_DB}/model-request"): REQUIRE_COURSE_STAFF,
    ("POST", f"{_DB}/model-request"): REQUIRE_COURSE_STAFF,
    ("PATCH", f"{_DB}/model-request"): REQUIRE_ADMIN,
    ("GET", f"{_DB}/training-runs"): REQUIRE_ADMIN,
    ("POST", f"{_DB}/training-runs"): REQUIRE_ADMIN,
    ("GET", f"{_DB}/training-runs/{{run_id}}"): REQUIRE_ADMIN,
    ("PATCH", f"{_DB}/training-runs/{{run_id}}"): REQUIRE_ADMIN,
    ("GET", "/api/db/serving-session"): REQUIRE_ADMIN,
    # ---- sessions (auth_routes.py) ----
    ("POST", "/api/auth/login"): PUBLIC,
    ("POST", "/api/auth/logout"): SESSION_OPTIONAL,
    ("GET", "/api/auth/session"): SESSION_OPTIONAL,
    ("POST", "/api/auth/join"): SESSION_OPTIONAL,
    ("GET", "/api/auth/invitations/{token}"): PUBLIC,
    ("POST", "/api/auth/invitations/{token}/accept"): PUBLIC,
    ("POST", "/api/auth/password"): REQUIRE_USER,
    # ---- administration (admin_routes.py) ----
    ("GET", "/api/admin/users"): REQUIRE_ADMIN,
    ("PATCH", "/api/admin/users/{user_id}"): REQUIRE_ADMIN,
    ("PUT", "/api/admin/users/{user_id}/courses/{course_id}"): REQUIRE_ADMIN,
    ("DELETE", "/api/admin/users/{user_id}/courses/{course_id}"): REQUIRE_ADMIN,
    ("POST", "/api/admin/users/{user_id}/reset-invite"): REQUIRE_ADMIN,
    ("POST", "/api/admin/invitations"): REQUIRE_ADMIN,
    ("GET", "/api/admin/invitations"): REQUIRE_ADMIN,
    ("POST", "/api/admin/invitations/{invitation_id}/revoke"): REQUIRE_ADMIN,
    ("GET", "/api/admin/audit"): REQUIRE_ADMIN,
    # ---- classroom codes (student_invite_routes.py) ----
    ("GET", f"{_C}/student-invites"): REQUIRE_COURSE_STAFF,
    ("POST", f"{_C}/student-invites"): REQUIRE_COURSE_STAFF,
    ("POST", f"{_C}/student-invites/{{invitation_id}}/revoke"): REQUIRE_COURSE_STAFF,
    # ---- the cluster's queue (training_queue_routes.py) ----
    ("GET", f"{_Q}/pending"): WORKER,
    ("POST", f"{_Q}/claim"): WORKER,
    ("POST", f"{_Q}/courses/{{course_id}}/runs/{{run_id}}/release"): WORKER,
    ("POST", f"{_Q}/courses/{{course_id}}/runs/{{run_id}}/submitted"): WORKER,
    ("POST", f"{_Q}/courses/{{course_id}}/runs/{{run_id}}/submission-failed"): WORKER,
    ("POST", f"{_Q}/courses/{{course_id}}/runs/{{run_id}}/failed"): WORKER,
    ("POST", f"{_Q}/courses/{{course_id}}/runs/{{run_id}}/completed"): WORKER,
    ("GET", f"{_Q}/courses/{{course_id}}/runs/{{run_id}}/dataset"): WORKER,
    ("GET", f"{_Q}/courses/{{course_id}}/runs/{{run_id}}/dataset/files/{{name}}"): WORKER,
    ("POST", f"{_Q}/courses/{{course_id}}/model-versions"): WORKER,
    ("POST", f"{_Q}/courses/{{course_id}}/model-versions/{{version}}/published"): WORKER,
    ("PUT", f"{_Q}/serving-sessions/{{session_id}}"): WORKER,
    ("POST", f"{_Q}/serving-sessions/{{session_id}}/stopped"): WORKER,
    ("GET", f"{_Q}/serving-session"): WORKER,
}


def iter_api_routes(app):
    """Every APIRoute the application mounts, however deeply it is included."""
    from fastapi.routing import APIRoute

    def walk(routes):
        for route in routes:
            if isinstance(route, APIRoute):
                yield route
                continue
            # `app.include_router` mounts an `_IncludedRouter` wrapper in this
            # FastAPI version; the routes are on the router it wraps.
            nested = getattr(route, "routes", None) or getattr(
                getattr(route, "original_router", None), "routes", None
            )
            if nested:
                yield from walk(nested)

    yield from walk(app.routes)


def guard_names(route) -> set[str]:
    """Names of every dependency callable in the route's dependency tree."""
    names: set[str] = set()

    def walk(dependant) -> None:
        for sub in dependant.dependencies:
            call = getattr(sub, "call", None)
            if call is not None:
                names.add(getattr(call, "__name__", repr(call)))
            walk(sub)

    walk(route.dependant)
    return names
