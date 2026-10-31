"""One servability rule for both ways a version becomes the one a course serves.

`course_model_versions.deployment = online` is what inference resolves
(`course_model_resolution`). Two writers can set it:

- Admin → Models → Activate (`db_routes.activate_course_model_version`);
- the cluster's publication report
  (`training_queue_routes.record_model_version_published`, sent by
  `training/promote_qlora_adapter.sh` through `scripts/report_model_published.py`).

Activation has always asked the fine-tuned service's `/health` first and
written nothing unless it listed the exact course and version. The publication
report did not: it marked a version online as soon as the adapter was in
Tillicum's serving tree, so one operator command could point production at a
version the VM did not map, and every Fine-Tuned and Fine-Tuned + RAG request
for that course then failed with a 409 the student saw as "temporarily
unavailable". Both writers now go through this module, so there is exactly one
rule, and a version becomes `online` only after the service reachable at
`FINETUNED_SERVICE_URL` has said it can serve it.

Two refusals, both before anything is written:

- 503 when the service cannot be asked. An unverifiable publication is not one;
  the previously published version keeps serving.
- 409 when the service answers but does not list the course and version. The
  detail names what *is* servable so the operator can see what to map.

The probe is passed in by the caller rather than imported here, so each route
module keeps `check_finetuned_service_health` in its own namespace — which is
where the route tests patch it.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping

from fastapi import HTTPException

HealthProbe = Callable[[], Awaitable[Mapping[str, Any]]]


def require_ready_version(
    registry: Mapping[str, Any] | None, course_id: str, version: str
) -> None:
    """Refuse a version this course cannot serve, before or after the service is asked."""
    record = ((registry or {}).get("versions") or {}).get(version)
    if not isinstance(record, dict):
        raise HTTPException(
            status_code=404,
            detail=f'Course "{course_id}" has no registered model version "{version}".',
        )
    if record.get("status") != "ready":
        raise HTTPException(
            status_code=409,
            detail=(
                f'Version "{version}" is "{record.get("status")}", not ready. '
                "Only a ready version can be activated."
            ),
        )


def servable_versions(health: Mapping[str, Any], course_id: str) -> list[str]:
    """The versions the fine-tuned service says it can answer this course with."""
    for entry in health.get("courses") or []:
        if isinstance(entry, dict) and entry.get("courseId") == course_id:
            versions = entry.get("versions")
            return [v for v in versions if isinstance(v, str)] if isinstance(versions, list) else []
    return []


def not_servable_detail(
    course_id: str, version: str, servable: list[str], *, then: str
) -> str:
    return (
        f"The fine-tuned service cannot serve {course_id} {version}: it is not "
        "mapped, or its Ollama model is missing. Install and map it on the VM "
        "(scripts/install_finetuned_adapter.py, then "
        f"scripts/aiswe_finetuned.sh set-mapping and restart), {then} "
        f"Servable now: {', '.join(servable) if servable else 'none'}."
    )


async def require_servable_version(
    course_id: str,
    version: str,
    *,
    probe: HealthProbe,
    action: str,
    then: str = "then activate it.",
) -> list[str]:
    """Ask the fine-tuned service, and refuse unless it can serve this exact pair.

    `action` is the past participle the 503 names ("activated", "published");
    `then` closes the 409's instructions. Returns the versions the service
    listed for the course, for the caller's own messages.
    """
    try:
        health = await probe()
    except HTTPException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not confirm that the fine-tuned service can serve "
                f"{course_id} {version}, so nothing was {action}. {exc.detail}"
            ),
        ) from exc

    servable = servable_versions(health, course_id)
    if version not in servable:
        raise HTTPException(
            status_code=409,
            detail=not_servable_detail(course_id, version, servable, then=then),
        )
    return servable
