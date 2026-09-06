"""What a participant may see of a course's examples.

The seed table holds three things a student has no business reading: the
instructor's review notes and validation detail, the grounding evidence the
generator recorded, and other students' unreviewed contributions. Staff read
the whole record; a participant reads this projection.

Visible to a participant:

- every example the instructor approved (or edited and thereby approved),
  whoever wrote it — these are the course's reviewed questions, and the
  Compare page suggests from them;
- the participant's own contributions in any state, marked `mine`, so the
  Contribute page can show what they added and let them remove it.

Not visible: unreviewed AI drafts, rejected examples, and other participants'
contributions until an instructor approves them. On every visible record the
review-internal fields are dropped.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

STUDENT_VISIBLE_REVIEW_STATUSES = frozenset({"approved", "edited"})

#: Fields that describe the review process or the generator's evidence rather
#: than the example itself, and the attribution a student has no use for.
INTERNAL_FIELDS = frozenset(
    {
        "reviewNotes",
        "reviewedAt",
        "validation",
        "notes",
        "evidenceQuote",
        "factId",
        "normalizedQuestionKey",
        "originalQuestion",
        "originalAnswer",
        "sourceChunkIds",
        "questionType",
        "participantId",
    }
)


def review_status_of(record: Mapping[str, Any]) -> str:
    """The same fallback `resolve_review_status` applies: reviewStatus, then status."""
    for key in ("reviewStatus", "status"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "generated"


def seed_for_participant(
    record: Mapping[str, Any], participant_id: str
) -> dict[str, Any] | None:
    """The participant's view of one seed, or None if they may not see it."""
    mine = record.get("participantId") == participant_id
    if not mine and review_status_of(record) not in STUDENT_VISIBLE_REVIEW_STATUSES:
        return None
    public = {key: value for key, value in record.items() if key not in INTERNAL_FIELDS}
    public["mine"] = mine
    return public


def seeds_for_participant(
    records: Iterable[Mapping[str, Any]], participant_id: str
) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for record in records:
        projected = seed_for_participant(record, participant_id)
        if projected is not None:
            visible.append(projected)
    return visible


#: What a student contribution may specify. Everything else — origin, review
#: state, validation, evidence — is decided by the server.
CONTRIBUTION_FIELDS = frozenset(
    {
        "id",
        "instruction",
        "question",
        "response",
        "answer",
        "category",
        "sourceSection",
        "difficulty",
        "directlyAnswered",
        "createdAt",
    }
)


def contribution_payload(body: Mapping[str, Any]) -> dict[str, Any]:
    """A participant's seed body reduced to what a contribution may set.

    `origin` is forced to `user` so a student cannot file an example as
    AI-generated or approved, and no review field survives: every contribution
    starts unreviewed, whatever the request said.
    """
    payload = {key: value for key, value in body.items() if key in CONTRIBUTION_FIELDS}
    payload["origin"] = "user"
    return payload
