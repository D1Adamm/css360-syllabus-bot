"""Who is asking.

A request can carry a staff session (a professor or administrator), the
participant sessions of the courses this browser joined (one anonymous
participant per course), both, or neither. `Principal` holds whatever was
found and answers the only questions the routes ask: is this an
administrator, may they act on this course as staff, may they read this
course at all, and which participant are they in it.

Every answer here comes from rows the server wrote — `users.role`,
`course_memberships`, `participants.course_id` — never from anything the
client sent. The frontend has no way to make one of these methods return True.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ADMIN = "admin"
PROFESSOR = "professor"
STAFF_ROLES = frozenset({ADMIN, PROFESSOR})


@dataclass(frozen=True)
class StaffUser:
    user_id: str
    email: str
    display_name: str
    role: str
    #: Courses this user holds an instructor membership in. Empty for an
    #: administrator, whose access does not go through memberships.
    course_ids: frozenset[str] = field(default_factory=frozenset)
    session_id: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN


@dataclass(frozen=True)
class Participant:
    participant_id: str
    course_id: str
    session_id: str | None = None
    #: SHA-256 of the cookie token that resolved this participant. Lets the join
    #: route rebuild the cookie from the tokens that are still live; it is the
    #: stored hash, never the token.
    token_hash: str | None = None


@dataclass(frozen=True, init=False)
class Principal:
    """Staff session, the browser's course participants, both, or neither.

    A browser holds one participant per course it joined — each its own
    pseudonymous research identity, never merged across courses. `participant`
    is accepted as a one-course convenience.
    """

    user: StaffUser | None
    participants: tuple[Participant, ...]

    def __init__(
        self,
        user: StaffUser | None = None,
        participant: Participant | None = None,
        participants: tuple[Participant, ...] = (),
    ) -> None:
        combined = tuple(participants) + ((participant,) if participant is not None else ())
        object.__setattr__(self, "user", user)
        object.__setattr__(self, "participants", combined)

    @property
    def is_anonymous(self) -> bool:
        return self.user is None and not self.participants

    @property
    def is_staff(self) -> bool:
        return self.user is not None

    @property
    def is_admin(self) -> bool:
        return self.user is not None and self.user.is_admin

    @property
    def is_professor(self) -> bool:
        return self.user is not None and self.user.role == PROFESSOR

    def can_staff_course(self, course_id: str) -> bool:
        """Act on a course as its staff: administrators always, professors by membership."""
        if self.user is None:
            return False
        if self.user.is_admin:
            return True
        return course_id in self.user.course_ids

    def participant_for(self, course_id: str) -> Participant | None:
        """The participant identity this request holds in `course_id`, if any."""
        for participant in self.participants:
            if participant.course_id == course_id:
                return participant
        return None

    def participant_course_ids(self) -> frozenset[str]:
        """Every course this browser joined with a class code."""
        return frozenset(participant.course_id for participant in self.participants)

    def can_access_course(self, course_id: str) -> bool:
        """Read a course: its staff, or a participant who joined it."""
        return self.can_staff_course(course_id) or self.participant_for(course_id) is not None

    def staff_course_ids(self) -> frozenset[str] | None:
        """Courses the staff half may act on, or None meaning every course."""
        if self.user is None:
            return frozenset()
        if self.user.is_admin:
            return None
        return self.user.course_ids


ANONYMOUS = Principal()
