"""Who is asking.

A request can carry a staff session (a professor or administrator), a
participant session (an anonymous student bound to one course), both, or
neither. `Principal` holds whatever was found and answers the only questions
the routes ask: is this an administrator, may they act on this course as
staff, may they read this course at all, and which participant are they in it.

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


@dataclass(frozen=True)
class Principal:
    user: StaffUser | None = None
    participant: Participant | None = None

    @property
    def is_anonymous(self) -> bool:
        return self.user is None and self.participant is None

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
        if self.participant is not None and self.participant.course_id == course_id:
            return self.participant
        return None

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
