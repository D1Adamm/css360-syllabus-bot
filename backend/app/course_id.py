"""Reusable courseId validation for filesystem-safe backend paths and keys."""

from __future__ import annotations

import re

COURSE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
UNSAFE_COURSE_ID_CHARS = re.compile(r"[./\\[\]$]")


def is_valid_course_id(course_id: object) -> bool:
    if not isinstance(course_id, str) or course_id == "":
        return False

    if UNSAFE_COURSE_ID_CHARS.search(course_id) or ".." in course_id:
        return False

    if course_id.startswith("-") or course_id.endswith("-"):
        return False

    return COURSE_ID_PATTERN.fullmatch(course_id) is not None


def assert_valid_course_id(course_id: object) -> str:
    if not is_valid_course_id(course_id):
        raise ValueError(
            f'Invalid courseId "{course_id}": must be non-empty, use lowercase letters, '
            "numbers, and hyphens only, and must not begin/end with a hyphen or contain "
            "path-unsafe characters."
        )
    return str(course_id)
