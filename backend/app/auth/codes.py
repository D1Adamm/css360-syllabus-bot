"""Classroom join codes.

The thing a professor writes on the board and a student types on a laptop:
six characters such as `7K4P9X`. It is the credential a student invitation is
redeemed with, so it has to be short enough to type and read aloud and
unguessable enough that trying codes at random is not a way in.

Alphabet: upper-case letters and digits with the four look-alikes removed
(0 and O, 1 and I, plus L). Thirty-one symbols, six positions: about 887
million codes. A classroom has a handful active at once, and the join route
throttles failed attempts per client and globally, so guessing one is not
practical. Raising `CODE_LENGTH` is the only change needed if that ever
stops being true.

Codes are stored as typed, not hashed. They have to be shown on the
professor's page again and again, they grant exactly what a seat in the class
already grants, and a six-character code cannot be protected by an unsalted
hash — the whole space is enumerable in seconds. The protection is server-side:
the code maps to an invitation row that decides everything, and the row can be
revoked or replaced at any time.
"""

from __future__ import annotations

import re
import secrets

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 6

_SEPARATORS = re.compile(r"[\s\-_.]+")
_ALPHABET_SET = frozenset(CODE_ALPHABET)


def generate_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def normalize_code(raw: object) -> str | None:
    """The canonical form of what a student typed, or None if it cannot be one.

    Forgiving about what people do to a code — lower case, spaces, hyphens, or
    pasting the whole join link — and strict about what comes out: exactly
    `CODE_LENGTH` characters from the alphabet, so a lookup never sees anything
    else.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if "/" in candidate:
        # A pasted link: keep the last path segment, drop any query string.
        candidate = candidate.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
    candidate = _SEPARATORS.sub("", candidate).upper()
    if len(candidate) != CODE_LENGTH:
        return None
    if any(character not in _ALPHABET_SET for character in candidate):
        return None
    return candidate


def is_code(value: object) -> bool:
    return normalize_code(value) == value
