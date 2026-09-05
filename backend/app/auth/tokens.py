"""Opaque secrets: session tokens and privileged invitation tokens.

A token is 256 bits from the operating system's CSPRNG, URL-safe so it can sit
in a cookie or a path segment unchanged. Only its SHA-256 is ever stored. That
is enough: a token this long cannot be brute-forced, so a slow hash would add
cost without adding protection, and a database dump yields nothing that can be
presented to the API.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

TOKEN_BYTES = 32

#: What `secrets.token_urlsafe(32)` produces: 43 characters from the URL-safe
#: base64 alphabet, no padding. Checked before a lookup so garbage never
#: reaches the database and so a malformed value is refused in constant shape.
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


def generate_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hashes_match(left: str, right: str) -> bool:
    """Constant-time comparison for hex digests."""
    return hmac.compare_digest(left, right)


def is_plausible_token(value: object) -> bool:
    return isinstance(value, str) and bool(TOKEN_PATTERN.match(value))
