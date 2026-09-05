"""Password hashing for professor and administrator accounts.

scrypt from the standard library, OpenSSL underneath — a memory-hard KDF on
OWASP's list of acceptable password hashes, with no new dependency to install
on the VM. The parameters are one of OWASP's recommended scrypt settings
(N=2^14, r=8, p=5: 16 MiB of memory per hash, tens of milliseconds on the VM)
and are written into every hash, so they can be raised later and older hashes
rehashed on the next successful login.

Stored form: `scrypt$<log2 N>$<r>$<p>$<salt b64>$<key b64>`.

If Argon2id is ever preferred, `hash_password` / `verify_password` /
`needs_rehash` are the whole interface; nothing else in the application knows
what a hash looks like.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

SCHEME = "scrypt"
SCRYPT_LOG2_N = 14
SCRYPT_R = 8
SCRYPT_P = 5
SALT_BYTES = 16
KEY_BYTES = 32

#: Above the memory scrypt needs for these parameters (128 * N * r = 16 MiB),
#: with headroom for a future step up. Python's default ceiling is 32 MiB.
SCRYPT_MAXMEM = 64 * 1024 * 1024

MIN_PASSWORD_LENGTH = 12
#: Long enough for any passphrase; short enough that a hostile client cannot
#: make the KDF chew on megabytes.
MAX_PASSWORD_LENGTH = 256


def _derive(password: bytes, salt: bytes, log2_n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password,
        salt=salt,
        n=1 << log2_n,
        r=r,
        p=p,
        dklen=KEY_BYTES,
        maxmem=SCRYPT_MAXMEM,
    )


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    key = _derive(password.encode("utf-8"), salt, SCRYPT_LOG2_N, SCRYPT_R, SCRYPT_P)
    return "$".join(
        (
            SCHEME,
            str(SCRYPT_LOG2_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            _encode(salt),
            _encode(key),
        )
    )


def _parse(stored: str) -> tuple[int, int, int, bytes, bytes] | None:
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != SCHEME:
        return None
    try:
        log2_n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt, key = _decode(parts[4]), _decode(parts[5])
    except (ValueError, TypeError):
        return None
    if not (1 <= log2_n <= 24 and 1 <= r <= 64 and 1 <= p <= 64):
        return None
    if not salt or not key:
        return None
    return log2_n, r, p, salt, key


def verify_password(password: str, stored: str) -> bool:
    """True only for the password that produced `stored`.

    A malformed or foreign hash is simply "does not match"; it is never an
    error a caller could distinguish from a wrong password.
    """
    parsed = _parse(stored)
    if parsed is None:
        return False
    if len(password) > MAX_PASSWORD_LENGTH:
        return False
    log2_n, r, p, salt, expected = parsed
    try:
        candidate = _derive(password.encode("utf-8"), salt, log2_n, r, p)
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate, expected)


def needs_rehash(stored: str) -> bool:
    """Whether a hash was made with weaker parameters than the current ones."""
    parsed = _parse(stored)
    if parsed is None:
        return True
    log2_n, r, p, _, _ = parsed
    return (log2_n, r, p) < (SCRYPT_LOG2_N, SCRYPT_R, SCRYPT_P)


def password_problem(password: object) -> str | None:
    """Why a candidate password is unacceptable, or None when it is fine.

    Length only. Composition rules push people toward predictable
    substitutions; a twelve-character minimum with no maximum short of the
    KDF's own is what current guidance recommends.
    """
    if not isinstance(password, str):
        return "A password is required."
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Use at least {MIN_PASSWORD_LENGTH} characters."
    if len(password) > MAX_PASSWORD_LENGTH:
        return f"Use at most {MAX_PASSWORD_LENGTH} characters."
    if password.strip() != password:
        return "A password cannot begin or end with a space."
    return None
