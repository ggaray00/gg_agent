"""Users: email + password, and the hashing that keeps passwords out of the table.

Deliberately minimal — an identity to own sessions, not an auth system. There
are no tokens or password resets. Passwords are hashed with scrypt from the
standard library (salted, memory-hard), so no new dependency.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import secrets
import uuid

from . import UserInfo

MIN_PASSWORD_CHARS = 8
# scrypt cost: n=2**14, r=8 → ~16MB and a few tens of ms per hash.
_N, _R, _P, _DKLEN = 2 ** 14, 8, 1, 32


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    local, _, domain = email.partition("@")
    if not local or "." not in domain or " " in email:
        raise ValueError(f"not a valid email address: {email!r}")
    return email


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    b64 = lambda b: base64.b64encode(b).decode()   # noqa: E731
    return f"scrypt${_N}${_R}${_P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt),
                                n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


async def register_user(store, email: str, password: str) -> UserInfo:
    """Create a user. Raises ValueError for a bad email, a short password, or an
    email that is already registered."""
    email = normalize_email(email)
    if len(password or "") < MIN_PASSWORD_CHARS:
        raise ValueError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
    return await store.create_user(uuid.uuid4().hex, email, hash_password(password))


async def authenticate(store, email: str, password: str) -> UserInfo | None:
    """The user, if the email exists and the password matches; otherwise None
    (the caller can't tell which one was wrong, by design)."""
    try:
        email = normalize_email(email)
    except ValueError:
        return None
    user = await store.get_user_by_email(email)
    if user is None:
        verify_password(password or "", _dummy_hash())   # same cost either way
        return None
    return user if verify_password(password or "", user.password_hash) else None


@functools.cache
def _dummy_hash() -> str:
    return hash_password(secrets.token_hex(8))
