"""
backend/security.py
-------------------
Authentication & cryptography helpers:
- Password hashing and verification with Argon2 using pwdlib.
- JWT access token generation and decoding.
- FastAPI dependency get_current_user for protected routes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database import get_db

settings = get_settings()

# Modern Argon2 password hashing via pwdlib
password_hash = PasswordHash((Argon2Hasher(),))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


# ── Password helpers ──────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Hash password using Argon2."""
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain text password against stored Argon2 hash."""
    try:
        return password_hash.verify(plain_password, hashed_password)
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────


def create_access_token(subject: str | int, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {"sub": str(subject), "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> str:
    """
    Decode and verify a JWT access token. Returns the subject claim (user_id).
    Raises JWTError if invalid or expired.
    """
    payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    sub: str | None = payload.get("sub")
    if sub is None:
        raise JWTError("Token missing 'sub' subject claim.")
    return sub


# ── FastAPI User Dependency ───────────────────────────────────────────────────


async def get_current_user(
    db: Annotated[AsyncSession, Depends(get_db)],
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
):
    """
    Authenticate request via JWT bearer token or guest session.
    Raises 401 HTTPException if no credentials provided or token is invalid.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from backend.models.entities import User

    if token in ("guest-session", "anonymous"):
        user_id = 1
    else:
        try:
            sub = decode_access_token(token)
            user_id = int(sub)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if user is None:
        if token in ("guest-session", "anonymous"):
            user = User(id=1, email="guest@cinematch.ai", hashed_password="guest")
            db.add(user)
            await db.commit()
            await db.refresh(user)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return user
