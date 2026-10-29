"""
backend/routes/auth.py
----------------------
Authentication routes:
- POST /auth/register: User account creation
- POST /auth/login: JWT token authentication with Argon2
- GET /auth/me: Retrieve current user profile
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.entities import User
from backend.rate_limiter import check_auth_rate_limit
from backend.schemas import TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse
from backend.security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_auth_rate_limit)],
    summary="Register a new user account",
)
async def register(
    payload: UserRegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Register a new user with email and password."""
    # Check if email is already taken
    existing = await db.execute(select(User).where(User.email == payload.email.lower().strip()))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    user = User(
        email=payload.email.lower().strip(),
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(check_auth_rate_limit)],
    summary="Authenticate and receive JWT access token",
)
async def login(
    payload: UserLoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Authenticate user with email and password, returning JWT access token."""
    query = select(User).where(User.email == payload.email.lower().strip())
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(subject=user.id)
    return TokenResponse(access_token=token, token_type="bearer")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Return the profile of the currently authenticated user."""
    return current_user
