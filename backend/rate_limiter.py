"""
backend/rate_limiter.py
-----------------------
In-memory sliding-window rate limiter for sensitive authentication endpoints.
Protects /auth/login and /auth/register against brute-force attacks and credential stuffing.
"""

import time
from collections import defaultdict

from fastapi import HTTPException, status
from starlette.requests import Request


class RateLimiter:
    """Sliding-window rate limiter based on client IP."""

    def __init__(self, max_requests: int = 40, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window_seconds

        # Clean timestamps older than window
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]

        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many authentication attempts. Please wait a minute before trying again.",
            )

        self.requests[client_ip].append(now)


auth_rate_limiter = RateLimiter(max_requests=40, window_seconds=60)


async def check_auth_rate_limit(request: Request) -> None:
    """FastAPI dependency to rate-limit authentication endpoints."""
    auth_rate_limiter.check(request)
