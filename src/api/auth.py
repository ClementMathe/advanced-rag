"""
API key authentication via X-API-Key header.

Disabled when AUTH_ENABLED=false (no-op for local dev).
"""

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    key: str = Security(api_key_header),  # noqa: B008
):
    """FastAPI dependency: verify API key if auth is enabled."""
    settings = request.app.state.app_state.settings
    if settings is None or not settings.auth.enabled:
        return None  # Auth disabled

    if not key or key != settings.auth.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key
