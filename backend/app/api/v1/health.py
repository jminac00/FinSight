from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter()

VERSION = "0.1.0"


@router.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.environment,
        "version": VERSION,
    }
