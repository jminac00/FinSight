from fastapi import APIRouter

from app.api.v1.deep_learning import router as dl_router
from app.api.v1.fundamental import router as fundamental_router
from app.api.v1.health import router as health_router
from app.api.v1.report import router as report_router
from app.api.v1.search import router as search_router
from app.api.v1.sentiment import router as sentiment_router
from app.api.v1.technical import router as technical_router

router = APIRouter()

router.include_router(health_router)
router.include_router(report_router)
router.include_router(sentiment_router)
router.include_router(dl_router)
router.include_router(fundamental_router)
router.include_router(technical_router)
router.include_router(search_router)
