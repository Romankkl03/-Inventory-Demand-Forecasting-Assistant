from fastapi import APIRouter

from app.routes.datasets import router as datasets_router
from app.routes.forecast import router as forecast_router
from app.routes.health import router as health_router
from app.routes.models import router as models_router
from app.routes.recommendations import router as recommendations_router
from app.routes.reports import router as reports_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(datasets_router)
api_router.include_router(forecast_router)
api_router.include_router(recommendations_router)
api_router.include_router(reports_router)
api_router.include_router(models_router)
