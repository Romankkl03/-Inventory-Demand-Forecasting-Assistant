from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.database.database import get_session
from app.schemas import ForecastRunRequest, ForecastRunResponse, ForecastRunStatusResponse
from app.services.forecasting_service import ForecastingService

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("/run", response_model=ForecastRunResponse)
def run_forecast(
    payload: ForecastRunRequest,
    session: Session = Depends(get_session),
) -> ForecastRunResponse:
    return ForecastingService(session).run_forecast(payload)


@router.get("/{forecast_run_id}", response_model=ForecastRunResponse)
def get_forecast(
    forecast_run_id: int,
    session: Session = Depends(get_session),
) -> ForecastRunResponse:
    return ForecastingService(session).get_forecast(forecast_run_id)


@router.get("/run/{forecast_run_id}/status", response_model=ForecastRunStatusResponse)
def get_forecast_run_status(
    forecast_run_id: int,
    session: Session = Depends(get_session),
) -> ForecastRunStatusResponse:
    return ForecastingService(session).get_run_status(forecast_run_id)
