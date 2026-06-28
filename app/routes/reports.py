from fastapi import APIRouter, Depends, Query
from sqlmodel import Session

from app.database.database import get_session
from app.schemas import ReportResponse
from app.services.report_service import ReportService

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{forecast_run_id}", response_model=ReportResponse)
def get_report(
    forecast_run_id: int,
    created_by: int = Query(default=1),
    session: Session = Depends(get_session),
) -> ReportResponse:
    return ReportService(session).get_or_create_report(forecast_run_id, created_by=created_by)
