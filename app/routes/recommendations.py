from fastapi import APIRouter, Depends, Query
from sqlmodel import Session

from app.database.database import get_session
from app.schemas import RecommendationsResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/{forecast_run_id}", response_model=RecommendationsResponse)
def get_recommendations(
    forecast_run_id: int,
    include_llm: bool = Query(default=False),
    session: Session = Depends(get_session),
) -> RecommendationsResponse:
    return RecommendationService(session).get_or_create_recommendations(
        forecast_run_id,
        include_llm=include_llm,
    )
