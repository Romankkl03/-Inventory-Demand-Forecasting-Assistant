from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.database.database import get_session
from app.models import ModelVersion
from app.schemas import ModelInfoResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelInfoResponse])
def list_models(session: Session = Depends(get_session)) -> list[ModelInfoResponse]:
    models = session.exec(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()
    return [
        ModelInfoResponse(
            id=item.id,
            name=item.name,
            version=item.version,
            model_type=item.model_type.value,
            features_version=item.features_version,
            created_at=item.created_at,
            metrics_json=item.metrics_json,
        )
        for item in models
    ]
