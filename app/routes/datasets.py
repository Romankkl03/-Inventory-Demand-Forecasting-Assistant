from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.database.database import get_session
from app.schemas import DatasetUploadRequest, DatasetUploadResponse
from app.services.data_service import DataService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetUploadResponse)
def upload_dataset(
    payload: DatasetUploadRequest,
    session: Session = Depends(get_session),
) -> DatasetUploadResponse:
    return DataService(session).upload_dataset(payload)
