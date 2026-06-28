"""Data service: dataset uploads and sales ingestion."""

from __future__ import annotations

from sqlmodel import Session, select

from fastapi import HTTPException, status

from app.models import Dataset, DatasetStatus, SalesRecord, Store, User
from app.schemas import DatasetUploadRequest, DatasetUploadResponse


class DataService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upload_dataset(self, payload: DatasetUploadRequest) -> DatasetUploadResponse:
        uploader = self.session.get(User, payload.uploaded_by)
        if uploader is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

        dataset = Dataset(
            name=payload.name,
            source=payload.source,
            uploaded_by=payload.uploaded_by,
            status=DatasetStatus.VALIDATING,
        )
        self.session.add(dataset)
        self.session.flush()

        inserted_stores = 0
        inserted_records = 0

        for row in payload.records:
            store = self.session.exec(
                select(Store).where(Store.external_id == row.store_external_id)
            ).first()
            if store is None:
                store = Store(
                    external_id=row.store_external_id,
                    store_type=row.store_type,
                    assortment=row.assortment,
                    competition_distance=row.competition_distance,
                )
                self.session.add(store)
                self.session.flush()
                inserted_stores += 1

            sales_record = SalesRecord(
                store_id=store.id,
                date=row.date,
                sales=row.sales,
                customers=row.customers,
                promo=row.promo,
                promo2=row.promo2,
                school_holiday=row.school_holiday,
                state_holiday=row.state_holiday,
                open=row.open,
            )
            self.session.add(sales_record)
            inserted_records += 1

        dataset.status = DatasetStatus.READY
        self.session.commit()
        self.session.refresh(dataset)

        return DatasetUploadResponse(
            dataset_id=dataset.id,
            status=dataset.status.value,
            inserted_records=inserted_records,
            inserted_stores=inserted_stores,
        )
