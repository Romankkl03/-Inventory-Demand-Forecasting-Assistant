"""Forecasting service: create runs, inference, status and outputs."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models import (
    Dataset,
    Forecast,
    ForecastRun,
    ForecastRunStatus,
    ModelVersion,
    SalesRecord,
    Store,
    User,
)
from app.schemas import (
    ForecastPointResponse,
    ForecastRunRequest,
    ForecastRunResponse,
    ForecastRunStatusResponse,
)


class ForecastingService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def run_forecast(self, payload: ForecastRunRequest) -> ForecastRunResponse:
        dataset = self.session.get(Dataset, payload.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

        model_version = self.session.get(ModelVersion, payload.model_version_id)
        if model_version is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Model version not found."
            )
        if self.session.get(User, payload.created_by) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

        forecast_run = ForecastRun(
            dataset_id=payload.dataset_id,
            model_version_id=payload.model_version_id,
            created_by=payload.created_by,
            status=ForecastRunStatus.RUNNING,
            horizon=payload.horizon,
            started_at=pd.Timestamp.utcnow(),
        )
        self.session.add(forecast_run)
        self.session.flush()

        points = self._generate_forecasts(run_id=forecast_run.id, payload=payload)

        forecast_run.status = ForecastRunStatus.COMPLETED
        forecast_run.finished_at = pd.Timestamp.utcnow()
        self.session.commit()
        self.session.refresh(forecast_run)

        return ForecastRunResponse(
            forecast_run_id=forecast_run.id,
            status=forecast_run.status.value,
            started_at=forecast_run.started_at,
            finished_at=forecast_run.finished_at,
            forecasts=points,
        )

    def get_forecast(self, forecast_run_id: int) -> ForecastRunResponse:
        run = self.session.get(ForecastRun, forecast_run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast run not found.")

        forecast_rows = self.session.exec(
            select(Forecast)
            .where(Forecast.forecast_run_id == forecast_run_id)
            .order_by(Forecast.store_id, Forecast.date)
        ).all()

        return ForecastRunResponse(
            forecast_run_id=run.id,
            status=run.status.value,
            started_at=run.started_at,
            finished_at=run.finished_at,
            forecasts=[
                ForecastPointResponse(
                    store_id=row.store_id,
                    date=row.date,
                    predicted_sales=row.predicted_sales,
                )
                for row in forecast_rows
            ],
        )

    def get_run_status(self, forecast_run_id: int) -> ForecastRunStatusResponse:
        run = self.session.get(ForecastRun, forecast_run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast run not found.")

        return ForecastRunStatusResponse(
            forecast_run_id=run.id,
            status=run.status.value,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )

    def _generate_forecasts(
        self,
        *,
        run_id: int,
        payload: ForecastRunRequest,
    ) -> list[ForecastPointResponse]:
        stores = self.session.exec(select(Store)).all()
        if not stores:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No stores available. Upload sales data first.",
            )

        start_date = payload.start_date or (date.today() + timedelta(days=1))
        points: list[ForecastPointResponse] = []

        for store in stores:
            history = self.session.exec(
                select(SalesRecord)
                .where(SalesRecord.store_id == store.id)
                .order_by(SalesRecord.date.desc())
            ).all()

            if history:
                mean_sales = float(pd.Series([row.sales for row in history]).tail(30).mean())
            else:
                mean_sales = 0.0

            for day_idx in range(payload.horizon):
                day = start_date + timedelta(days=day_idx)
                weekly_seasonality = 1.08 if day.weekday() in (4, 5) else 1.0
                trend = 1.0 + min(0.05, day_idx * 0.003)
                predicted = round(max(0.0, mean_sales * weekly_seasonality * trend), 2)

                row = Forecast(
                    forecast_run_id=run_id,
                    store_id=store.id,
                    date=day,
                    predicted_sales=predicted,
                )
                self.session.add(row)
                points.append(
                    ForecastPointResponse(
                        store_id=store.id,
                        date=day,
                        predicted_sales=predicted,
                    )
                )

        return points
