"""Forecasting service: create runs, inference, status and outputs."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.forecasting.data.reader import DataReader
from app.forecasting.data.spliter import TemporalSpliter
from app.forecasting.postprocessing import apply_sales_uplift_to_log_predictions
from app.forecasting.preprocessing.module import PreprocessingModule
from app.models import (
    Dataset,
    DatasetStatus,
    Forecast,
    ForecastRun,
    ForecastRunStatus,
    ModelType,
    ModelVersion,
    SalesRecord,
    Store,
    User,
    UserRole,
)
from app.schemas import (
    ForecastPointResponse,
    ForecastRunRequest,
    ForecastRunResponse,
    ForecastRunStatusResponse,
    RandomValForecastRequest,
    RandomValForecastResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "hgb_full.joblib"


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

    def run_random_val_inference(
        self,
        payload: RandomValForecastRequest,
    ) -> RandomValForecastResponse:
        user = self.session.get(User, payload.created_by)
        if user is None:
            user = self.session.exec(select(User).order_by(User.id)).first()
        if user is None:
            user = User(name="System Admin", email="admin@example.com", role=UserRole.ADMIN)
            self.session.add(user)
            self.session.flush()

        model_version = self.session.get(ModelVersion, payload.model_version_id)
        if model_version is None:
            model_version = self.session.exec(select(ModelVersion).order_by(ModelVersion.id)).first()
        if model_version is None:
            model_version = ModelVersion(
                name="baseline-demand-model",
                version="1.0.0",
                model_type=ModelType.BASELINE,
                features_version="baseline-v1",
                metrics_json={"mae": 0.0, "note": "auto-created for inference"},
            )
            self.session.add(model_version)
            self.session.flush()

        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found: {MODEL_PATH}",
            )

        try:
            raw_data = DataReader().read()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        split = TemporalSpliter(train_ratio=0.8, val_ratio=0.2).split(raw_data["train"])
        if split.val is None or split.val.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Validation split is empty.",
            )

        preprocessor = PreprocessingModule(use_log=True)
        preprocessor.fit(split.train, raw_data["store"])
        val_df = preprocessor.transform(split.val, raw_data["store"])
        feature_cols = preprocessor.feature_columns_
        if not feature_cols:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to infer feature columns for inference.",
            )

        model = joblib.load(MODEL_PATH)
        rng = np.random.default_rng(payload.seed)
        val_df = val_df.sort_values(["Store", "Date"]).reset_index(drop=True)
        eligible_stores = (
            val_df.groupby("Store")
            .size()
            .loc[lambda series: series >= payload.horizon]
            .index.to_list()
        )
        if not eligible_stores:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No validation store has enough rows for requested horizon.",
            )

        store_external_id = str(int(rng.choice(eligible_stores)))
        store_rows = val_df[val_df["Store"] == int(store_external_id)].sort_values("Date").reset_index(drop=True)
        max_start_idx = len(store_rows) - payload.horizon
        start_idx = int(rng.integers(0, max_start_idx + 1))
        run_rows = store_rows.iloc[start_idx : start_idx + payload.horizon].copy()
        x_rows = run_rows[feature_cols]

        pred_log = model.predict(x_rows)
        pred_log_pp, uplift_fraction = apply_sales_uplift_to_log_predictions(pred_log, x_rows)

        pred_sales_raw = np.clip(np.expm1(pred_log), 0, None)
        pred_sales_pp = np.clip(np.expm1(pred_log_pp), 0, None)
        actual_sales = np.clip(np.expm1(run_rows["y"].to_numpy()), 0, None)

        store = self.session.exec(
            select(Store).where(Store.external_id == store_external_id)
        ).first()
        if store is None:
            raw_store_row = raw_data["store"][raw_data["store"]["Store"] == int(store_external_id)]
            store = Store(
                external_id=store_external_id,
                store_type=raw_store_row["StoreType"].iloc[0] if not raw_store_row.empty else None,
                assortment=raw_store_row["Assortment"].iloc[0] if not raw_store_row.empty else None,
                competition_distance=(
                    float(raw_store_row["CompetitionDistance"].iloc[0])
                    if not raw_store_row.empty and pd.notna(raw_store_row["CompetitionDistance"].iloc[0])
                    else None
                ),
            )
            self.session.add(store)
            self.session.flush()

        dataset = self.session.exec(select(Dataset).order_by(Dataset.id)).first()
        if dataset is None:
            dataset = Dataset(
                name="rossmann-val-random",
                source="data/raw/rossmann/train.csv",
                uploaded_by=user.id,
                status=DatasetStatus.READY,
            )
            self.session.add(dataset)
            self.session.flush()

        forecast_run = ForecastRun(
            dataset_id=dataset.id,
            model_version_id=model_version.id,
            created_by=user.id,
            status=ForecastRunStatus.COMPLETED,
            horizon=payload.horizon,
            started_at=pd.Timestamp.utcnow(),
            finished_at=pd.Timestamp.utcnow(),
        )
        self.session.add(forecast_run)
        self.session.flush()

        for row_idx, row in enumerate(run_rows.itertuples(index=False)):
            forecast_item = Forecast(
                forecast_run_id=forecast_run.id,
                store_id=store.id,
                date=pd.to_datetime(row.Date).date(),
                predicted_sales=round(float(pred_sales_pp[row_idx]), 2),
            )
            self.session.add(forecast_item)
        self.session.commit()

        return RandomValForecastResponse(
            forecast_run_id=forecast_run.id,
            status=forecast_run.status.value,
            store_id=store.id,
            horizon=payload.horizon,
            forecast_start_date=pd.to_datetime(run_rows["Date"].iloc[0]).date(),
            actual_sales_total=round(float(actual_sales.sum()), 2),
            predicted_sales_raw_total=round(float(pred_sales_raw.sum()), 2),
            predicted_sales_postprocessed_total=round(float(pred_sales_pp.sum()), 2),
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
