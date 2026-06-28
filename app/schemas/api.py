"""Pydantic request/response schemas for REST API."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class SalesRecordInput(BaseModel):
    store_external_id: str
    date: date
    sales: float = Field(ge=0)
    customers: int = Field(ge=0)
    promo: bool = False
    promo2: bool = False
    school_holiday: bool = False
    state_holiday: str = "0"
    open: bool = True
    store_type: str | None = None
    assortment: str | None = None
    competition_distance: float | None = Field(default=None, ge=0)


class DatasetUploadRequest(BaseModel):
    name: str
    source: str
    uploaded_by: int
    records: list[SalesRecordInput]


class DatasetUploadResponse(BaseModel):
    dataset_id: int
    status: str
    inserted_records: int
    inserted_stores: int


class ForecastRunRequest(BaseModel):
    dataset_id: int
    model_version_id: int
    created_by: int
    horizon: int = Field(ge=1, le=90)
    start_date: date | None = None


class ForecastPointResponse(BaseModel):
    store_id: int
    date: date
    predicted_sales: float


class ForecastRunResponse(BaseModel):
    forecast_run_id: int
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    forecasts: list[ForecastPointResponse]


class ForecastRunStatusResponse(BaseModel):
    forecast_run_id: int
    status: str
    started_at: datetime | None
    finished_at: datetime | None


class RecommendationItemResponse(BaseModel):
    store_id: int
    expected_demand: float
    recommended_order: float
    risk_level: str
    comment: str | None


class RecommendationsResponse(BaseModel):
    forecast_run_id: int
    recommendations: list[RecommendationItemResponse]
    llm_summary: str | None = None
    llm_explanation: str | None = None
    supplier_document_draft: str | None = None


class ReportResponse(BaseModel):
    forecast_run_id: int
    report_id: int
    summary: str
    created_at: datetime


class ModelInfoResponse(BaseModel):
    id: int
    name: str
    version: str
    model_type: str
    features_version: str
    created_at: datetime
    metrics_json: dict


class HealthResponse(BaseModel):
    status: str
    service: str
