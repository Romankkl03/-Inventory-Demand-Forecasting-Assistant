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


class RandomValForecastRequest(BaseModel):
    created_by: int
    model_version_id: int = 1
    seed: int | None = None
    horizon: int = Field(default=14, ge=1, le=90)


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


class RandomValForecastResponse(BaseModel):
    forecast_run_id: int
    status: str
    store_id: int
    horizon: int
    forecast_start_date: date
    actual_sales_total: float
    predicted_sales_raw_total: float
    predicted_sales_postprocessed_total: float


class ForecastRunStatusResponse(BaseModel):
    forecast_run_id: int
    status: str
    started_at: datetime | None
    finished_at: datetime | None


class RecommendationItemResponse(BaseModel):
    store_id: int
    store_external_id: str | None = None
    status: str
    expected_demand: float
    demand_vs_baseline: str
    demand_vs_baseline_pct: float
    recommended_order: float
    priority: str
    reason: str
    reason_tags: list[str]
    action: str
    risk_level: str
    comment: str | None = None


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
    executive_summary: dict
    kpis: dict
    main_insights: list[str]
    store_level_actions: list[str]
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
