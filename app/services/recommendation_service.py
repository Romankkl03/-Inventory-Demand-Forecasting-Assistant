"""Recommendation service: rule-based recommendations and optional LLM text."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.database.config import get_settings
from app.models import Forecast, ForecastRun, Recommendation, RiskLevel
from app.schemas import RecommendationItemResponse, RecommendationsResponse
from app.services.recommendation_engine import VLLMWriter, aggregate_forecast, build_recommendation


class RecommendationService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()

    def get_or_create_recommendations(
        self,
        forecast_run_id: int,
        *,
        include_llm: bool = False,
    ) -> RecommendationsResponse:
        run = self.session.get(ForecastRun, forecast_run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast run not found.")

        existing = self.session.exec(
            select(Recommendation).where(Recommendation.forecast_run_id == forecast_run_id)
        ).all()
        if not existing:
            existing = self._create_recommendations(forecast_run_id=forecast_run_id)

        llm_summary = None
        llm_explanation = None
        supplier_document_draft = None

        if include_llm and existing:
            features, rule_payload = self._build_context_for_llm(forecast_run_id, existing[0].store_id)
            llm_text = self._generate_llm_text(existing[0].store_id, features, rule_payload)
            if llm_text is not None:
                llm_summary = llm_text.summary
                llm_explanation = llm_text.explanation
                supplier_document_draft = llm_text.supplier_document_draft

        return RecommendationsResponse(
            forecast_run_id=forecast_run_id,
            recommendations=[
                RecommendationItemResponse(
                    store_id=item.store_id,
                    expected_demand=item.expected_demand,
                    recommended_order=item.recommended_order,
                    risk_level=item.risk_level.value,
                    comment=item.comment,
                )
                for item in existing
            ],
            llm_summary=llm_summary,
            llm_explanation=llm_explanation,
            supplier_document_draft=supplier_document_draft,
        )

    def _create_recommendations(self, *, forecast_run_id: int) -> list[Recommendation]:
        forecast_rows = self.session.exec(
            select(Forecast)
            .where(Forecast.forecast_run_id == forecast_run_id)
            .order_by(Forecast.store_id, Forecast.date)
        ).all()
        if not forecast_rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Forecasts not found for this run.",
            )

        grouped: dict[int, list[Forecast]] = {}
        for row in forecast_rows:
            grouped.setdefault(row.store_id, []).append(row)

        created: list[Recommendation] = []
        for store_id, rows in grouped.items():
            df = pd.DataFrame(
                [{"date": row.date.isoformat(), "predicted_sales": row.predicted_sales} for row in rows]
            )
            features = aggregate_forecast(df)
            rule = build_recommendation(features)

            recommendation = Recommendation(
                forecast_run_id=forecast_run_id,
                store_id=store_id,
                expected_demand=round(rule.expected_demand, 2),
                recommended_order=round(rule.recommended_order, 2),
                risk_level=RiskLevel(rule.risk_level.value),
                comment=json.dumps({"reason_flags": rule.reason_flags}),
            )
            self.session.add(recommendation)
            created.append(recommendation)

        self.session.commit()
        for item in created:
            self.session.refresh(item)
        return created

    def _build_context_for_llm(self, forecast_run_id: int, store_id: int):
        rows = self.session.exec(
            select(Forecast)
            .where(Forecast.forecast_run_id == forecast_run_id, Forecast.store_id == store_id)
            .order_by(Forecast.date)
        ).all()
        df = pd.DataFrame(
            [{"date": row.date.isoformat(), "predicted_sales": row.predicted_sales} for row in rows]
        )
        features = aggregate_forecast(df)
        rule = build_recommendation(features)
        return features, rule

    def _generate_llm_text(self, store_id: int, features, rule):
        if not self.settings.VLLM_BASE_URL:
            return None
        writer = VLLMWriter(
            base_url=self.settings.VLLM_BASE_URL,
            model=self.settings.VLLM_MODEL,
            api_key=self.settings.VLLM_API_KEY,
            timeout_sec=self.settings.VLLM_TIMEOUT_SEC,
        )
        try:
            return writer.generate_text(store_id=store_id, features=features, recommendation=rule)
        except RuntimeError:
            return None
