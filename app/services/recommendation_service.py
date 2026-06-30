"""Recommendation service: rule-based recommendations and optional LLM text."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.database.config import get_settings
from app.models import Forecast, ForecastRun, Recommendation, RiskLevel, SalesRecord
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
            recommendations=[self._to_action_row(item) for item in existing],
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
            rows = sorted(rows, key=lambda item: item.date)
            df = pd.DataFrame(
                [{"date": row.date.isoformat(), "predicted_sales": row.predicted_sales} for row in rows]
            )
            previous_period_total = self._previous_period_total(
                store_id=store_id,
                horizon=len(rows),
                before_date=rows[0].date,
            )
            features = aggregate_forecast(df, previous_period_total=previous_period_total)
            rule = build_recommendation(features)

            recommendation = Recommendation(
                forecast_run_id=forecast_run_id,
                store_id=store_id,
                expected_demand=round(rule.expected_demand, 2),
                recommended_order=round(rule.recommended_order, 2),
                risk_level=RiskLevel(rule.risk_level.value),
                comment=json.dumps(
                    {
                        "reason_flags": rule.reason_flags,
                        "baseline_total": previous_period_total,
                        "forecast_total": features.total_demand,
                    }
                ),
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

    def _previous_period_total(self, *, store_id: int, horizon: int, before_date) -> float:
        history = self.session.exec(
            select(SalesRecord)
            .where(SalesRecord.store_id == store_id, SalesRecord.date < before_date)
            .order_by(SalesRecord.date.desc())
        ).all()
        if not history:
            return 0.0
        selected = history[:horizon]
        return float(sum(item.sales for item in selected))

    def _to_action_row(self, recommendation: Recommendation) -> RecommendationItemResponse:
        meta = self._parse_comment_meta(recommendation.comment)
        baseline = float(meta.get("baseline_total", 0.0))
        forecast_total = float(meta.get("forecast_total", recommendation.expected_demand))
        reason_flags = meta.get("reason_flags", [])

        demand_vs_baseline_pct = 0.0
        if baseline > 0:
            demand_vs_baseline_pct = (forecast_total - baseline) / baseline
        demand_vs_baseline = f"{demand_vs_baseline_pct:+.0%} vs usual"

        reason_tags = self._reason_tags(reason_flags)
        status_label = self._status_label(
            recommended_order=recommendation.recommended_order,
            demand_vs_baseline_pct=demand_vs_baseline_pct,
            reason_tags=reason_tags,
        )
        priority = self._priority_label(
            risk_level=recommendation.risk_level,
            demand_vs_baseline_pct=demand_vs_baseline_pct,
            reason_tags=reason_tags,
            status_label=status_label,
        )
        reason = self._human_reason(reason_tags)
        action = self._action_label(
            status_label=status_label,
            priority=priority,
            demand_vs_baseline_pct=demand_vs_baseline_pct,
            reason_tags=reason_tags,
        )

        return RecommendationItemResponse(
            store_id=recommendation.store_id,
            status=status_label,
            expected_demand=recommendation.expected_demand,
            demand_vs_baseline=demand_vs_baseline,
            demand_vs_baseline_pct=round(demand_vs_baseline_pct, 4),
            recommended_order=recommendation.recommended_order,
            priority=priority,
            reason=reason,
            reason_tags=reason_tags,
            action=action,
            risk_level=recommendation.risk_level.value,
            comment=recommendation.comment,
        )

    @staticmethod
    def _parse_comment_meta(comment: str | None) -> dict:
        if not comment:
            return {}
        try:
            parsed = json.loads(comment)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _status_label(
        recommended_order: float,
        demand_vs_baseline_pct: float,
        reason_tags: list[str],
    ) -> str:
        if "Low inventory cover" in reason_tags and recommended_order > 0:
            return "Increase order"
        if recommended_order > 0 and demand_vs_baseline_pct >= 0.05:
            return "Increase order"
        if demand_vs_baseline_pct <= -0.10:
            return "Reduce order"
        return "Maintain order"

    @staticmethod
    def _priority_label(
        risk_level: RiskLevel,
        demand_vs_baseline_pct: float,
        reason_tags: list[str],
        status_label: str,
    ) -> str:
        if status_label == "Maintain order" and risk_level != RiskLevel.HIGH:
            return "Low"
        if "Low inventory cover" in reason_tags and status_label == "Increase order":
            return "High"
        if risk_level == RiskLevel.HIGH or demand_vs_baseline_pct >= 0.20:
            return "High"
        if risk_level == RiskLevel.MEDIUM or demand_vs_baseline_pct >= 0.08:
            return "Medium"
        return "Low"

    @staticmethod
    def _reason_tags(flags: list[str]) -> list[str]:
        mapping = {
            "high_demand_regime": "High recent demand",
            "sharp_growth_vs_previous_period": "Demand growth vs previous period",
            "demand_spike_detected": "Demand spike",
            "high_forecast_volatility": "High forecast volatility",
            "low_inventory_cover": "Low inventory cover",
            "sufficient_inventory": "Sufficient inventory",
        }
        normalized = [mapping.get(item, item.replace("_", " ").title()) for item in flags]
        seen: set[str] = set()
        deduplicated: list[str] = []
        for item in normalized:
            if item in seen:
                continue
            seen.add(item)
            deduplicated.append(item)
        return deduplicated

    @staticmethod
    def _human_reason(reason_tags: list[str]) -> str:
        if not reason_tags:
            return "No critical drivers detected"
        return ", ".join(reason_tags)

    @staticmethod
    def _action_label(
        status_label: str,
        priority: str,
        demand_vs_baseline_pct: float,
        reason_tags: list[str],
    ) -> str:
        if status_label == "Increase order":
            if priority == "High":
                return "Urgent supplier reorder"
            if demand_vs_baseline_pct >= 0.12:
                percent = int(round(demand_vs_baseline_pct * 100))
                return f"Increase next order by {percent}%"
            return "Place replenishment order"
        if status_label == "Reduce order":
            return "Review manually"
        if "Sufficient inventory" in reason_tags:
            return "Keep current replenishment level"
        return "Review manually"
