"""Report service: build and fetch report summaries."""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.database.config import get_settings
from app.models import ForecastRun, Recommendation, Report, User
from app.schemas import ReportResponse
from app.services.recommendation_service import RecommendationService
from app.services.recommendation_engine import VLLMWriter


class ReportService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()

    def get_or_create_report(self, forecast_run_id: int, created_by: int = 1) -> ReportResponse:
        run = self.session.get(ForecastRun, forecast_run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast run not found.")
        if self.session.get(User, created_by) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        recommendation_rows = RecommendationService(self.session).get_or_create_recommendations(
            forecast_run_id
        ).recommendations

        report = self.session.exec(
            select(Report).where(Report.forecast_run_id == forecast_run_id)
        ).first()
        if report is None:
            db_recommendations = self.session.exec(
                select(Recommendation).where(Recommendation.forecast_run_id == forecast_run_id)
            ).all()
            if not db_recommendations:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No recommendations found for this run.",
                )

            summary = self._build_management_summary(
                forecast_run_id=forecast_run_id,
                recommendation_rows=recommendation_rows,
            )

            report = Report(
                forecast_run_id=forecast_run_id,
                created_by=created_by,
                summary=summary,
            )
            self.session.add(report)
            self.session.commit()
            self.session.refresh(report)

        return ReportResponse(
            forecast_run_id=forecast_run_id,
            report_id=report.id,
            summary=report.summary,
            executive_summary=self._build_executive_summary(recommendation_rows),
            kpis=self._build_kpis(recommendation_rows),
            main_insights=self._build_main_insights(recommendation_rows),
            store_level_actions=self._build_store_actions(recommendation_rows),
            created_at=report.created_at,
        )

    def _build_management_summary(self, *, forecast_run_id: int, recommendation_rows) -> str:
        total_expected = sum(item.expected_demand for item in recommendation_rows)
        total_order = sum(item.recommended_order for item in recommendation_rows)
        high_risk = sum(1 for item in recommendation_rows if item.risk_level == "high")

        avg_delta_pct = (
            sum(item.demand_vs_baseline_pct for item in recommendation_rows) / max(1, len(recommendation_rows))
            if recommendation_rows
            else 0.0
        )

        if avg_delta_pct >= 0.10:
            main_conclusion = "expected demand is above usual level, replenishment is required"
        elif avg_delta_pct <= -0.10:
            main_conclusion = "expected demand is below usual level, conservative ordering is recommended"
        else:
            main_conclusion = "expected demand is near baseline, targeted replenishment is required"

        insight_1 = (
            "Forecast demand is above baseline"
            if avg_delta_pct > 0.01
            else "Forecast demand is close to baseline"
            if -0.01 <= avg_delta_pct <= 0.01
            else "Forecast demand is below baseline"
        )
        insight_2 = self._dominant_reason(recommendation_rows)
        insight_3 = (
            "No critical stockout risk detected" if high_risk == 0 else "Critical risk stores require urgent action"
        )
        insight_4 = "Place supplier order for the next planning period"

        block3_lines = []
        for item in recommendation_rows:
            store_label = item.store_external_id or str(item.store_id)
            block3_lines.append(
                f"Store {store_label}: {item.status} | Priority: {item.priority} | "
                f"Reason: {item.reason} | Action: {item.action}"
            )

        payload = {
            "forecast_run_id": forecast_run_id,
            "stores_processed": len(recommendation_rows),
            "total_expected_demand": round(total_expected, 2),
            "total_recommended_order": round(total_order, 2),
            "high_risk_stores": high_risk,
            "main_conclusion": main_conclusion,
            "insights": [insight_1, insight_2, insight_3, insight_4],
            "store_actions": [item.__dict__ for item in recommendation_rows],
        }

        llm_summary = self._build_summary_with_llm(payload)
        if llm_summary:
            return llm_summary

        return (
            "Executive Summary\n"
            f"Processed stores: {len(recommendation_rows)}\n"
            f"Total expected demand: {round(total_expected, 2)}\n"
            f"Total recommended order: {round(total_order, 2)}\n"
            f"High-risk stores: {high_risk}\n"
            f"Main conclusion: {main_conclusion}\n\n"
            "Main Insights\n"
            f"• {insight_1}\n"
            f"• {insight_2}\n"
            f"• {insight_3}\n"
            f"• {insight_4}\n\n"
            "Store-level Actions\n"
            + "\n".join(block3_lines)
        )

    @staticmethod
    def _dominant_reason(recommendation_rows) -> str:
        counters: dict[str, int] = {}
        for item in recommendation_rows:
            for token in item.reason.split(","):
                key = token.strip()
                if key:
                    counters[key] = counters.get(key, 0) + 1
        if not counters:
            return "Доминирующий фактор рекомендации не выявлен"
        reason = max(counters, key=counters.get)
        return f"Основной фактор рекомендации: {reason}"

    @staticmethod
    def _build_kpis(recommendation_rows) -> dict:
        total_expected = sum(item.expected_demand for item in recommendation_rows)
        total_order = sum(item.recommended_order for item in recommendation_rows)
        avg_delta_pct = (
            sum(item.demand_vs_baseline_pct for item in recommendation_rows) / max(1, len(recommendation_rows))
            if recommendation_rows
            else 0.0
        )
        priority_order = {"Low": 1, "Medium": 2, "High": 3}
        top_priority = max(
            (item.priority for item in recommendation_rows),
            key=lambda value: priority_order.get(value, 0),
            default="Low",
        )
        stores_requiring_action = sum(
            1
            for item in recommendation_rows
            if item.status in {"Increase order", "Reduce order"} or item.priority in {"Medium", "High"}
        )
        return {
            "expected_demand": round(total_expected, 2),
            "recommended_order": round(total_order, 2),
            "priority": top_priority,
            "demand_vs_usual": f"{avg_delta_pct:+.1%}",
            "stores_requiring_action": stores_requiring_action,
        }

    def _build_executive_summary(self, recommendation_rows) -> dict:
        total_expected = round(sum(item.expected_demand for item in recommendation_rows), 2)
        total_order = round(sum(item.recommended_order for item in recommendation_rows), 2)
        high_risk = sum(1 for item in recommendation_rows if item.priority == "High")
        avg_delta_pct = (
            sum(item.demand_vs_baseline_pct for item in recommendation_rows) / max(1, len(recommendation_rows))
            if recommendation_rows
            else 0.0
        )
        if avg_delta_pct > 0.10:
            conclusion = "Ожидается повышенный спрос, требуется оперативное пополнение."
        elif avg_delta_pct < -0.10:
            conclusion = "Ожидается сниженный спрос, заказ можно сократить."
        else:
            conclusion = "Спрос близок к обычному уровню, требуется точечное пополнение."
        return {
            "processed_stores": len(recommendation_rows),
            "total_expected_demand": total_expected,
            "total_recommended_order": total_order,
            "high_risk_stores": high_risk,
            "main_conclusion": conclusion,
        }

    def _build_main_insights(self, recommendation_rows) -> list[str]:
        avg_delta_pct = (
            sum(item.demand_vs_baseline_pct for item in recommendation_rows) / max(1, len(recommendation_rows))
            if recommendation_rows
            else 0.0
        )
        insight_1 = (
            "Прогноз спроса выше базового уровня"
            if avg_delta_pct > 0.01
            else "Прогноз спроса ниже базового уровня"
            if avg_delta_pct < -0.01
            else "Прогноз спроса близок к базовому уровню"
        )
        high_risk = sum(1 for item in recommendation_rows if item.priority == "High")
        insight_2 = self._dominant_reason(recommendation_rows)
        insight_3 = (
            "Обнаружен критический риск дефицита, требуется срочное пополнение"
            if high_risk > 0
            else "Критических рисков дефицита не выявлено"
        )
        insight_4 = "Рекомендуется оформить заказ поставщику на следующий период"
        return [insight_1, insight_2, insight_3, insight_4]

    @staticmethod
    def _build_store_actions(recommendation_rows) -> list[str]:
        actions: list[str] = []
        for item in recommendation_rows:
            store_label = item.store_external_id or str(item.store_id)
            actions.append(
                f"Store {store_label}: {item.status}; priority {item.priority}; "
                f"reason: {item.reason}; action: {item.action}."
            )
        return actions

    def _build_summary_with_llm(self, payload: dict) -> str | None:
        if not self.settings.VLLM_BASE_URL:
            return None
        writer = VLLMWriter(
            base_url=self.settings.VLLM_BASE_URL,
            model=self.settings.VLLM_MODEL,
            api_key=self.settings.VLLM_API_KEY,
            timeout_sec=self.settings.VLLM_TIMEOUT_SEC,
        )
        try:
            report_text = writer.generate_management_summary(payload=payload)
            return report_text.summary
        except RuntimeError:
            return None
