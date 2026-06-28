"""Report service: build and fetch report summaries."""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models import ForecastRun, Recommendation, Report, User
from app.schemas import ReportResponse


class ReportService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create_report(self, forecast_run_id: int, created_by: int = 1) -> ReportResponse:
        run = self.session.get(ForecastRun, forecast_run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast run not found.")
        if self.session.get(User, created_by) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

        report = self.session.exec(
            select(Report).where(Report.forecast_run_id == forecast_run_id)
        ).first()
        if report is None:
            recommendations = self.session.exec(
                select(Recommendation).where(Recommendation.forecast_run_id == forecast_run_id)
            ).all()
            if not recommendations:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No recommendations found for this run.",
                )

            total_expected = sum(item.expected_demand for item in recommendations)
            total_order = sum(item.recommended_order for item in recommendations)
            high_risk = sum(1 for item in recommendations if item.risk_level.value == "high")
            summary = (
                f"Run {forecast_run_id}: recommendations={len(recommendations)}, "
                f"total_expected_demand={round(total_expected, 2)}, "
                f"total_recommended_order={round(total_order, 2)}, "
                f"high_risk_stores={high_risk}."
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
            created_at=report.created_at,
        )
