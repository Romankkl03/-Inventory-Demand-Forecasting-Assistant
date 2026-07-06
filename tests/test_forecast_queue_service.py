from __future__ import annotations

from sqlmodel import Session

from app.models import ForecastRun, ForecastRunStatus
from app.schemas import ForecastRunRequest
from app.services.forecasting_service import ForecastingService


def test_enqueue_forecast_marks_run_as_queued(
    session: Session,
    seeded_forecast_input: dict[str, int],
) -> None:
    payload = ForecastRunRequest(
        dataset_id=seeded_forecast_input["dataset_id"],
        model_version_id=seeded_forecast_input["model_version_id"],
        created_by=seeded_forecast_input["user_id"],
        horizon=5,
    )

    response = ForecastingService(session).enqueue_forecast(payload)
    saved_run = session.get(ForecastRun, response.forecast_run_id)

    assert response.status == ForecastRunStatus.QUEUED.value
    assert response.forecasts == []
    assert saved_run is not None
    assert saved_run.status == ForecastRunStatus.QUEUED
    assert saved_run.started_at is None
    assert saved_run.finished_at is None


def test_worker_claim_and_execute_changes_status_and_writes_forecasts(
    session: Session,
    seeded_forecast_input: dict[str, int],
) -> None:
    service = ForecastingService(session)
    payload = ForecastRunRequest(
        dataset_id=seeded_forecast_input["dataset_id"],
        model_version_id=seeded_forecast_input["model_version_id"],
        created_by=seeded_forecast_input["user_id"],
        horizon=4,
    )
    queued = service.enqueue_forecast(payload)

    claimed_run_id = service.claim_next_queued_run()
    assert claimed_run_id == queued.forecast_run_id

    executed = service.execute_run(claimed_run_id)
    saved_run = session.get(ForecastRun, queued.forecast_run_id)

    assert executed.status == ForecastRunStatus.COMPLETED.value
    assert len(executed.forecasts) == payload.horizon
    assert saved_run is not None
    assert saved_run.status == ForecastRunStatus.COMPLETED
    assert saved_run.started_at is not None
    assert saved_run.finished_at is not None
