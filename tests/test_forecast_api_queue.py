from __future__ import annotations

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import ForecastRun, ForecastRunStatus


def test_post_forecast_run_returns_202_and_queued_status(
    client: TestClient,
    session: Session,
    seeded_forecast_input: dict[str, int],
) -> None:
    payload = {
        "dataset_id": seeded_forecast_input["dataset_id"],
        "model_version_id": seeded_forecast_input["model_version_id"],
        "created_by": seeded_forecast_input["user_id"],
        "horizon": 7,
    }

    response = client.post("/forecast/run", json=payload)
    body = response.json()
    run_id = body["forecast_run_id"]
    persisted = session.get(ForecastRun, run_id)

    assert response.status_code == 202
    assert body["status"] == ForecastRunStatus.QUEUED.value
    assert body["forecasts"] == []
    assert persisted is not None
    assert persisted.status == ForecastRunStatus.QUEUED

    status_response = client.get(f"/forecast/run/{run_id}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == ForecastRunStatus.QUEUED.value
