"""Queue worker for forecast runs.

Run:
    uv run python -m app.workers.forecast_worker
"""

from __future__ import annotations

import logging
import os
import time

from sqlmodel import Session

from app.database.database import engine
from app.services.forecasting_service import ForecastingService

logger = logging.getLogger("forecast-worker")


def _poll_interval_sec() -> float:
    raw = os.getenv("FORECAST_WORKER_POLL_SEC", "1.0")
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 1.0


def run_forecast_worker_forever() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    interval = _poll_interval_sec()
    logger.info("Forecast worker started (poll_interval_sec=%s)", interval)

    while True:
        run_id: int | None = None
        try:
            with Session(engine) as claim_session:
                run_id = ForecastingService(claim_session).claim_next_queued_run()

            if run_id is None:
                time.sleep(interval)
                continue

            logger.info("Picked queued run id=%s", run_id)
            with Session(engine) as exec_session:
                ForecastingService(exec_session).execute_run(run_id)
            logger.info("Completed run id=%s", run_id)
        except Exception as exc:  # pragma: no cover - defensive worker guard
            logger.exception("Worker error on run id=%s: %s", run_id, exc)
            time.sleep(interval)


if __name__ == "__main__":
    run_forecast_worker_forever()
