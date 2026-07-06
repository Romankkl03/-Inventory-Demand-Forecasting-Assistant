"""Aggregation and feature engineering for recommendation generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


@dataclass(slots=True)
class AggregatedForecastFeatures:
    """Features derived from point forecasts for recommendation logic."""

    horizon_days: int
    total_demand: float
    avg_daily_demand: float
    max_daily_demand: float
    change_vs_previous_period: float
    demand_spike_index: float
    forecast_volatility: float
    expected_demand: float
    high_demand_regime: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_forecast_dataframe(
    forecast: pd.DataFrame | Iterable[dict],
) -> pd.DataFrame:
    if isinstance(forecast, pd.DataFrame):
        df = forecast.copy()
    else:
        df = pd.DataFrame(list(forecast))

    required_columns = {"date", "predicted_sales"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required forecast columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["predicted_sales"] = pd.to_numeric(df["predicted_sales"], errors="coerce")
    df = df.dropna(subset=["date", "predicted_sales"])
    if df.empty:
        raise ValueError("Forecast data is empty after normalization.")

    return df.sort_values("date").reset_index(drop=True)


def aggregate_forecast(
    forecast: pd.DataFrame | Iterable[dict],
    previous_period_total: float | None = None,
) -> AggregatedForecastFeatures:
    """Aggregate daily forecast and prepare features for recommendation rules.

    Args:
        forecast: Point forecast rows with ``date`` and ``predicted_sales``.
        previous_period_total: Historical demand for the same horizon in the
            previous period. If unavailable, change is set to 0.
    """

    df = _normalize_forecast_dataframe(forecast)
    horizon_days = int(df["date"].nunique())

    total_demand = float(df["predicted_sales"].sum())
    avg_daily_demand = float(df["predicted_sales"].mean())
    max_daily_demand = float(df["predicted_sales"].max())

    if previous_period_total is None or previous_period_total <= 0:
        change_vs_previous_period = 0.0
    else:
        change_vs_previous_period = float(
            (total_demand - previous_period_total) / previous_period_total
        )

    demand_spike_index = float(max_daily_demand / avg_daily_demand) if avg_daily_demand > 0 else 0.0
    std = float(df["predicted_sales"].std(ddof=0))
    forecast_volatility = float(std / avg_daily_demand) if avg_daily_demand > 0 else 0.0

    high_demand_regime = bool(
        change_vs_previous_period >= 0.20
        or demand_spike_index >= 1.60
        or forecast_volatility >= 0.35
    )

    risk_buffer = min(0.25, max(0.0, forecast_volatility * 0.5))
    regime_buffer = 0.10 if high_demand_regime else 0.03
    expected_demand = float(total_demand * (1.0 + risk_buffer + regime_buffer))

    return AggregatedForecastFeatures(
        horizon_days=horizon_days,
        total_demand=total_demand,
        avg_daily_demand=avg_daily_demand,
        max_daily_demand=max_daily_demand,
        change_vs_previous_period=change_vs_previous_period,
        demand_spike_index=demand_spike_index,
        forecast_volatility=forecast_volatility,
        expected_demand=expected_demand,
        high_demand_regime=high_demand_regime,
    )
