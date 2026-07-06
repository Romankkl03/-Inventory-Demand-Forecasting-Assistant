"""Deterministic recommendation rules based on aggregated forecast features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.models.entities import RiskLevel
from app.services.recommendation_engine.aggregator import AggregatedForecastFeatures


@dataclass(slots=True)
class RuleBasedRecommendation:
    """Recommendation output produced by business rules."""

    expected_demand: float
    recommended_order: float
    risk_level: RiskLevel
    reason_flags: list[str]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["risk_level"] = self.risk_level.value
        return payload


def build_recommendation(
    features: AggregatedForecastFeatures,
    *,
    current_stock: float = 0.0,
    in_transit_stock: float = 0.0,
    lead_time_days: int = 7,
) -> RuleBasedRecommendation:
    """Compute recommended order, risk level, and reason flags."""

    lead_time_days = max(1, lead_time_days)
    horizon_days = max(1, features.horizon_days)
    lead_time_ratio = min(1.0, lead_time_days / horizon_days)

    demand_for_lead_time = features.expected_demand * lead_time_ratio
    safety_stock = features.avg_daily_demand * lead_time_days * min(
        0.60, 0.15 + features.forecast_volatility
    )
    inventory_position = max(0.0, current_stock + in_transit_stock)

    recommended_order = max(0.0, demand_for_lead_time + safety_stock - inventory_position)

    risk_score = 0
    reason_flags: list[str] = []

    if features.high_demand_regime:
        risk_score += 2
        reason_flags.append("high_demand_regime")

    if features.change_vs_previous_period >= 0.20:
        risk_score += 1
        reason_flags.append("sharp_growth_vs_previous_period")
    elif features.change_vs_previous_period <= -0.15:
        reason_flags.append("demand_decline_vs_previous_period")

    if features.demand_spike_index >= 1.60:
        risk_score += 1
        reason_flags.append("demand_spike_detected")

    if features.forecast_volatility >= 0.35:
        risk_score += 1
        reason_flags.append("high_forecast_volatility")

    if inventory_position <= features.avg_daily_demand * max(2, lead_time_days // 2):
        risk_score += 1
        reason_flags.append("low_inventory_cover")

    if risk_score >= 4:
        risk_level = RiskLevel.HIGH
    elif risk_score >= 2:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW

    if recommended_order <= 0:
        reason_flags.append("sufficient_inventory")

    return RuleBasedRecommendation(
        expected_demand=features.expected_demand,
        recommended_order=float(round(recommended_order, 2)),
        risk_level=risk_level,
        reason_flags=reason_flags,
    )
