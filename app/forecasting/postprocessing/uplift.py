from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SalesUpliftConfig:
    """Configuration for additive sales uplift corrections."""

    promo_uplift: float = 0.02
    month_end_uplift: float = 0.02
    state_holiday_uplift: float = 0.04
    high_demand_uplift: float = 0.02
    max_total_uplift: float = 0.05
    high_demand_trend_threshold: float = 1.15
    high_demand_lag_ratio_threshold: float = 1.15
    high_demand_rolling_max_ratio_threshold: float = 1.30


DEFAULT_SALES_UPLIFT_CONFIG = SalesUpliftConfig()


def _as_bool_series(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    return np.asarray(values, dtype=bool)


def high_demand_mask(
    features: pd.DataFrame,
    config: SalesUpliftConfig = DEFAULT_SALES_UPLIFT_CONFIG,
) -> np.ndarray:
    """
    Identify high-demand observations using history-based feature proxies.

    A row is treated as high-demand when at least one of the following holds:
    ``trend_7_28`` exceeds its threshold, ``lag_7_to_rolling_mean_28`` exceeds
    its threshold, or ``rolling_max_7 / rolling_mean_28`` exceeds its threshold.
    """
    trend_high = features["trend_7_28"] > config.high_demand_trend_threshold
    lag_ratio_high = (
        features["lag_7_to_rolling_mean_28"] > config.high_demand_lag_ratio_threshold
    )

    rolling_mean_28 = features["rolling_mean_28"].replace(0, np.nan)
    rolling_max_ratio = features["rolling_max_7"] / rolling_mean_28
    rolling_max_high = (
        rolling_max_ratio > config.high_demand_rolling_max_ratio_threshold
    )

    return (trend_high | lag_ratio_high | rolling_max_high.fillna(False)).to_numpy()


def compute_sales_uplift_fraction(
    features: pd.DataFrame,
    config: SalesUpliftConfig = DEFAULT_SALES_UPLIFT_CONFIG,
) -> np.ndarray:
    """
    Compute per-observation additive uplift fractions with a global cap.

    Individual rules add small percentage uplifts:
    promo, month-end, state holiday, and high-demand regime. The total uplift
    for each observation is capped at ``config.max_total_uplift``.
    """
    uplift = np.zeros(len(features), dtype=float)

    promo_mask = _as_bool_series(features["Promo"] == 1)
    month_end_mask = _as_bool_series(features["is_month_end"] == 1)
    state_holiday_mask = _as_bool_series(features["is_state_holiday"] == 1)
    demand_mask = high_demand_mask(features, config)

    uplift[promo_mask] += config.promo_uplift
    uplift[month_end_mask] += config.month_end_uplift
    uplift[state_holiday_mask] += config.state_holiday_uplift
    uplift[demand_mask] += config.high_demand_uplift

    return np.minimum(uplift, config.max_total_uplift)


def apply_sales_uplift(
    y_pred_sales: np.ndarray | pd.Series,
    features: pd.DataFrame,
    config: SalesUpliftConfig = DEFAULT_SALES_UPLIFT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply capped additive uplift corrections in the original sales scale.

    Args:
        y_pred_sales: Predicted sales values (non-log scale).
        features: Feature table with promo, calendar, and history columns.
        config: Uplift rule configuration.

    Returns:
        Tuple of adjusted predictions and per-observation uplift fractions.
    """
    y_pred_sales = np.asarray(y_pred_sales, dtype=float)
    uplift_fraction = compute_sales_uplift_fraction(features, config)
    adjusted = y_pred_sales * (1.0 + uplift_fraction)
    return adjusted, uplift_fraction


def apply_sales_uplift_to_log_predictions(
    y_pred_log: np.ndarray | pd.Series,
    features: pd.DataFrame,
    config: SalesUpliftConfig = DEFAULT_SALES_UPLIFT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert log predictions to sales, apply uplift, and convert back to log scale.
    """
    y_pred_sales = np.clip(np.expm1(y_pred_log), 0, None)
    adjusted_sales, uplift_fraction = apply_sales_uplift(
        y_pred_sales,
        features,
        config,
    )
    return np.log1p(adjusted_sales), uplift_fraction
