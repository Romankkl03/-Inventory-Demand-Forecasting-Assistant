import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmspe(y_true, y_pred):
    """
    Compute Root Mean Squared Percentage Error (RMSPE).

    Observations with ``y_true <= 0`` are excluded because relative error is
    undefined at zero sales.

    Args:
        y_true (array-like): Ground-truth sales values.
        y_pred (array-like): Predicted sales values.

    Returns:
        float: RMSPE over positive ground-truth observations.
    """
    mask = y_true > 0

    return np.sqrt(
        np.mean(
            ((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2
        )
    )


def mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Observations with ``y_true <= 0`` are excluded because relative error is
    undefined at zero sales.

    Args:
        y_true (array-like): Ground-truth sales values.
        y_pred (array-like): Predicted sales values.

    Returns:
        float: MAPE over positive ground-truth observations.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0

    if not np.any(mask):
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def segment_error_metrics(y_true, y_pred):
    """
    Compute error metrics for a single segment in the original sales scale.

    Args:
        y_true (array-like): Ground-truth sales values.
        y_pred (array-like): Predicted sales values.

    Returns:
        dict: Segment metrics including ``n``, ``MAE``, ``RMSE``, ``MAPE``,
            ``RMSPE``, ``bias``, and ``underprediction_share``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "RMSPE": np.nan,
            "bias": np.nan,
            "underprediction_share": np.nan,
        }

    error = y_pred - y_true

    return {
        "n": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
        "RMSPE": rmspe(y_true, y_pred),
        "bias": float(np.mean(error)),
        "underprediction_share": float(np.mean(y_pred < y_true)),
    }


def error_analysis_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """
    Aggregate error metrics for each value of a segment column.

    Args:
        df (pd.DataFrame): Table with segment labels and sales predictions.
        segment_col (str): Column used to split observations into segments.
        y_true_col (str): Ground-truth sales column. Defaults to ``"y_true"``.
        y_pred_col (str): Predicted sales column. Defaults to ``"y_pred"``.

    Returns:
        pd.DataFrame: One row per segment with error metrics.
    """
    rows = []
    for segment_value, group in df.groupby(segment_col, dropna=False):
        metrics = segment_error_metrics(group[y_true_col], group[y_pred_col])
        metrics["segment"] = segment_col
        metrics["segment_value"] = segment_value
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("segment_value").reset_index(drop=True)


def evaluate_model(model_name, y_true_log, y_pred_log):
    """
    Evaluate regression predictions in the original sales scale.

    Converts log-transformed targets and predictions back with ``expm1``,
    clips negative predictions to zero, and computes standard regression
    metrics used in the Rossmann baseline.

    Args:
        model_name (str): Human-readable model identifier for the result dict.
        y_true_log (array-like): Ground-truth values in ``log1p(Sales)`` scale.
        y_pred_log (array-like): Predicted values in ``log1p(Sales)`` scale.

    Returns:
        dict: Dictionary with keys ``"model"``, ``"MAE"``, ``"RMSE"``,
            ``"RMSPE"``, and ``"R2"``.
    """
    y_true_sales = np.expm1(y_true_log)
    y_pred_sales = np.expm1(y_pred_log)

    y_pred_sales = np.clip(y_pred_sales, 0, None)

    mae = mean_absolute_error(y_true_sales, y_pred_sales)
    rmse = np.sqrt(mean_squared_error(y_true_sales, y_pred_sales))
    rmspe_value = rmspe(y_true_sales, y_pred_sales)
    r2 = r2_score(y_true_sales, y_pred_sales)

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "RMSPE": rmspe_value,
        "R2": r2,
    }
