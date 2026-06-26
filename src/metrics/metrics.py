import numpy as np
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
