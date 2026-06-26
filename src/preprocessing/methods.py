import numpy as np
import pandas as pd


def merge_train_store(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """
    Join transaction rows with store metadata.

    Args:
        train (pd.DataFrame): Transaction table with a ``Store`` column.
        store (pd.DataFrame): Store metadata table keyed by ``Store``.

    Returns:
        pd.DataFrame: Left join of ``train`` and ``store`` on ``Store``.
    """
    return train.merge(store, on="Store", how="left")


def parse_date_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse ``Date`` as datetime and sort by store and date.

    Args:
        df (pd.DataFrame): Input table containing ``Store`` and ``Date``.

    Returns:
        pd.DataFrame: Copy sorted by ``Store`` and ``Date`` with reset index.
    """
    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    return result.sort_values(["Store", "Date"]).reset_index(drop=True)


def filter_open_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where the store was open.

    Args:
        df (pd.DataFrame): Input table containing an ``Open`` column.

    Returns:
        pd.DataFrame: Rows with ``Open == 1`` and reset index.
    """
    return df[df["Open"] == 1].reset_index(drop=True)


def add_competition_missing_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a flag for missing competitor opening date fields.

    Args:
        df (pd.DataFrame): Table with ``CompetitionOpenSinceMonth`` and
            ``CompetitionOpenSinceYear``.

    Returns:
        pd.DataFrame: Copy with boolean column ``Competition_is_missing``.
    """
    result = df.copy()
    result["Competition_is_missing"] = (
        result["CompetitionOpenSinceMonth"].isna()
        | result["CompetitionOpenSinceYear"].isna()
    )
    return result


def get_competition_open_date(df: pd.DataFrame) -> pd.Series:
    """
    Build the first day of the competitor opening month for each row.

    Args:
        df (pd.DataFrame): Table with competition opening month and year.

    Returns:
        pd.Series: Parsed competitor opening dates with ``NaT`` where inputs
            are invalid.
    """
    return pd.to_datetime(
        {
            "year": df["CompetitionOpenSinceYear"],
            "month": df["CompetitionOpenSinceMonth"],
            "day": 1,
        },
        errors="coerce",
    )


def competition_duration_days(df: pd.DataFrame) -> pd.Series:
    """
    Compute the number of days since the nearest competitor opened.

    Missing competition dates yield zero duration.

    Args:
        df (pd.DataFrame): Table with ``Date`` and competition opening fields.

    Returns:
        pd.Series: Non-negative competition duration in days.
    """
    duration = (df["Date"] - get_competition_open_date(df)).dt.days
    duration = duration.where(duration >= 0)
    if "Competition_is_missing" in df.columns:
        missing = df["Competition_is_missing"]
    else:
        missing = (
            df["CompetitionOpenSinceMonth"].isna() | df["CompetitionOpenSinceYear"].isna()
        )
    return duration.where(~missing, 0).fillna(0)


def handle_promo2_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Promo2-related missing values for inactive stores.

    When ``Promo2 == 0``, related fields are explicitly set to ``NA``.

    Args:
        df (pd.DataFrame): Table with Promo2 metadata columns.

    Returns:
        pd.DataFrame: Copy with cleaned Promo2 fields.
    """
    result = df.copy()
    no_promo2 = result["Promo2"] == 0
    for col in ("Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"):
        result.loc[no_promo2, col] = pd.NA
    return result


def annotate_observation_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure calendar gaps between consecutive store observations.

    Args:
        df (pd.DataFrame): Table sorted by ``Store`` and ``Date``.

    Returns:
        pd.DataFrame: Copy with gap annotations and first-observation flags.
    """
    result = df.sort_values(["Store", "Date"]).copy()
    result["prev_observed_date"] = result.groupby("Store")["Date"].shift(1)
    result["days_since_prev_observation"] = (
        result["Date"] - result["prev_observed_date"]
    ).dt.days
    return impute_observation_gaps(result)


def impute_observation_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finalize observation-gap features for modeling.

    Args:
        df (pd.DataFrame): Table with ``days_since_prev_observation``.

    Returns:
        pd.DataFrame: Copy with ``is_first_observation`` and zero-filled gap
            lengths for the first row of each store.
    """
    result = df.copy()
    result["is_first_observation"] = result["days_since_prev_observation"].isna().astype(int)
    result["days_since_prev_observation"] = result["days_since_prev_observation"].fillna(0)
    return result


def get_calendar_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    List missing store-date pairs in the observed calendar range.

    Args:
        df (pd.DataFrame): Table with ``Store`` and ``Date`` columns.

    Returns:
        pd.DataFrame: Store-date combinations absent from ``df``.
    """
    stores = df["Store"].unique()
    dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [stores, dates],
        names=["Store", "Date"],
    )
    observed_index = pd.MultiIndex.from_frame(df[["Store", "Date"]])
    return full_index.difference(observed_index).to_frame(index=False)


_SEASON_BY_MONTH = {
    12: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 3,
    9: 4,
    10: 4,
    11: 4,
}


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Derive calendar-based features from a date column.

    Args:
        df (pd.DataFrame): Input table.
        date_col (str): Name of the date column. Defaults to ``"Date"``.

    Returns:
        pd.DataFrame: Copy with year, month, week, day, weekday, weekend,
            month-boundary, quarter, and season features.
    """
    result = df.copy()
    date = result[date_col]

    result["year"] = date.dt.year
    result["month"] = date.dt.month
    result["week"] = date.dt.isocalendar().week.astype(int)
    result["day"] = date.dt.day
    result["dayofweek"] = date.dt.dayofweek
    result["is_weekend"] = result["dayofweek"] >= 5
    result["is_month_start"] = date.dt.is_month_start
    result["is_month_end"] = date.dt.is_month_end
    result["quarter"] = date.dt.quarter
    result["season"] = result["month"].map(_SEASON_BY_MONTH)
    return result


DEFAULT_LAGS = (1, 7, 14, 28)


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Sales",
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Add calendar-based sales lags for each store.

    Each lag looks up the target value exactly ``lag`` calendar days before the
    current observation for the same store.

    Args:
        df (pd.DataFrame): Table with ``Store``, ``Date``, and ``target_col``.
        target_col (str): Sales column used for lag values. Defaults to
            ``"Sales"``.
        lags (tuple[int, ...]): Calendar lag offsets in days. Defaults to
            ``(1, 7, 14, 28)``.

    Returns:
        pd.DataFrame: Copy with columns ``lag_{n}`` for each requested lag.
    """
    result = df.sort_values(["Store", "Date"]).copy()
    history = result[["Store", "Date", target_col]].rename(
        columns={"Date": "lag_date", target_col: "lag_value"}
    )

    for lag in lags:
        keys = result[["Store", "Date"]].copy()
        keys["lag_date"] = keys["Date"] - pd.Timedelta(days=lag)
        lag_values = keys.merge(history, on=["Store", "lag_date"], how="left")["lag_value"]
        result[f"lag_{lag}"] = lag_values.to_numpy()

    return result.reset_index(drop=True)


DEFAULT_ROLLING_WINDOWS = (7, 14, 28)


def _rolling_min_periods(window: int, min_periods: int | None) -> int:
    """
    Resolve the minimum number of observations for a rolling window.

    Args:
        window (int): Rolling window size in days.
        min_periods (int | None): Explicit minimum periods. If ``None``, uses
            ``max(1, window // 2)``.

    Returns:
        int: Effective ``min_periods`` value.
    """
    if min_periods is not None:
        return min_periods
    return max(1, window // 2)


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Sales",
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Add past-only rolling mean and std of sales for each store.

    The current day is excluded via ``shift(1)`` to avoid target leakage.

    Args:
        df (pd.DataFrame): Table with ``Store``, ``Date``, and ``target_col``.
        target_col (str): Sales column used for rolling statistics. Defaults to
            ``"Sales"``.
        windows (tuple[int, ...]): Rolling window sizes in days. Defaults to
            ``(7, 14, 28)``.
        min_periods (int | None): Minimum observations per rolling window.
            Defaults to half the window size.

    Returns:
        pd.DataFrame: Copy with ``rolling_mean_{w}`` and ``rolling_std_{w}``
            columns for each window.
    """
    result = df.sort_values(["Store", "Date"]).copy()

    for window in windows:
        mp = _rolling_min_periods(window, min_periods)
        past_values = result.groupby("Store")[target_col].shift(1)
        rolling = past_values.groupby(result["Store"]).rolling(window, min_periods=mp)
        result[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        result[f"rolling_std_{window}"] = rolling.std().reset_index(level=0, drop=True)

    return result.reset_index(drop=True)


def impute_lag_features(
    df: pd.DataFrame,
    store_median_sales: pd.Series,
    global_median_sales: float,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Impute missing lag values and add missingness indicators.

    Args:
        df (pd.DataFrame): Table with ``lag_{n}`` columns.
        store_median_sales (pd.Series): Per-store median sales indexed by store
            id.
        global_median_sales (float): Fallback median when a store is unseen.
        lags (tuple[int, ...]): Lag offsets to impute. Defaults to
            ``DEFAULT_LAGS``.

    Returns:
        pd.DataFrame: Copy with ``lag_{n}_is_missing`` flags and imputed lag
            values.
    """
    result = df.copy()
    store_fallback = result["Store"].map(store_median_sales).fillna(global_median_sales)

    for lag in lags:
        col = f"lag_{lag}"
        result[f"{col}_is_missing"] = result[col].isna().astype(int)
        result[col] = result[col].fillna(store_fallback)

    return result


def impute_rolling_features(
    df: pd.DataFrame,
    store_median_sales: pd.Series,
    global_median_sales: float,
    store_median_std: pd.Series,
    global_median_std: float,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Impute missing rolling statistics and add missingness indicators.

    Args:
        df (pd.DataFrame): Table with rolling mean and std columns.
        store_median_sales (pd.Series): Per-store median sales for mean
            imputation.
        global_median_sales (float): Global median fallback for rolling means.
        store_median_std (pd.Series): Per-store sales std for std imputation.
        global_median_std (float): Global std fallback for rolling stds.
        windows (tuple[int, ...]): Rolling window sizes to impute. Defaults to
            ``DEFAULT_ROLLING_WINDOWS``.

    Returns:
        pd.DataFrame: Copy with ``rolling_*_is_missing`` flags and imputed
            rolling values.
    """
    result = df.copy()
    mean_fallback = result["Store"].map(store_median_sales).fillna(global_median_sales)
    std_fallback = result["Store"].map(store_median_std).fillna(global_median_std)

    for window in windows:
        mean_col = f"rolling_mean_{window}"
        std_col = f"rolling_std_{window}"
        result[f"{mean_col}_is_missing"] = result[mean_col].isna().astype(int)
        result[f"{std_col}_is_missing"] = result[std_col].isna().astype(int)
        result[mean_col] = result[mean_col].fillna(mean_fallback)
        result[std_col] = result[std_col].fillna(std_fallback)

    return result


_MONTH_ABBR = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def competition_duration_months(df: pd.DataFrame) -> pd.Series:
    """
    Compute the number of months since the nearest competitor opened.

    Missing competition dates yield zero duration.

    Args:
        df (pd.DataFrame): Table with ``Date`` and competition opening fields.

    Returns:
        pd.Series: Non-negative competition duration in months.
    """
    open_date = get_competition_open_date(df)
    duration = (df["Date"].dt.year - open_date.dt.year) * 12 + (
        df["Date"].dt.month - open_date.dt.month
    )
    duration = duration.where(duration >= 0)
    if "Competition_is_missing" in df.columns:
        missing = df["Competition_is_missing"]
    else:
        missing = (
            df["CompetitionOpenSinceMonth"].isna() | df["CompetitionOpenSinceYear"].isna()
        )
    return duration.where(~missing, 0).fillna(0)


def _promo2_start_date(df: pd.DataFrame) -> pd.Series:
    """
    Parse the Promo2 campaign start date from week and year fields.

    Args:
        df (pd.DataFrame): Table with ``Promo2SinceWeek`` and
            ``Promo2SinceYear``.

    Returns:
        pd.Series: Promo2 start dates with ``NaT`` where parsing fails.
    """
    year = df["Promo2SinceYear"].astype("Int64").astype(str)
    week = df["Promo2SinceWeek"].astype("Int64").astype(str).str.zfill(2)
    return pd.to_datetime(year + "-W" + week + "-1", format="%G-W%V-%u", errors="coerce")


def promo2_duration_weeks(df: pd.DataFrame) -> pd.Series:
    """
    Compute the number of weeks since Promo2 started for active stores.

    Args:
        df (pd.DataFrame): Table with ``Date``, ``Promo2``, and Promo2 start
            fields.

    Returns:
        pd.Series: Non-negative Promo2 duration in weeks, zero when Promo2 is
            inactive.
    """
    start_date = _promo2_start_date(df)
    duration = ((df["Date"] - start_date).dt.days // 7).clip(lower=0)
    duration = duration.where(df["Promo2"] == 1, 0)
    return duration.fillna(0).astype(int)


def _is_promo2_month_active(
    date: pd.Series,
    promo2: pd.Series,
    promo_interval: pd.Series,
) -> pd.Series:
    """
    Check whether Promo2 is active in the current calendar month.

    Args:
        date (pd.Series): Observation dates.
        promo2 (pd.Series): Promo2 participation flag.
        promo_interval (pd.Series): Comma-separated active months for Promo2.

    Returns:
        pd.Series: Boolean mask of active Promo2 months.
    """
    month_abbr = date.dt.month.map(_MONTH_ABBR)

    def _month_in_interval(month: str, interval: object) -> bool:
        if pd.isna(interval):
            return False
        return month in str(interval).split(",")

    is_active_month = pd.Series(
        (_month_in_interval(month, interval) for month, interval in zip(month_abbr, promo_interval)),
        index=date.index,
    )
    return (promo2 == 1) & is_active_month


def add_promo_holiday_store_features(
    df: pd.DataFrame,
    dummy_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add promo, holiday, competition, and categorical store features.

    Encodes ``StoreType``, ``Assortment``, and ``StateHoliday`` with one-hot
    columns and derives Promo2 duration and competition tenure features.

    Args:
        df (pd.DataFrame): Base transformed transaction table.
        dummy_columns (list[str] | None): Fixed one-hot column order from fit.
            Unknown categories are zero-filled when provided.

    Returns:
        pd.DataFrame: Copy concatenated with engineered and encoded features.
    """
    result = df.copy()

    result["Promo"] = result["Promo"].astype(int)
    result["Promo2"] = result["Promo2"].astype(int)
    result["SchoolHoliday"] = result["SchoolHoliday"].astype(int)
    result["promo2_duration_weeks"] = promo2_duration_weeks(result)
    result["is_current_month_in_promo_interval"] = _is_promo2_month_active(
        result["Date"],
        result["Promo2"],
        result["PromoInterval"],
    ).astype(int)
    result["is_state_holiday"] = (result["StateHoliday"] != "0").astype(int)

    store_type_dummies = pd.get_dummies(result["StoreType"], prefix="StoreType", dtype=int)
    assortment_dummies = pd.get_dummies(result["Assortment"], prefix="Assortment", dtype=int)
    state_holiday_dummies = pd.get_dummies(
        result["StateHoliday"],
        prefix="StateHoliday",
        dtype=int,
    ).drop(columns=["StateHoliday_0"], errors="ignore")
    encoded = pd.concat(
        [store_type_dummies, assortment_dummies, state_holiday_dummies],
        axis=1,
    )
    if dummy_columns is not None:
        encoded = encoded.reindex(columns=dummy_columns, fill_value=0)

    result["competition_duration_months"] = competition_duration_months(result)
    result["competition_duration_days"] = competition_duration_days(result)

    return pd.concat([result, encoded], axis=1)


def extract_target(
    df: pd.DataFrame,
    target_col: str = "Sales",
    use_log: bool = False,
) -> pd.Series:
    """
    Extract the modeling target from raw sales.

    Args:
        df (pd.DataFrame): Table containing ``target_col``.
        target_col (str): Name of the sales column. Defaults to ``"Sales"``.
        use_log (bool): If ``True``, apply ``log1p`` transform. Defaults to
            ``False``.

    Returns:
        pd.Series: Target values in original or log scale.
    """
    target = df[target_col].astype(float)
    if use_log:
        return np.log1p(target)
    return target


def inverse_target_transform(
    values: pd.Series | np.ndarray,
    use_log: bool = False,
) -> pd.Series | np.ndarray:
    """
    Convert model predictions back to the original sales scale.

    Args:
        values (pd.Series | np.ndarray): Target or prediction values.
        use_log (bool): If ``True``, apply ``expm1`` inverse transform.
            Defaults to ``False``.

    Returns:
        pd.Series | np.ndarray: Values in the original sales scale. Returns a
            ``pd.Series`` when the input is a Series.
    """
    if use_log:
        transformed = np.expm1(values)
    else:
        transformed = values

    if isinstance(values, pd.Series):
        return pd.Series(transformed, index=values.index)
    return transformed
