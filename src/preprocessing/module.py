import numpy as np
import pandas as pd

from .methods import (
    DEFAULT_ROLLING_WINDOWS,
    add_calendar_features,
    add_competition_missing_flag,
    add_holiday_calendar_features,
    add_history_ratio_features,
    add_lag_features,
    add_promo_holiday_store_features,
    add_promo_sequence_features,
    add_rolling_features,
    annotate_observation_gaps,
    extract_target,
    filter_open_days,
    handle_promo2_missing,
    impute_lag_features,
    impute_rolling_features,
    merge_train_store,
    parse_date_and_sort,
)

MODEL_LAGS = (7, 14, 28)

_NON_FEATURE_COLUMNS = {
    "Sales",
    "Customers",
    "y",
    "Id",
    "Date",
    "StoreType",
    "Assortment",
    "StateHoliday",
    "PromoInterval",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "competition_duration_days",
    "prev_observed_date",
    "Open",
    "DayOfWeek",
    "StateHoliday_0",
}

_EXCLUDED_DUMMY_COLUMNS = {"StateHoliday_0"}


class PreprocessingModule:
    """
    Stateful preprocessing pipeline for Rossmann store sales forecasting.

    Fits imputation statistics and sales history on the training split, then
    transforms train, validation, and test data into model-ready features with
    optional ``log1p`` target encoding.

    Attributes:
        use_log (bool): If ``True``, store the target as ``log1p(Sales)`` in
            column ``"y"``.
        target_col (str): Name of the raw sales column. Defaults to ``"Sales"``.
        store_df_ (pd.DataFrame | None): Store metadata saved during ``fit``.
        sales_history_ (pd.DataFrame | None): Historical ``(Store, Date, Sales)``
            table used to build lag and rolling features at inference time.
        dummy_columns_ (list[str] | None): One-hot column order fixed at fit
            time for consistent encoding.
        feature_columns_ (list[str] | None): Model feature names inferred at
            fit time.
        competition_distance_median_ (float | None): Train median used to
            impute missing ``CompetitionDistance``.
        store_median_sales_ (pd.Series | None): Per-store median sales for lag
            and rolling imputation.
        global_median_sales_ (float | None): Global median sales fallback.
        store_median_std_ (pd.Series | None): Per-store sales std for rolling
            imputation.
        global_median_std_ (float | None): Global std fallback for rolling
            features.
    """

    def __init__(self, use_log: bool = False, target_col: str = "Sales"):
        self.use_log = use_log
        self.target_col = target_col
        self.store_df_: pd.DataFrame | None = None
        self.sales_history_: pd.DataFrame | None = None
        self.dummy_columns_: list[str] | None = None
        self.feature_columns_: list[str] | None = None
        self.competition_distance_median_: float | None = None
        self.store_median_sales_: pd.Series | None = None
        self.global_median_sales_: float | None = None
        self.store_median_std_: pd.Series | None = None
        self.global_median_std_: float | None = None

    def fit(self, transactions: pd.DataFrame, store: pd.DataFrame) -> "PreprocessingModule":
        """
        Learn preprocessing state from training transactions.

        Args:
            transactions (pd.DataFrame): Training transaction table.
            store (pd.DataFrame): Store metadata table.

        Returns:
            PreprocessingModule: Fitted preprocessor instance.
        """
        self.store_df_ = store.copy()
        base = self._transform_base(transactions, store)
        self.sales_history_ = base[["Store", "Date", self.target_col]].copy()
        self._fit_sales_statistics(base)
        self.competition_distance_median_ = float(base["CompetitionDistance"].median())

        fitted_sample = self._finalize_features(
            self._impute_history_features(self._add_history_features(base.copy())),
        )
        self.dummy_columns_ = self._get_dummy_columns(fitted_sample)
        self.feature_columns_ = self._infer_feature_columns(fitted_sample)
        return self

    def transform(
        self,
        transactions: pd.DataFrame,
        store: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Apply the fitted preprocessing pipeline to new transactions.

        Args:
            transactions (pd.DataFrame): Transaction table to transform.
            store (pd.DataFrame | None): Optional store metadata override. If
                ``None``, the store table from ``fit`` is reused.

        Returns:
            pd.DataFrame: Processed table with identifier columns, features,
                and target column ``"y"`` when ``target_col`` is present.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self.store_df_ is None:
            raise RuntimeError("PreprocessingModule is not fitted. Call fit first.")

        store_df = store.copy() if store is not None else self.store_df_
        processed = self._transform_base(transactions, store_df)
        processed = self._finalize_features(
            self._impute_history_features(self._add_history_features(processed)),
        )

        if self.target_col in processed.columns:
            processed["y"] = extract_target(processed, self.target_col, self.use_log)

        return processed[self._output_columns(processed)]

    def fit_transform(
        self,
        transactions: pd.DataFrame,
        store: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the same training data.

        Args:
            transactions (pd.DataFrame): Training transaction table.
            store (pd.DataFrame): Store metadata table.

        Returns:
            pd.DataFrame: Processed training table.
        """
        self.fit(transactions, store)
        return self.transform(transactions, store)

    def _fit_sales_statistics(self, df: pd.DataFrame) -> None:
        """
        Compute per-store and global sales statistics for imputation.

        Args:
            df (pd.DataFrame): Base transformed training table.
        """
        self.store_median_sales_ = df.groupby("Store")[self.target_col].median()
        self.global_median_sales_ = float(df[self.target_col].median())
        store_stds = df.groupby("Store")[self.target_col].std().fillna(0)
        self.store_median_std_ = store_stds
        self.global_median_std_ = float(store_stds.median()) if not store_stds.empty else 0.0

    def _impute_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing lag and rolling features using fitted statistics.

        Args:
            df (pd.DataFrame): Table with raw lag and rolling columns.

        Returns:
            pd.DataFrame: Table with imputed history features and missing flags.
        """
        result = impute_lag_features(
            df,
            store_median_sales=self.store_median_sales_,
            global_median_sales=self.global_median_sales_,
            lags=MODEL_LAGS,
        )
        result = impute_rolling_features(
            result,
            store_median_sales=self.store_median_sales_,
            global_median_sales=self.global_median_sales_,
            store_median_std=self.store_median_std_,
            global_median_std=self.global_median_std_,
            windows=DEFAULT_ROLLING_WINDOWS,
        )
        return add_history_ratio_features(result)

    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add promo, holiday, store encodings and final missing-value handling.

        Args:
            df (pd.DataFrame): Table after history feature engineering.

        Returns:
            pd.DataFrame: Fully engineered feature table.
        """
        processed = add_promo_holiday_store_features(
            df,
            dummy_columns=self.dummy_columns_,
        )
        if self.competition_distance_median_ is not None:
            processed["CompetitionDistance"] = processed["CompetitionDistance"].fillna(
                self.competition_distance_median_
            )
        return processed

    def _transform_base(self, transactions: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
        """
        Run base cleaning and calendar feature generation.

        Args:
            transactions (pd.DataFrame): Raw transaction table.
            store (pd.DataFrame): Store metadata table.

        Returns:
            pd.DataFrame: Merged, filtered, and calendar-enriched table without
                lag or rolling features.
        """
        df = merge_train_store(transactions, store)
        df = parse_date_and_sort(df)
        df = add_promo_sequence_features(df)
        df = add_holiday_calendar_features(df)
        df = filter_open_days(df)
        df = add_competition_missing_flag(df)
        df = handle_promo2_missing(df)
        df = annotate_observation_gaps(df)
        return add_calendar_features(df)

    def _add_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build lag and rolling features using fitted sales history.

        Concatenates stored training history with the current batch so that
        validation and test rows can access past sales without leakage from
        future target values.

        Args:
            df (pd.DataFrame): Base transformed table for the current split.

        Returns:
            pd.DataFrame: Input table merged with lag and rolling columns.
        """
        history = self.sales_history_.copy()
        if self.target_col in df.columns:
            current_sales = df[["Store", "Date", self.target_col]].copy()
        else:
            current_sales = df[["Store", "Date"]].copy()
            current_sales[self.target_col] = np.nan

        combined = pd.concat([history, current_sales], ignore_index=True)
        combined = parse_date_and_sort(combined)
        combined = combined.drop_duplicates(subset=["Store", "Date"], keep="last")
        combined[self.target_col] = pd.to_numeric(combined[self.target_col], errors="coerce")
        combined = add_lag_features(combined, target_col=self.target_col, lags=MODEL_LAGS)
        combined = add_rolling_features(combined, target_col=self.target_col)

        feature_cols = ["Store", "Date"] + [
            col
            for col in combined.columns
            if col.startswith(("lag_", "rolling_"))
        ]
        history_features = combined[feature_cols]
        return df.merge(history_features, on=["Store", "Date"], how="left")

    @staticmethod
    def _get_dummy_columns(df: pd.DataFrame) -> list[str]:
        """
        Extract one-hot column names for store and holiday categoricals.

        Args:
            df (pd.DataFrame): Sample table after dummy encoding.

        Returns:
            list[str]: Ordered dummy column names, excluding the dropped
                ``StateHoliday_0`` baseline category.
        """
        return [
            col
            for col in df.columns
            if col.startswith(("StoreType_", "Assortment_", "StateHoliday_"))
            and col not in _EXCLUDED_DUMMY_COLUMNS
        ]

    def _infer_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Infer model feature column names from a processed sample table.

        Args:
            df (pd.DataFrame): Fully processed training sample.

        Returns:
            list[str]: Feature names excluding identifiers and raw target
                columns.
        """
        excluded = _NON_FEATURE_COLUMNS | {"Store", "Date"}
        return [col for col in df.columns if col not in excluded]

    def _output_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Build the ordered output column list for ``transform``.

        Args:
            df (pd.DataFrame): Processed table before column selection.

        Returns:
            list[str]: Unique ordered list of id, feature, and target columns.
        """
        id_cols = [col for col in ("Id", "Store", "Date") if col in df.columns]
        feature_cols = [
            col
            for col in self.feature_columns_
            if col in df.columns and col not in id_cols
        ]
        target_cols = ["y"] if "y" in df.columns else []
        ordered = id_cols + feature_cols + target_cols
        unique_columns: list[str] = []
        seen: set[str] = set()
        for col in ordered:
            if col in seen:
                continue
            seen.add(col)
            unique_columns.append(col)
        return unique_columns
