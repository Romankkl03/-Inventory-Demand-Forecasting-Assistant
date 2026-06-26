from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalSplit:
    """
    Container for temporally partitioned datasets.

    Attributes:
        train (pd.DataFrame): Training partition.
        val (pd.DataFrame | None): Validation partition, if configured.
        test (pd.DataFrame | None): Test partition, if configured.
    """

    train: pd.DataFrame
    val: pd.DataFrame | None = None
    test: pd.DataFrame | None = None


class TemporalSpliter:
    """
    Split a time-indexed DataFrame into train, validation, and test partitions.

    Supports two configuration modes: explicit date boundaries or fractional
    splits over unique dates. Ratio-based splits preserve chronological order
    and never shuffle rows.

    Attributes:
        train_start (pd.Timestamp | None): Inclusive start of the train window.
        train_end (pd.Timestamp | None): Inclusive end of the train window.
        val_start (pd.Timestamp | None): Inclusive start of the validation window.
        val_end (pd.Timestamp | None): Inclusive end of the validation window.
        test_start (pd.Timestamp | None): Inclusive start of the test window.
        test_end (pd.Timestamp | None): Inclusive end of the test window.
        train_ratio (float | None): Fraction of unique dates assigned to train.
        val_ratio (float | None): Fraction of unique dates assigned to validation.
        test_ratio (float | None): Fraction of unique dates assigned to test.
        date_col (str): Name of the date column used for splitting. Defaults to
            ``"Date"``.
    """

    def __init__(
        self,
        train_start: str | pd.Timestamp | None = None,
        train_end: str | pd.Timestamp | None = None,
        val_start: str | pd.Timestamp | None = None,
        val_end: str | pd.Timestamp | None = None,
        test_start: str | pd.Timestamp | None = None,
        test_end: str | pd.Timestamp | None = None,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
        date_col: str = "Date",
    ):
        self.train_start = self._to_timestamp(train_start)
        self.train_end = self._to_timestamp(train_end)
        self.val_start = self._to_timestamp(val_start)
        self.val_end = self._to_timestamp(val_end)
        self.test_start = self._to_timestamp(test_start)
        self.test_end = self._to_timestamp(test_end)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.date_col = date_col
        self._validate_config()

    def split(self, df: pd.DataFrame) -> TemporalSplit:
        """
        Partition ``df`` into temporal train, validation, and test subsets.

        Args:
            df (pd.DataFrame): Input table containing ``date_col``.

        Returns:
            TemporalSplit: Partitioned DataFrames with reset indices.

        Raises:
            KeyError: If ``date_col`` is missing from ``df``.
            ValueError: If the train partition is empty.
        """
        if self.date_col not in df.columns:
            raise KeyError(f"Column {self.date_col!r} not found in dataframe")

        dates = pd.to_datetime(df[self.date_col])

        if self._uses_ratios():
            train_bounds, val_bounds, test_bounds = self._boundaries_from_ratios(dates)
            train = self._select_range(df, dates, *train_bounds)
            val = self._select_range(df, dates, *val_bounds) if val_bounds else None
            test = self._select_range(df, dates, *test_bounds) if test_bounds else None
        else:
            train = self._select_range(df, dates, self.train_start, self.train_end)
            val = self._select_range(df, dates, self.val_start, self.val_end)
            test = self._select_range(df, dates, self.test_start, self.test_end)

        if train is None:
            raise ValueError("Train split is empty. Check split dates or ratios.")

        return TemporalSplit(
            train=train.reset_index(drop=True),
            val=val.reset_index(drop=True) if val is not None else None,
            test=test.reset_index(drop=True) if test is not None else None,
        )

    def _validate_config(self) -> None:
        """
        Validate that split configuration is consistent.

        Raises:
            ValueError: If both explicit dates and ratios are provided, if
                ratios do not sum to 1.0, or if ``train_ratio`` is missing or
                non-positive.
        """
        uses_dates = any(
            value is not None
            for value in (
                self.train_start,
                self.train_end,
                self.val_start,
                self.val_end,
                self.test_start,
                self.test_end,
            )
        )
        uses_ratios = self._uses_ratios()

        if uses_dates and uses_ratios:
            raise ValueError("Use either explicit dates or ratios, not both.")

        if uses_ratios:
            ratios = self._active_ratios()
            total = sum(ratio for _, ratio in ratios)
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}.")
            if self.train_ratio is None or self.train_ratio <= 0:
                raise ValueError("train_ratio must be positive when using ratio split.")

    def _uses_ratios(self) -> bool:
        """
        Check whether ratio-based splitting is configured.

        Returns:
            bool: ``True`` if any of ``train_ratio``, ``val_ratio``, or
                ``test_ratio`` is set.
        """
        return any(ratio is not None for ratio in (self.train_ratio, self.val_ratio, self.test_ratio))

    def _active_ratios(self) -> list[tuple[str, float]]:
        """
        Collect configured non-null split ratios.

        Returns:
            list[tuple[str, float]]: Ordered ``(split_name, ratio)`` pairs.
        """
        ratios: list[tuple[str, float]] = []
        if self.train_ratio is not None:
            ratios.append(("train", self.train_ratio))
        if self.val_ratio is not None:
            ratios.append(("val", self.val_ratio))
        if self.test_ratio is not None:
            ratios.append(("test", self.test_ratio))
        return ratios

    def _boundaries_from_ratios(
        self,
        dates: pd.Series,
    ) -> tuple[
        tuple[pd.Timestamp, pd.Timestamp],
        tuple[pd.Timestamp, pd.Timestamp] | None,
        tuple[pd.Timestamp, pd.Timestamp] | None,
    ]:
        """
        Convert fractional split ratios into inclusive date boundaries.

        Args:
            dates (pd.Series): Date column from the input DataFrame.

        Returns:
            tuple: ``(train_bounds, val_bounds, test_bounds)`` where each
                bound is ``(start_date, end_date)`` or ``None`` if the split
                is not configured.

        Raises:
            ValueError: If the input contains no dates.
        """
        unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
        n_dates = len(unique_dates)
        if n_dates == 0:
            raise ValueError("Cannot split by ratios: dataframe has no dates.")

        ratios = self._active_ratios()
        boundaries: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        start_idx = 0

        for index, (name, ratio) in enumerate(ratios):
            if index == len(ratios) - 1:
                end_idx = n_dates
            else:
                end_idx = start_idx + int(n_dates * ratio)
                end_idx = max(end_idx, start_idx + 1)

            boundaries[name] = (unique_dates[start_idx], unique_dates[end_idx - 1])
            start_idx = end_idx

        return (
            boundaries["train"],
            boundaries.get("val"),
            boundaries.get("test"),
        )

    @staticmethod
    def _to_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
        """
        Parse an optional date boundary into a timestamp.

        Args:
            value (str | pd.Timestamp | None): Date-like value to parse.

        Returns:
            pd.Timestamp | None: Parsed timestamp, or ``None`` if ``value`` is
                ``None``.
        """
        if value is None:
            return None
        return pd.to_datetime(value)

    @staticmethod
    def _select_range(
        df: pd.DataFrame,
        dates: pd.Series,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        """
        Select rows whose dates fall within an inclusive range.

        Args:
            df (pd.DataFrame): Input table.
            dates (pd.Series): Parsed date column aligned with ``df``.
            start (pd.Timestamp | None): Inclusive lower bound.
            end (pd.Timestamp | None): Inclusive upper bound.

        Returns:
            pd.DataFrame | None: Filtered copy, or ``None`` if both bounds are
                ``None`` or the selection is empty.
        """
        if start is None and end is None:
            return None

        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= dates >= start
        if end is not None:
            mask &= dates <= end

        selected = df[mask]
        if selected.empty:
            return None
        return selected.copy()
