from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RAW_ROSSMANN_DIR = _PROJECT_ROOT / "data" / "raw" / "rossmann"

_RAW_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "store": "store.csv",
}


class DataReader:
    """
    Loader for raw Rossmann Store Sales CSV files.

    Reads ``train.csv``, ``test.csv``, and ``store.csv`` from a configurable
    directory and exposes dataset shapes for quick inspection.

    Attributes:
        filter (str): Data access mode. Only ``"raw"`` is supported.
        data_dir (Path): Directory containing Rossmann CSV files. Defaults to
            ``data/raw/rossmann`` relative to the project root.
    """

    def __init__(self, filter: str = "raw", data_dir: Path | None = None):
        self.filter = filter
        self.data_dir = data_dir or _RAW_ROSSMANN_DIR

    def read(self) -> dict[str, pd.DataFrame]:
        """
        Load all configured Rossmann datasets.

        Returns:
            dict[str, pd.DataFrame]: Mapping with keys ``"train"``, ``"test"``,
                and ``"store"`` pointing to the corresponding DataFrames.

        Raises:
            NotImplementedError: If ``filter`` is not ``"raw"``.
            FileNotFoundError: If any expected CSV file is missing.
        """
        if self.filter == "raw":
            return self._read_raw()
        raise NotImplementedError(f"Data filter {self.filter!r} is not implemented yet")

    def get_dataset_sizes(self) -> dict[str, tuple[int, int]]:
        """
        Return row and column counts for each loaded dataset.

        Returns:
            dict[str, tuple[int, int]]: Dataset name to ``(n_rows, n_cols)``.
        """
        datasets = self.read()
        return {name: df.shape for name, df in datasets.items()}

    def _read_raw(self) -> dict[str, pd.DataFrame]:
        """
        Read Rossmann CSV files from ``data_dir``.

        Returns:
            dict[str, pd.DataFrame]: Loaded train, test, and store tables.

        Raises:
            FileNotFoundError: If a required CSV file does not exist.
        """
        datasets: dict[str, pd.DataFrame] = {}
        for name, filename in _RAW_FILES.items():
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            read_kwargs = {"low_memory": False} if name != "store" else {}
            datasets[name] = pd.read_csv(path, **read_kwargs)
        return datasets
