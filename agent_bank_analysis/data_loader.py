"""Data loading helpers for the transaction analysis agent."""
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DATA_DIR, DEFAULT_INPUT_PATTERN


def discover_statement(path: Optional[Path] = None, pattern: str = DEFAULT_INPUT_PATTERN) -> Path:
    """Return the first statement file matching the pattern.

    The notebook previously read a specific CSV from the ``data`` directory.
    Here we keep the same default while allowing callers to override the
    location or file pattern.
    """
    base_dir = Path(path) if path else DATA_DIR
    if base_dir.is_file():
        return base_dir

    matches = sorted(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files matching {pattern} found under {base_dir.resolve()}"
        )
    return matches[0]


def load_statement(path: Optional[Path] = None, pattern: str = DEFAULT_INPUT_PATTERN) -> pd.DataFrame:
    """Load a bank statement into a pandas ``DataFrame``.

    The loader supports both CSV and Excel files and preserves the original
    column names so that downstream feature engineering matches the notebook
    logic.
    """
    file_path = discover_statement(path, pattern)
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    return df
