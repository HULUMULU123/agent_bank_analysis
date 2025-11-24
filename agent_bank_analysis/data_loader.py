"""Data loading helpers for the transaction analysis agent."""
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from .config import DATA_DIR, DEFAULT_INPUT_PATTERNS

InputPattern = Union[str, Iterable[str]]


def _normalize_patterns(pattern: Optional[InputPattern]) -> Iterable[str]:
    if pattern is None:
        return DEFAULT_INPUT_PATTERNS
    if isinstance(pattern, str):
        return (pattern,)
    return tuple(pattern)


def discover_statement(
    path: Optional[Path] = None, pattern: Optional[InputPattern] = DEFAULT_INPUT_PATTERNS
) -> Path:
    """Return the first statement file matching the pattern.

    The notebook previously read a specific CSV from the ``data`` directory.
    Here we keep the same default while allowing callers to override the
    location or file pattern.
    """
    base_dir = Path(path) if path else DATA_DIR
    if base_dir.is_file():
        return base_dir

    patterns = _normalize_patterns(pattern)
    matches = []
    for patt in patterns:
        matches.extend(base_dir.glob(patt))

    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"No files matching {patterns} found under {base_dir.resolve()}"
        )
    return matches[0]


def load_statement(
    path: Optional[Path] = None, pattern: Optional[InputPattern] = DEFAULT_INPUT_PATTERNS
) -> pd.DataFrame:
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
