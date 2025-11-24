"""Time series helper for trend-aware scoring."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesResult:
    scores: pd.DataFrame


class TimeSeriesTool:
    """Compute rolling volatility to capture temporal anomalies."""

    def __init__(self, window: int = 7) -> None:
        self.window = window

    def run(self, df: pd.DataFrame) -> TimeSeriesResult:
        df_sorted = df.sort_values(["inn" if "inn" in df.columns else "debit_inn", "date"])
        grouped = df_sorted.groupby("debit_inn") if "debit_inn" in df_sorted.columns else df_sorted.groupby("inn")

        volatility = grouped["amount"].transform(lambda s: s.rolling(self.window, min_periods=1).std()).fillna(0)
        accel = grouped["amount"].transform(lambda s: s.diff()).fillna(0)

        scores = pd.DataFrame({
            "txn_id": df_sorted["txn_id"],
            "ts_volatility": volatility.values,
            "ts_acceleration": accel.values,
        })
        scores["ts_score"] = (scores["ts_volatility"] - scores["ts_volatility"].min()) / (
            scores["ts_volatility"].max() - scores["ts_volatility"].min() + 1e-6
        )
        return TimeSeriesResult(scores=scores)
