"""Individual behavior analysis tool using IsolationForest."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class IndividualBehaviorResult:
    scores: pd.DataFrame


class IndividualBehaviorTool:
    """Run per-entity IsolationForest models for debit and credit roles."""

    def __init__(self, random_state: int = 42, contamination: float = 0.05) -> None:
        self.random_state = random_state
        self.contamination = contamination

    def _fit_side(self, table: pd.DataFrame) -> pd.Series:
        feature_cols = [
            "log_amount",
            "roll_cnt_30d",
            "roll_mean_30d",
            "roll_std_30d",
            "amount_spike_ratio_7d",
            "tx_rate_spike_7d",
            "amount_volatility_30d",
            "accel_ratio_7d",
            "to_p95_ratio",
            "new_counterparty_ratio",
            "fan_ratio",
            "volatility_z",
            "daily_total",
            "daily_txn_count",
            "daily_percent",
            "days_since_last_txn",
        ]
        model = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(table[feature_cols])
        raw_score = -model.decision_function(table[feature_cols])
        normalized = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-6)
        return pd.Series(normalized, index=table.index)

    def run(self, debit_table: pd.DataFrame, credit_table: pd.DataFrame) -> IndividualBehaviorResult:
        debit_score = self._fit_side(debit_table)
        credit_score = self._fit_side(credit_table)

        scores = pd.DataFrame({
            "txn_id": debit_table["txn_id"],
            "iforest_score_debit": debit_score,
            "iforest_score_credit": credit_score,
        })
        scores["behavior_score"] = scores[["iforest_score_debit", "iforest_score_credit"]].max(axis=1)
        return IndividualBehaviorResult(scores=scores)
