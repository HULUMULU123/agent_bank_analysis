"""Global anomaly analysis using HBOS and COPOD."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS


@dataclass
class GlobalBehaviorResult:
    scores: pd.DataFrame


class GlobalBehaviorTool:
    """Detect globally rare transactions independent of client history."""

    def __init__(self, contamination: float = 0.02, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state

    def run(self, df: pd.DataFrame) -> GlobalBehaviorResult:
        base_cols = [
            "amount",
            "debit_roll_cnt_30d", "debit_roll_mean_30d", "debit_roll_std_30d",
            "debit_amount_spike_ratio_7d", "debit_tx_rate_spike_7d", "debit_amount_volatility_30d",
            "credit_roll_cnt_30d", "credit_roll_mean_30d", "credit_roll_std_30d",
            "credit_amount_spike_ratio_7d", "credit_tx_rate_spike_7d", "credit_amount_volatility_30d",
            "daily_debit_transaction_count", "daily_credit_transaction_count",
            "daily_debit_percent", "daily_credit_percent",
            "days_since_last_txn_debit", "days_since_last_txn_credit",
            "debit_fan_out_ratio", "credit_fan_in_ratio",
            "in_out_ratio_30d", "round_large_amount",
            "day_of_week", "is_weekend", "is_month_end",
            "purpose_stopword_high",
        ]

        X = df[base_cols].astype(float)
        hbos = HBOS(contamination=self.contamination)
        copod = COPOD(contamination=self.contamination)

        hbos.fit(X)
        copod.fit(X)

        hbos_score = hbos.decision_function(X)
        copod_score = copod.decision_function(X)
        global_score = (hbos_score + copod_score) / 2.0

        scores = pd.DataFrame({
            "txn_id": df["txn_id"],
            "hbos_score": hbos_score,
            "copod_score": copod_score,
            "global_score": global_score,
        })
        return GlobalBehaviorResult(scores=scores)
