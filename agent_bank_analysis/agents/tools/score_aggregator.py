"""Combine scores from all tools into a unified dataframe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AggregationResult:
    dataframe: pd.DataFrame


class ScoreAggregator:
    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.weights = weights or {
            "behavior_score": 0.35,
            "global_score": 0.25,
            "graph_edge_score": 0.2,
            "graph_node_score": 0.1,
            "ts_score": 0.1,
        }

    def run(
        self,
        base_df: pd.DataFrame,
        behavior: pd.DataFrame,
        global_scores: pd.DataFrame,
        graph_enriched: pd.DataFrame,
        ts_scores: pd.DataFrame,
    ) -> AggregationResult:
        df = base_df.copy()
        df = df.merge(behavior, on="txn_id", how="left")
        df = df.merge(global_scores, on="txn_id", how="left")
        df = df.merge(graph_enriched[["txn_id", "graph_edge_score", "graph_node_score"]], on="txn_id", how="left")
        df = df.merge(ts_scores, on="txn_id", how="left")

        df["behavior_score"].fillna(0, inplace=True)
        df["global_score"].fillna(0, inplace=True)
        df["graph_edge_score"].fillna(0, inplace=True)
        df["graph_node_score"].fillna(0, inplace=True)
        df["ts_score"].fillna(0, inplace=True)

        df["overall_risk"] = sum(
            df[col] * weight for col, weight in self.weights.items() if col in df.columns
        )
        return AggregationResult(dataframe=df)
