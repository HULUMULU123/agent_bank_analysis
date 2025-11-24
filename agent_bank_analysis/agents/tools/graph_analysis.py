"""Graph-based anomaly detection stub inspired by the notebook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class GraphAnalysisResult:
    edge_scores: pd.DataFrame
    node_scores: pd.DataFrame


class GraphAnalysisTool:
    """Compute simple graph-based risk scores.

    The original notebook used a GCN autoencoder. Here we keep a lightweight
    approximation based on edge frequency and neighborhood overlap. This keeps
    the module deterministic while leaving room for a future deep model.
    """

    def __init__(self) -> None:
        self.graph = nx.Graph()

    def _build_graph(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            debit = str(row["debit_inn"])
            credit = str(row["credit_inn"])
            amount = float(row.get("amount", 0))
            self.graph.add_edge(debit, credit, amount=amount)

    def _edge_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = []
        for u, v, data in self.graph.edges(data=True):
            neighbors_u = set(self.graph.neighbors(u))
            neighbors_v = set(self.graph.neighbors(v))
            jaccard = len(neighbors_u & neighbors_v) / (len(neighbors_u | neighbors_v) + 1e-6)
            weight = data.get("amount", 0.0)
            rarity = 1.0 / (self.graph.number_of_edges(u, v) + 1)
            score = float((1 - jaccard) * 0.6 + rarity * 0.4)
            scores.append({"debit_inn": u, "credit_inn": v, "graph_edge_score": score, "total_amount": weight})
        return pd.DataFrame(scores)

    def _node_scores(self) -> pd.DataFrame:
        scores = []
        for node in self.graph.nodes:
            degree = self.graph.degree[node]
            clustering = nx.clustering(self.graph, node)
            score = float((1 - clustering) * 0.5 + (1 / (degree + 1)) * 0.5)
            scores.append({"inn": node, "graph_node_score": score})
        return pd.DataFrame(scores)

    def run(self, df: pd.DataFrame) -> GraphAnalysisResult:
        self._build_graph(df)
        edge_scores = self._edge_scores(df)
        node_scores = self._node_scores()
        return GraphAnalysisResult(edge_scores=edge_scores, node_scores=node_scores)

    @staticmethod
    def enrich_transactions(df: pd.DataFrame, result: GraphAnalysisResult) -> pd.DataFrame:
        df = df.copy()
        edge_df = result.edge_scores
        node_df = result.node_scores

        df["debit_inn"] = df["debit_inn"].astype(str)
        df["credit_inn"] = df["credit_inn"].astype(str)

        if not edge_df.empty:
            df = df.merge(edge_df, on=["debit_inn", "credit_inn"], how="left")
        else:
            df["graph_edge_score"] = np.nan

        if not node_df.empty:
            df = df.merge(
                node_df.rename(columns={"inn": "debit_inn", "graph_node_score": "graph_node_score_debit"}),
                on="debit_inn",
                how="left",
            )
            df = df.merge(
                node_df.rename(columns={"inn": "credit_inn", "graph_node_score": "graph_node_score_credit"}),
                on="credit_inn",
                how="left",
            )
            df["graph_node_score"] = df[["graph_node_score_debit", "graph_node_score_credit"]].max(axis=1)
            df.drop(columns=["graph_node_score_debit", "graph_node_score_credit"], inplace=True)
        else:
            df["graph_node_score"] = np.nan
        return df
