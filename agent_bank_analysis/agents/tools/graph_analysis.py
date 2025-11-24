"""Graph-based anomaly detection stub inspired by the notebook."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class GraphAnalysisResult:
    edge_scores: pd.DataFrame
    node_scores: pd.DataFrame


class GraphAnalysisTool:
    """Compute simple graph-based risk scores.

    Lightweight approximation based on:
    - edge rarity (кол-во транзакций между парой ИНН),
    - neighborhood overlap (Jaccard соседей),
    - локальная кластеризация узла.

    ВАЖНО:
    - считаем скор ОДИН раз на каждую пару ИНН (debit_inn, credit_inn),
      а не на каждое отдельное ребро MultiGraph;
    - нормализуем скор по всей выборке в [0, 1], чтобы не залипать в одном значении.
    """

    def __init__(self) -> None:
        # MultiGraph, чтобы учитывать кратные связи между ИНН
        self.graph = nx.MultiGraph()

    # ------------------------------------------------------------------
    # ГРАФ
    # ------------------------------------------------------------------
    def _build_graph(self, df: pd.DataFrame) -> None:
        """Строим MultiGraph по парам ИНН."""
        self.graph.clear()

        for _, row in df.iterrows():
            debit = str(row["debit_inn"])
            credit = str(row["credit_inn"])
            amount = float(row.get("amount", 0.0))

            # добавляем ребро; MultiGraph позволяет хранить несколько транзакций
            # между одной и той же парой контрагентов
            self.graph.add_edge(debit, credit, amount=amount)

    # ------------------------------------------------------------------
    # EDGE SCORES
    # ------------------------------------------------------------------
    def _edge_scores(self) -> pd.DataFrame:
        """
        Считаем скор для КАЖДОЙ УНИКАЛЬНОЙ пары ИНН (u, v):
        - edge_txn_count: сколько транзакций между ними;
        - edge_total_amount: суммарный объём;
        - jaccard: похожесть окружений;
        - rarity: редкость пары по числу транзакций.

        Затем нормализуем score в [0, 1] по всей выборке.
        """
        if self.graph.number_of_edges() == 0:
            return pd.DataFrame(
                columns=[
                    "debit_inn",
                    "credit_inn",
                    "graph_edge_score",
                    "edge_txn_count",
                    "edge_total_amount",
                ]
            )

        rows = []

        # обходим уникальные пары соседей через adjacency-структуру,
        # чтобы не дублировать (u, v) и (v, u) и не считать пару
        # несколько раз из-за MultiGraph
        for u in self.graph.nodes:
            for v, edges_between in self.graph[u].items():
                # гарантируем порядок, чтобы (u, v) и (v, u) не дублировались
                if u >= v:
                    continue

                # edges_between — dict с ключами edge_key -> edge_attr
                edge_txn_count = len(edges_between)
                edge_total_amount = float(
                    sum(d.get("amount", 0.0) for d in edges_between.values())
                )

                neighbors_u = set(self.graph.neighbors(u)) - {v}
                neighbors_v = set(self.graph.neighbors(v)) - {u}
                union_sz = len(neighbors_u | neighbors_v)
                inter_sz = len(neighbors_u & neighbors_v)
                jaccard = inter_sz / (union_sz + 1e-6)

                # редкость пары: чем меньше транзакций, тем выше rarity
                rarity = 1.0 / (edge_txn_count + 1.0)

                # базовый скор: высоко, если мало общих соседей и мало транзакций
                raw_score = float((1.0 - jaccard) * 0.6 + rarity * 0.4)

                rows.append(
                    {
                        "debit_inn": u,
                        "credit_inn": v,
                        "graph_edge_score": raw_score,
                        "edge_txn_count": edge_txn_count,
                        "edge_total_amount": edge_total_amount,
                    }
                )

        edge_df = pd.DataFrame(rows)

        if edge_df.empty:
            return edge_df

        # ---- НОРМАЛИЗАЦИЯ В [0, 1], чтобы избежать «залипания» ----
        s = edge_df["graph_edge_score"].astype(float)
        s_min = float(s.min())
        s_max = float(s.max())
        denom = s_max - s_min

        if denom <= 1e-6:
            # если все значения почти одинаковые — ставим 0.5
            edge_df["graph_edge_score"] = 0.5
        else:
            edge_df["graph_edge_score"] = (s - s_min) / (denom + 1e-6)

        return edge_df

    # ------------------------------------------------------------------
    # NODE SCORES
    # ------------------------------------------------------------------
    def _node_scores(self) -> pd.DataFrame:
        """
        Скор для узла: чем он более «висящий» и менее кластеризованный,
        тем выше graph_node_score. Потом нормализуем в [0, 1].
        """
        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=["inn", "graph_node_score"])

        scores = []
        # приводим к простому графу для расчёта кластеризации
        simple_g = nx.Graph(self.graph)

        for node in simple_g.nodes:
            degree = simple_g.degree[node]
            clustering = nx.clustering(simple_g, node)
            # чем меньше кластеризация и чем меньше degree — тем выше скор
            raw_score = float((1.0 - clustering) * 0.5 + (1.0 / (degree + 1.0)) * 0.5)
            scores.append({"inn": node, "graph_node_score": raw_score})

        node_df = pd.DataFrame(scores)

        if node_df.empty:
            return node_df

        # нормализация в [0, 1]
        s = node_df["graph_node_score"].astype(float)
        s_min = float(s.min())
        s_max = float(s.max())
        denom = s_max - s_min

        if denom <= 1e-6:
            node_df["graph_node_score"] = 0.5
        else:
            node_df["graph_node_score"] = (s - s_min) / (denom + 1e-6)

        return node_df

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> GraphAnalysisResult:
        self._build_graph(df)
        edge_scores = self._edge_scores()
        node_scores = self._node_scores()
        return GraphAnalysisResult(edge_scores=edge_scores, node_scores=node_scores)

    @staticmethod
    def enrich_transactions(df: pd.DataFrame, result: GraphAnalysisResult) -> pd.DataFrame:
        """Добавляет в транзакции graph_edge_score и graph_node_score."""
        df = df.copy()
        edge_df = result.edge_scores
        node_df = result.node_scores

        df["debit_inn"] = df["debit_inn"].astype(str)
        df["credit_inn"] = df["credit_inn"].astype(str)

        # edge-level
        if edge_df is not None and not edge_df.empty:
            df = df.merge(
                edge_df[["debit_inn", "credit_inn", "graph_edge_score"]],
                on=["debit_inn", "credit_inn"],
                how="left",
            )
        else:
            df["graph_edge_score"] = np.nan

        # node-level
        if node_df is not None and not node_df.empty:
            df = df.merge(
                node_df.rename(
                    columns={"inn": "debit_inn", "graph_node_score": "graph_node_score_debit"}
                ),
                on="debit_inn",
                how="left",
            )
            df = df.merge(
                node_df.rename(
                    columns={"inn": "credit_inn", "graph_node_score": "graph_node_score_credit"}
                ),
                on="credit_inn",
                how="left",
            )
            df["graph_node_score"] = df[
                ["graph_node_score_debit", "graph_node_score_credit"]
            ].max(axis=1)
            df.drop(
                columns=["graph_node_score_debit", "graph_node_score_credit"], inplace=True
            )
        else:
            df["graph_node_score"] = np.nan

        return df
