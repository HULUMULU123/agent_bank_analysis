"""Pipeline orchestrating all analysis layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..data_loader import load_statement
from ..feature_engineering import FeatureEngineer
from .llm_client import LLMClient  # тип на будущее, сейчас не обязателен
from .tools.global_behavior import GlobalBehaviorTool
from .tools.graph_analysis import GraphAnalysisTool
from .tools.individual_behavior import IndividualBehaviorTool
from .tools.score_aggregator import ScoreAggregator
from .tools.semantic_analysis import SemanticAnalysisTool
from .tools.time_series import TimeSeriesTool
from .tools.memory import kb_init, kb_upsert_transactions


@dataclass
class AgentOutputs:
    enriched: pd.DataFrame
    prompt: str
    llm_responses: pd.DataFrame


class TransactionRiskAgent:
    """Production-ready wrapper around the notebook logic."""

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        # llm тут пока не используется напрямую — вся логика LLM уехала в run_gigachat_over_df
        self.llm = llm

        self.feature_engineer = FeatureEngineer()
        self.individual_tool = IndividualBehaviorTool()
        self.global_tool = GlobalBehaviorTool()
        self.graph_tool = GraphAnalysisTool()
        self.ts_tool = TimeSeriesTool()
        # Новый ScoreAggregator:
        #  - выбирает между global_behavior_score и iforest_score_*
        #  - добавляет корректоры (semantic_risk_score, TS, graph)
        #  - подмешивает историю из memory (kb_inn_stats) и даёт overall_risk
        self.aggregator = ScoreAggregator()

        # SemanticAnalysisTool сам дергает GigaChat батчами через llm_client.run_gigachat_over_df
        # Это LEGAL/текстовый слой, не тот semantic_risk_score из фичей.
        self.semantic_tool = SemanticAnalysisTool(
            sample_n=20,                # сколько случайных транзакций отправлять в LLM
            min_amount_for_llm=50_000,  # порог суммы
            batch_size=10,              # размер батча
        )

        # Инициализация SQLite-памяти агента (если БД уже есть — просто ничего не делает)
        kb_init()

    def run(self, path: Optional[str] = None) -> AgentOutputs:
        # 1. Загрузка и feature engineering
        raw_df = load_statement(path)
        engineered = self.feature_engineer.transform(raw_df)
        base_df = engineered.dataframe.copy()

        # 2. Индивидуальное поведение (IsolationForest по ролям/ИНН)
        # behavior_result.scores: txn_id, iforest_score_debit, iforest_score_credit, ...
        behavior_result = self.individual_tool.run(
            engineered.debit_table, engineered.credit_table
        )
        df_work = base_df.merge(
            behavior_result.scores[
                ["txn_id", "iforest_score_debit", "iforest_score_credit"]
            ],
            on="txn_id",
            how="left",
        )

        # 3. Глобальный поведенческий слой (HBOS + COPOD)
        # global_result.scores может содержать:
        #   txn_id, hbos_score, copod_score, global_score
        #   (а в новой версии — ещё и global_behavior_score / global_behavior_rank)
        global_result = self.global_tool.run(base_df)
        df_work = df_work.merge(
            global_result.scores,   # <-- берём все колонки, без жёсткого списка
            on="txn_id",
            how="left",
        )

        # 4. Графовый слой (edge/node risk)
        graph_result = self.graph_tool.run(base_df)
        graph_enriched = self.graph_tool.enrich_transactions(base_df, graph_result)
        df_work = df_work.merge(
            graph_enriched[["txn_id", "graph_edge_score", "graph_node_score"]],
            on="txn_id",
            how="left",
        )

        # 5. Временные ряды (TS-аномалии по дням/ИНН → ts_anomaly_score_txn)
        ts_result = self.ts_tool.run(base_df)
        # TimeSeriesResult.scores: txn_id, ts_anomaly_score_debit, ts_anomaly_score_credit,
        #                         ts_anomaly_score_txn, ts_anomaly_rank_txn, ...
        ts_cols = [c for c in ts_result.scores.columns if c.startswith("ts_anomaly_")]
        df_work = df_work.merge(
            ts_result.scores[["txn_id"] + ts_cols],
            on="txn_id",
            how="left",
        )

        # Здесь в df_work уже есть:
        #  - iforest_score_debit / credit
        #  - hbos_score / copod_score / (возможно) global_behavior_score
        #  - graph_edge_score / graph_node_score
        #  - ts_anomaly_score_txn
        #  - semantic_risk_score (из FeatureEngineer, если он туда включён)

        # 6. Интеграция всех слоёв + история из памяти:
        aggregated = self.aggregator.run(df_work)

        # 7. LLM-анализ (GigaChat, батчи + random sampling) — отдельный LEGAL/текстовый слой
        semantic = self.semantic_tool.run(aggregated.dataframe)

        # 8. Мержим всё обратно по txn_id
        enriched = aggregated.dataframe.merge(
            semantic.responses,
            on="txn_id",
            how="left",
            suffixes=("", "_llm"),
        )

        # 9. Обновляем память агента (SQLite KB) по результатам текущего запуска
        kb_upsert_transactions(enriched)

        return AgentOutputs(
            enriched=enriched,
            prompt=semantic.prompt,
            llm_responses=semantic.responses,
        )
