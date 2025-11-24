"""Pipeline orchestrating all analysis layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..data_loader import load_statement
from ..feature_engineering import FeatureEngineer
from .llm_client import EchoLLMClient, LLMClient
from .tools.global_behavior import GlobalBehaviorTool
from .tools.graph_analysis import GraphAnalysisTool
from .tools.individual_behavior import IndividualBehaviorTool
from .tools.score_aggregator import ScoreAggregator
from .tools.semantic_analysis import SemanticAnalysisTool
from .tools.time_series import TimeSeriesTool


@dataclass
class AgentOutputs:
    enriched: pd.DataFrame
    prompt: str
    llm_responses: pd.DataFrame


class TransactionRiskAgent:
    """Production-ready wrapper around the notebook logic."""

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or EchoLLMClient()
        self.feature_engineer = FeatureEngineer()
        self.individual_tool = IndividualBehaviorTool()
        self.global_tool = GlobalBehaviorTool()
        self.graph_tool = GraphAnalysisTool()
        self.ts_tool = TimeSeriesTool()
        self.aggregator = ScoreAggregator()
        self.semantic_tool = SemanticAnalysisTool(self.llm)

    def run(self, path: Optional[str] = None) -> AgentOutputs:
        raw_df = load_statement(path)

        engineered = self.feature_engineer.transform(raw_df)
        behavior_result = self.individual_tool.run(engineered.debit_table, engineered.credit_table)
        global_result = self.global_tool.run(engineered.dataframe)
        graph_result = self.graph_tool.run(engineered.dataframe)
        graph_enriched = self.graph_tool.enrich_transactions(engineered.dataframe, graph_result)
        ts_result = self.ts_tool.run(engineered.dataframe)

        aggregated = self.aggregator.run(
            base_df=engineered.dataframe,
            behavior=behavior_result.scores,
            global_scores=global_result.scores,
            graph_enriched=graph_enriched,
            ts_scores=ts_result.scores,
        )

        # Placeholder for INN hand-off: here we could call another agent with INNs from risk clusters
        # Example: inn_list = aggregated.dataframe.query("overall_risk > 0.5")["debit_inn"].unique()

        semantic = self.semantic_tool.run(aggregated.dataframe)
        enriched = aggregated.dataframe.merge(
            semantic.responses,
            on="txn_id",
            how="left",
            suffixes=("", "_llm"),
        )

        return AgentOutputs(enriched=enriched, prompt=semantic.prompt, llm_responses=semantic.responses)
