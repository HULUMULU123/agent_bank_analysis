"""Semantic analysis tool for transaction purposes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from ..llm_client import LLMClient, LLMResponse


@dataclass
class SemanticAnalysisResult:
    prompt: str
    responses: pd.DataFrame


class SemanticAnalysisTool:
    """Prepare LLM prompts and parse responses for legal-style reasoning."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def _prompt_template() -> str:
        return (
            "Ты аналитик комплаенс. Оцени транзакции по степени риска и дай обоснование.\n"
            "Верни JSON со списком объектов вида {txn_id, risk_level, risk, recommendation, reasoning}."
        )

    def run(self, df: pd.DataFrame) -> SemanticAnalysisResult:
        prompt_lines: List[str] = [self._prompt_template(), "Данные:"]
        for _, row in df.iterrows():
            prompt_lines.append(
                f"- txn_id={row['txn_id']}, amount={row['amount']}, purpose={row['purpose_clean']}"
            )
        prompt = "\n".join(prompt_lines)

        llm_result: LLMResponse = self.llm.generate(prompt)
        parsed = self.llm.parse_json_list(llm_result)
        responses = pd.DataFrame(parsed)
        return SemanticAnalysisResult(prompt=prompt, responses=responses)
