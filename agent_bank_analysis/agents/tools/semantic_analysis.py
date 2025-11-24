"""Semantic analysis tools:
- GigaChat юридический анализ транзакций (батчи),
- семантический risk-score по SVD-фичам текста.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# батчевый раннер GigaChat из llm_client.py
from ..llm_client import run_gigachat_over_df


# =====================================================================
# 1. LLM-АНАЛИТИКА (GigaChat, батчи, random sampling)
# =====================================================================

@dataclass
class SemanticAnalysisResult:
    prompt: str
    responses: pd.DataFrame


class SemanticAnalysisTool:
    """
    Обёртка над GigaChat-аналитикой:
    - фильтр по сумме (min_amount_for_llm),
    - случайный выбор N транзакций (sample_n),
    - батчевые вызовы (batch_size),
    - возврат таблицы с колонками LLM.

    Важно:
    - df обязан содержать 'txn_id' и 'amount'.
    - все остальные аналитические поля выбираются внутри run_gigachat_over_df.
    """

    def __init__(
        self,
        sample_n: Optional[int] = 20,
        min_amount_for_llm: Optional[float] = 50_000,
        batch_size: int = 10,
    ) -> None:
        self.sample_n = sample_n
        self.min_amount_for_llm = min_amount_for_llm
        self.batch_size = batch_size

    def run(self, df: pd.DataFrame) -> SemanticAnalysisResult:
        if "txn_id" not in df.columns:
            raise ValueError("SemanticAnalysisTool: в df нет колонки 'txn_id'.")
        if "amount" not in df.columns:
            raise ValueError("SemanticAnalysisTool: в df нет колонки 'amount'.")

        df_in = df.copy()
        df_in["txn_id"] = df_in["txn_id"].astype(str)

        # вызов GigaChat через helper из llm_client.py
        df_llm = run_gigachat_over_df(
            df_in,
            batch_size=self.batch_size,
            sample_n=self.sample_n,
            min_amount_for_llm=self.min_amount_for_llm,
        )

        # компактная текстовая сводка (вместо гигантского промпта)
        prompt_info = (
            f"GigaChat legal analysis: "
            f"batch_size={self.batch_size}, "
            f"sample_n={self.sample_n}, "
            f"min_amount_for_llm={self.min_amount_for_llm}, "
            f"rows_sent={len(df_llm)}"
        )

        responses = df_llm.copy()
        return SemanticAnalysisResult(prompt=prompt_info, responses=responses)


# =====================================================================
# 2. СЕМАНТИЧЕСКИЙ RISK-SCORE ПО SVD-ФИЧАМ НАЗНАЧЕНИЯ
# =====================================================================

@dataclass
class SemanticRiskResult:
    scores: pd.DataFrame


class SemanticRiskTool:
    """
    Строит "семантический" риск по тексту назначения платежа:
    - локальный сдвиг текста для ИНН (debit / credit),
    - глобальная редкость текста,
    - флаг keyword-risk (purpose_stopword_high),
    - итоговый semantic_risk_score и ранги.

    Требуемые колонки во входном df:
        - 'txn_id'
        - 'purpose'
        - 'debit_inn', 'credit_inn'
        - 'purpose_svd_1' ... 'purpose_svd_50' (или меньше, возьмём все существующие)
        - (опционально) 'purpose_stopword_high' (если нет — считаем 0.0)

    Возвращает:
        scores: DataFrame с колонками
            ['txn_id',
             'purpose_semantic_shift_debit',
             'purpose_semantic_shift_credit',
             'purpose_semantic_shift_local',
             'purpose_semantic_global_outlier',
             'purpose_keyword_risk',
             'semantic_risk_score',
             'semantic_risk_rank']
    """

    def __init__(self, min_group_size: int = 5) -> None:
        self.min_group_size = min_group_size

    @staticmethod
    def _compute_semantic_shift_per_inn(
        df_base: pd.DataFrame,
        svd_cols: list[str],
        inn_col: str,
        suffix: str,
        min_group_size: int,
    ) -> pd.Series:
        """
        Для каждой строки считаем, насколько её текст (purpose_svd_*)
        далёк от "типичного" текста этого ИНН.
        dist_norm ∈ [0,1] внутри каждого ИНН.
        """
        svd_mat = df_base[svd_cols].fillna(0.0).astype(float).values

        shift = np.zeros(len(df_base), dtype=np.float32)
        groups = df_base.groupby(inn_col).indices  # dict: inn -> np.array индексов

        for _, idx_arr in groups.items():
            idx = np.asarray(idx_arr, dtype=int)
            if idx.size < min_group_size:
                shift[idx] = 0.0
                continue

            X = svd_mat[idx]  # [n_txn, k]
            center = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-6

            Z = (X - center) / std
            dist = np.sqrt((Z ** 2).sum(axis=1))

            d_min, d_max = dist.min(), dist.max()
            if d_max > d_min:
                dist_norm = (dist - d_min) / (d_max - d_min)
            else:
                dist_norm = np.zeros_like(dist)

            shift[idx] = dist_norm.astype(np.float32)

        col_name = f"purpose_semantic_shift_{suffix}"
        return pd.Series(shift, index=df_base.index, name=col_name)

    def run(self, df: pd.DataFrame) -> SemanticRiskResult:
        # --- проверки колонок ---
        required = ["txn_id", "purpose", "debit_inn", "credit_inn"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"SemanticRiskTool: в df нет обязательных колонок: {missing}")

        svd_cols = [c for c in df.columns if c.startswith("purpose_svd_")]
        if not svd_cols:
            raise ValueError("SemanticRiskTool: нет колонок purpose_svd_*. Нужны SVD-компоненты текста.")

        df_sem = df.copy()
        df_sem["txn_id"] = df_sem["txn_id"].astype(str)

        # purpose_stopword_high, если нет — нулевой флаг
        if "purpose_stopword_high" not in df_sem.columns:
            df_sem["purpose_stopword_high"] = 0.0
        df_sem["purpose_stopword_high"] = df_sem["purpose_stopword_high"].astype(float)

        # --- 1. Локальный semantic shift по дебету и кредиту ---
        df_sem["purpose_semantic_shift_debit"] = self._compute_semantic_shift_per_inn(
            df_base=df_sem,
            svd_cols=svd_cols,
            inn_col="debit_inn",
            suffix="debit",
            min_group_size=self.min_group_size,
        )

        df_sem["purpose_semantic_shift_credit"] = self._compute_semantic_shift_per_inn(
            df_base=df_sem,
            svd_cols=svd_cols,
            inn_col="credit_inn",
            suffix="credit",
            min_group_size=self.min_group_size,
        )

        # агрегированный локальный сдвиг: максимум по ролям
        df_sem["purpose_semantic_shift_local"] = df_sem[
            ["purpose_semantic_shift_debit", "purpose_semantic_shift_credit"]
        ].max(axis=1)

        # --- 2. Глобальная семантическая редкость текста ---
        X_svd = df_sem[svd_cols].fillna(0.0).astype(float).values
        global_center = X_svd.mean(axis=0, keepdims=True)
        global_std = X_svd.std(axis=0, keepdims=True) + 1e-6

        Zg = (X_svd - global_center) / global_std
        dist_global = np.sqrt((Zg ** 2).sum(axis=1))

        dg_min, dg_max = dist_global.min(), dist_global.max()
        if dg_max > dg_min:
            dist_global_norm = (dist_global - dg_min) / (dg_max - dg_min)
        else:
            dist_global_norm = np.zeros_like(dist_global)

        df_sem["purpose_semantic_global_outlier"] = dist_global_norm.astype(np.float32)

        # --- 3. Keyword-risk ---
        df_sem["purpose_keyword_risk"] = df_sem["purpose_stopword_high"].astype(float)

        # --- 4. Итоговый semantic_risk_score ---
        w_local = 0.5
        w_global = 0.3
        w_kw = 0.2

        semantic_raw = (
            w_local * df_sem["purpose_semantic_shift_local"]
            + w_global * df_sem["purpose_semantic_global_outlier"]
            + w_kw * df_sem["purpose_keyword_risk"]
        )

        s_min, s_max = semantic_raw.min(), semantic_raw.max()
        if s_max > s_min:
            semantic_score = (semantic_raw - s_min) / (s_max - s_min)
        else:
            semantic_score = pd.Series(0.0, index=df_sem.index)

        df_sem["semantic_risk_score"] = semantic_score.astype(np.float32)
        df_sem["semantic_risk_rank"] = df_sem["semantic_risk_score"].rank(
            method="average", pct=True
        )

        # --- 5. Собираем компактный результат по txn_id ---
        scores = df_sem[
            [
                "txn_id",
                "purpose_semantic_shift_debit",
                "purpose_semantic_shift_credit",
                "purpose_semantic_shift_local",
                "purpose_semantic_global_outlier",
                "purpose_keyword_risk",
                "semantic_risk_score",
                "semantic_risk_rank",
            ]
        ].copy()

        return SemanticRiskResult(scores=scores)
