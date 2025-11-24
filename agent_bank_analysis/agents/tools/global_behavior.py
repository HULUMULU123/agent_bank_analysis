"""Global anomaly analysis using HBOS and COPOD."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler


@dataclass
class GlobalBehaviorResult:
    scores: pd.DataFrame


class GlobalBehaviorTool:
    """Detect globally rare transactions independent of client history.

    Логика синхронизирована с ноутбуком:
    - фиксированный набор глобальных признаков (base_feat_cols + purpose_svd_*),
    - лог-фичи по сумме и интервалам,
    - стандартизация (StandardScaler),
    - HBOS и COPOD по одной и той же матрице Xs,
    - нормировка обоих скорингов в [0, 1],
    - интегральный global_score = среднее двух нормализованных скоров.
    """

    def __init__(self, contamination: float = 0.02, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state

    def run(self, df: pd.DataFrame) -> GlobalBehaviorResult:
        df_g = df.copy()

        # --- 0. Подстраховка по колонкам (как в ноутбуке) ---
        base_feat_cols = [
            # базовая сумма
            "amount",

            # поведенческие окна: дебет
            "debit_roll_cnt_30d",
            "debit_roll_mean_30d",
            "debit_roll_std_30d",
            "debit_amount_spike_ratio_7d",
            "debit_tx_rate_spike_7d",
            "debit_amount_volatility_30d",

            # поведенческие окна: кредит
            "credit_roll_cnt_30d",
            "credit_roll_mean_30d",
            "credit_roll_std_30d",
            "credit_amount_spike_ratio_7d",
            "credit_tx_rate_spike_7d",
            "credit_amount_volatility_30d",

            # активность за день
            "daily_debit_transaction_count",
            "daily_credit_transaction_count",

            # доля текущей операции в суточном объёме
            "daily_debit_percent",
            "daily_credit_percent",

            # интервалы между операциями (в днях)
            "days_since_last_txn_debit",
            "days_since_last_txn_credit",

            # fan-out / fan-in
            "debit_fan_out_ratio",
            "credit_fan_in_ratio",

            # дисбаланс потоков
            "in_out_ratio_30d",

            # округлённые суммы
            "round_large_amount",

            # время
            "day_of_week",
            "is_weekend",
            "is_month_end",

            # риск по ключевым словам в назначении
            "purpose_stopword_high",
        ]

        svd_cols = [f"purpose_svd_{i}" for i in range(1, 51)]
        feat_cols_all = base_feat_cols + svd_cols

        # подстраховка: если чего-то нет – создаём и забиваем нулями
        for c in feat_cols_all:
            if c not in df_g.columns:
                df_g[c] = 0.0

        # флаги → float
        flag_cols = ["round_large_amount", "is_weekend", "is_month_end", "purpose_stopword_high"]
        for c in flag_cols:
            if c in df_g.columns:
                df_g[c] = df_g[c].astype(float)

        # базовые числовые
        df_g["amount"] = df_g["amount"].astype(float)
        df_g["days_since_last_txn_debit"] = df_g["days_since_last_txn_debit"].astype(float)
        df_g["days_since_last_txn_credit"] = df_g["days_since_last_txn_credit"].astype(float)

        # --- 1. Лог-фичи для стабилизации хвостов ---
        df_g["log_amount"] = np.log1p(df_g["amount"].abs())
        df_g["log_days_since_last_debit"] = np.log1p(
            df_g["days_since_last_txn_debit"].clip(lower=0)
        )
        df_g["log_days_since_last_credit"] = np.log1p(
            df_g["days_since_last_txn_credit"].clip(lower=0)
        )

        feat_cols = feat_cols_all + [
            "log_amount",
            "log_days_since_last_debit",
            "log_days_since_last_credit",
        ]

        # --- 2. Подготовка матрицы признаков ---
        X = (
            df_g[feat_cols]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .astype(float)
        )

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # --- 3. HBOS ---
        hbos = HBOS(contamination=self.contamination)
        hbos.fit(Xs)
        hbos_raw = hbos.decision_function(Xs)  # выше — аномальнее

        # нормировка в [0,1]
        h_min, h_max = hbos_raw.min(), hbos_raw.max()
        h_denom = h_max - h_min
        if h_denom <= 1e-9:
            hbos_score = np.full_like(hbos_raw, 0.5, dtype=float)
        else:
            hbos_score = (hbos_raw - h_min) / (h_denom + 1e-9)

        df_g["hbos_score"] = hbos_score

        # --- 4. COPOD ---
        copod = COPOD(contamination=self.contamination)
        copod.fit(Xs)
        copod_raw = copod.decision_function(Xs)

        c_min, c_max = copod_raw.min(), copod_raw.max()
        c_denom = c_max - c_min
        if c_denom <= 1e-9:
            copod_score = np.full_like(copod_raw, 0.5, dtype=float)
        else:
            copod_score = (copod_raw - c_min) / (c_denom + 1e-9)

        df_g["copod_score"] = copod_score

        # --- 5. Интегральный глобальный скор ---
        df_g["global_score"] = 0.5 * df_g["hbos_score"] + 0.5 * df_g["copod_score"]

        # можно добавить ранг, если понадобится дальше
        # df_g["global_behavior_rank"] = df_g["global_score"].rank(method="average", pct=True)

        scores = pd.DataFrame(
            {
                "txn_id": df_g["txn_id"].astype(str),
                "hbos_score": df_g["hbos_score"].values,
                "copod_score": df_g["copod_score"].values,
                "global_score": df_g["global_score"].values,
            }
        )

        return GlobalBehaviorResult(scores=scores)
