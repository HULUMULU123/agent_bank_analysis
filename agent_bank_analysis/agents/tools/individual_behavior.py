"""Individual behavior analysis tool using IsolationForest (per-INN)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class IndividualBehaviorResult:
    scores: pd.DataFrame


class IndividualBehaviorTool:
    """
    Индивидуальные поведенческие аномалии по ИНН.

    Логика приближена к ноутбуку:
    - отдельная модель IsolationForest для КАЖДОГО ИНН и КАЖДОЙ роли (debit/credit);
    - набор расширенных фич (как в ноутбуке: amount, roll_*, spikes, текст, календарь, глобальные фичи и т.п.);
    - адаптивный contamination по количеству транзакций по ИНН;
    - нормировка score внутри ИНН в [0, 1];
    - для транзакции общий behavior_score = max(iforest_score_debit, iforest_score_credit).
    """

    def __init__(
        self,
        random_state: int = 42,
        min_tx_per_inn: int = 10,
        min_contamination: float = 0.02,
        max_contamination: float = 0.20,
    ) -> None:
        self.random_state = random_state
        self.min_tx_per_inn = min_tx_per_inn
        self.min_contamination = min_contamination
        self.max_contamination = max_contamination

    # ----------------------- ВСПОМОГАТЕЛЬНОЕ -----------------------

    def _adaptive_contamination(
        self,
        n_samples: int,
        min_anom: int = 3,
        max_anom: int = 10,
    ) -> float:
        """
        Адаптивный уровень contamination:
        - хотим 3–10 аномалий на ИНН,
        - но в пределах [min_contamination, max_contamination].
        """
        if n_samples <= 0:
            return self.min_contamination

        target = max(min_anom / n_samples, self.min_contamination)
        target = min(target, max_anom / n_samples)
        return float(np.clip(target, self.min_contamination, self.max_contamination))

    def _prepare_side_features(self, table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Приводим таблицу роли (debit/credit) к формату, максимально близкому к ноутбуку.

        - добавляем при необходимости log_amount;
        - считаем log_days_since_last (если нет);
        - считаем rel_amount_to_mean30d (если нет);
        - гарантируем наличие всех фич из ноутбука, заполняя отсутствующие нулями.

        Ожидается, что table получен из FeatureEngineer._split_roles
        и уже содержит дебетовые/кредитовые признаки.
        """
        tbl = table.copy()

        # amount / log_amount
        if "amount" not in tbl.columns:
            if "log_amount" in tbl.columns:
                tbl["amount"] = np.expm1(tbl["log_amount"]).clip(lower=0)
            else:
                tbl["amount"] = 0.0

        if "log_amount" not in tbl.columns:
            tbl["log_amount"] = np.log1p(tbl["amount"].astype(float).clip(lower=0))

        # log_days_since_last
        if "log_days_since_last" not in tbl.columns:
            if "days_since_last_txn" in tbl.columns:
                tbl["log_days_since_last"] = np.log1p(
                    tbl["days_since_last_txn"].astype(float)
                )
            else:
                tbl["log_days_since_last"] = 0.0

        # относительная сумма к среднему за 30 дней
        if "rel_amount_to_mean30d" not in tbl.columns:
            if "roll_mean_30d" in tbl.columns:
                rel = tbl["amount"].astype(float) / (
                    tbl["roll_mean_30d"].replace(0, np.nan) + 1e-6
                )
                tbl["rel_amount_to_mean30d"] = (
                    rel.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                )
            else:
                tbl["rel_amount_to_mean30d"] = 0.0

        # daily_txn_cnt: приводим к имени из ноутбука
        if "daily_txn_cnt" not in tbl.columns:
            if "daily_txn_count" in tbl.columns:
                tbl["daily_txn_cnt"] = tbl["daily_txn_count"].astype(float)
            else:
                tbl["daily_txn_cnt"] = 0.0

        # фичи как в ноутбуке (feat_cols)
        feature_cols = [
            "amount",
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
            "daily_txn_cnt",
            "in_out_ratio_30d",
            "net_flow_ratio_7d",
            "transit_same_day_flag",
            "round_large_amount",
            "client_round_ratio_30d",
            "purpose_len",
            "purpose_digits_ratio",
            "purpose_upper_ratio",
            "is_weekend",
            "is_month_end",
            "is_quarter_end",
            "is_year_end",
            "log_days_since_last",
            "rel_amount_to_mean30d",
        ]

        # гарантируем наличие всех колонок (если чего-то нет — забиваем нулями)
        for c in feature_cols:
            if c not in tbl.columns:
                tbl[c] = 0.0

        # приводим к float и убираем NaN/inf
        X = (
            tbl[feature_cols]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .astype(float)
        )
        tbl[feature_cols] = X

        return tbl, feature_cols

    def _fit_side(self, table: pd.DataFrame, side: str) -> pd.Series:
        """
        Обучаем отдельный IsolationForest на КАЖДЫЙ ИНН внутри одной роли
        (debit / credit), как в ноутбуке.
        Возвращаем pd.Series с индексом table.index и значениями [0,1].
        """
        if "inn" not in table.columns:
            raise ValueError(f"IndividualBehaviorTool: expected 'inn' column for side={side}.")

        tbl, feature_cols = self._prepare_side_features(table)

        # итоговый скор по всем строкам данной роли
        scores = pd.Series(index=tbl.index, dtype=float, name=f"iforest_score_{side}")

        # проходим по каждому ИНН
        for inn, sub in tbl.groupby("inn"):
            n = len(sub)
            if n < self.min_tx_per_inn:
                continue  # мало транзакций — не учим модель, скор остаётся NaN

            X = sub[feature_cols].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            cont = self._adaptive_contamination(n)
            model = IsolationForest(
                n_estimators=300,
                max_samples="auto",
                contamination=cont,
                random_state=self.random_state,
                bootstrap=False,
                n_jobs=-1,
            )
            model.fit(Xs)

            # score_samples: чем ниже, тем более аномально → берём -score
            raw = -model.score_samples(Xs)  # положительное выше = аномальнее
            # нормировка внутри ИНН в [0,1]
            s_norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

            scores.loc[sub.index] = s_norm

        return scores

    # ----------------------- PUBLIC API -----------------------

    def run(
        self,
        debit_table: pd.DataFrame,
        credit_table: pd.DataFrame,
    ) -> IndividualBehaviorResult:
        """
        Ожидается, что debit_table и credit_table получены из FeatureEngineer._split_roles
        и содержат одинаковый порядок txn_id (по одной строке на роль на каждую операцию).
        """
        # отдельные скоринги по ролям
        debit_score = self._fit_side(debit_table, side="debit")
        credit_score = self._fit_side(credit_table, side="credit")

        # собираем в один df по txn_id
        scores = pd.DataFrame(
            {
                "txn_id": debit_table["txn_id"].astype(str),
                "iforest_score_debit": debit_score.values,
                "iforest_score_credit": credit_score.values,
            }
        )

        # единый индивидуальный скор как максимум по ролям
        scores["behavior_score"] = scores[
            ["iforest_score_debit", "iforest_score_credit"]
        ].max(axis=1)

        return IndividualBehaviorResult(scores=scores)
