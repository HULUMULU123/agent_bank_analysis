"""Feature engineering extracted from the original notebook."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import SAFE_ZERO_COLS  # если больше не нужен – можно удалить импорт


@dataclass
class FeatureEngineeringResult:
    """Container for feature engineering outputs."""

    dataframe: pd.DataFrame
    debit_table: pd.DataFrame
    credit_table: pd.DataFrame


class FeatureEngineer:
    """Generate features for individual and global tools.

    The logic mirrors the original notebook but is packaged as a reusable class.
    """

    def __init__(self, text_patterns: Iterable[str] | None = None) -> None:
        self.text_patterns = list(text_patterns) if text_patterns else [
            r"\bзайм\w*\b", r"\bдолг\w*\b", r"\bобнал\w*\b", r"\bналич\w*\b",
            r"\bперевод\w*\W*карт", r"\bвознагражд\w*\b", r"\bагентск\w*\b",
            r"\bдарен\w*\b", r"\bкрипт\w*\b", r"\bbtc\b", r"\busdt\b",
        ]
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    # ------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_text(value: str) -> str:
        value = str(value).lower()
        value = re.sub(r"[^a-zа-я0-9\s]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _make_txn_id(row: pd.Series) -> str:
        parts = [
            str(row.get("date", "")),
            str(row.get("debit_inn", "")),
            str(row.get("credit_inn", "")),
            str(row.get("amount", "")),
            str(row.get("purpose", "")),
        ]
        raw = "||".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 0. БАЗОВАЯ ПОДГОТОВКА
    # ------------------------------------------------------------------
    def _prepare_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # дата
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # суммы
        for col in ["debit_amount", "credit_amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0

        if "amount" not in df.columns:
            df["amount"] = df["debit_amount"].fillna(df["credit_amount"])
        df["debit_amount"] = df["debit_amount"].fillna(0.0)
        df["credit_amount"] = df["credit_amount"].fillna(0.0)

        # txn_id
        if "txn_id" not in df.columns:
            df["txn_id"] = df.apply(self._make_txn_id, axis=1)

        df = df.sort_values(["date", "txn_id"]).reset_index(drop=True)

        # базовые календарные фичи
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        df["month"] = df["date"].dt.month
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

        # лог-сумма и "круглость"
        df["log_amount"] = np.log1p(df["amount"].clip(lower=1))
        df["is_round_10k"] = (df["amount"] % 10000 == 0).astype(int)
        df["is_round_100k"] = (df["amount"] % 100000 == 0).astype(int)
        df["is_round_large"] = (
            (df["is_round_10k"] == 1) | (df["is_round_100k"] == 1)
        ).astype(int)

        # текстовые размеры
        df["purpose_len"] = df["purpose"].astype(str).str.len()
        df["purpose_digits_ratio"] = (
            df["purpose"].astype(str).str.count(r"\d") / (df["purpose_len"] + 1e-6)
        )
        df["purpose_upper_ratio"] = (
            df["purpose"].astype(str).str.count(r"[A-ZА-Я]") / (df["purpose_len"] + 1e-6)
        )

        return df

    # ------------------------------------------------------------------
    # 2. СУТОЧНЫЕ АГРЕГАТЫ
    # ------------------------------------------------------------------
    @staticmethod
    def _add_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
        df["daily_total_debit"] = df.groupby(
            ["debit_inn", "date"]
        )["amount"].transform("sum")
        df["daily_total_credit"] = df.groupby(
            ["credit_inn", "date"]
        )["amount"].transform("sum")

        df["daily_debit_txn_count"] = df.groupby(
            ["debit_inn", "date"]
        )["amount"].transform(lambda s: (s > 0).sum())
        df["daily_credit_txn_count"] = df.groupby(
            ["credit_inn", "date"]
        )["amount"].transform(lambda s: (s > 0).sum())

        df["daily_debit_percent"] = (
            df["amount"] / df["daily_total_debit"].replace(0, np.nan)
        ).fillna(0.0)
        df["daily_credit_percent"] = (
            df["amount"] / df["daily_total_credit"].replace(0, np.nan)
        ).fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # 3. ИНТЕРВАЛЫ МЕЖДУ ОПЕРАЦИЯМИ
    # ------------------------------------------------------------------
    @staticmethod
    def _add_intervals(df: pd.DataFrame) -> pd.DataFrame:
        df["days_since_last_db"] = (
            df.groupby("debit_inn")["date"].diff().dt.days.fillna(9999)
        )
        df["days_since_last_cr"] = (
            df.groupby("credit_inn")["date"].diff().dt.days.fillna(9999)
        )
        return df

    # ------------------------------------------------------------------
    # 4. РОЛЛИНГИ
    # ------------------------------------------------------------------
    @staticmethod
    def _add_rolling_side(
        df: pd.DataFrame,
        side: str,
        amt_col: str,
        windows: Tuple[int, ...] = (7, 14, 30, 90),
    ) -> pd.DataFrame:
        inn_col = f"{side}_inn"
        parts: List[pd.DataFrame] = []

        need = df[[inn_col, "date", amt_col]].copy()
        need[amt_col] = need[amt_col].fillna(0.0)

        for inn, sub in need.groupby(inn_col):
            daily = (
                sub.groupby("date")[amt_col]
                .sum()
                .to_frame("amt_day")
                .sort_index()
            )
            idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
            daily = daily.reindex(idx, fill_value=0.0)
            daily["tx_day"] = (daily["amt_day"] > 0).astype(int)
            daily.index.name = "date"

            for W in windows:
                daily[f"{side}_roll_sum_{W}d"] = daily["amt_day"].rolling(
                    W, min_periods=1
                ).sum()
                daily[f"{side}_roll_cnt_{W}d"] = daily["tx_day"].rolling(
                    W, min_periods=1
                ).sum()
                daily[f"{side}_roll_mean_{W}d"] = daily["amt_day"].rolling(
                    W, min_periods=1
                ).mean()
                daily[f"{side}_roll_std_{W}d"] = (
                    daily["amt_day"]
                    .rolling(W, min_periods=1)
                    .std()
                    .fillna(0.0)
                )
                daily[f"{side}_roll_p95_{W}d"] = daily["amt_day"].rolling(
                    W, min_periods=1
                ).quantile(0.95)

            daily = daily.reset_index()
            daily[inn_col] = inn
            parts.append(daily)

        rolls = pd.concat(parts, ignore_index=True)
        df = df.merge(rolls, on=["date", inn_col], how="left")
        return df

    # ------------------------------------------------------------------
    # 5. ВСПЛЕСКИ И УСКОРЕНИЯ
    # ------------------------------------------------------------------
    @staticmethod
    def _add_spikes_and_accel(df: pd.DataFrame) -> pd.DataFrame:
        df["debit_amount_spike_7d"] = df["debit_roll_sum_7d"] / (
            df["debit_roll_sum_30d"] / 4 + 1e-6
        )
        df["credit_amount_spike_7d"] = df["credit_roll_sum_7d"] / (
            df["credit_roll_sum_30d"] / 4 + 1e-6
        )

        df["debit_tx_spike_7d"] = df["debit_roll_cnt_7d"] / (
            df["debit_roll_cnt_30d"] / 4 + 1e-6
        )
        df["credit_tx_spike_7d"] = df["credit_roll_cnt_7d"] / (
            df["credit_roll_cnt_30d"] / 4 + 1e-6
        )

        df["debit_accel"] = (
            (df["debit_roll_sum_7d"] - df["debit_roll_sum_14d"])
            / (df["debit_roll_sum_14d"] + 1e-6)
        ).clip(-10, 10)
        df["credit_accel"] = (
            (df["credit_roll_sum_7d"] - df["credit_roll_sum_14d"])
            / (df["credit_roll_sum_14d"] + 1e-6)
        ).clip(-10, 10)

        df["debit_to_p95_90"] = df["debit_roll_sum_7d"] / (
            df["debit_roll_p95_90d"] + 1e-6
        )
        df["credit_to_p95_90"] = df["credit_roll_sum_7d"] / (
            df["credit_roll_p95_90d"] + 1e-6
        )

        return df

    # ------------------------------------------------------------------
    # 6. НОВИЗНА КОНТРАГЕНТОВ
    # ------------------------------------------------------------------
    @staticmethod
    def _add_new_counterparty(df: pd.DataFrame) -> pd.DataFrame:
        def rolling_new_counterparty(gr: pd.DataFrame) -> pd.Series:
            seen = set()
            flags = []
            for c in gr["credit_inn"]:
                flags.append(0 if c in seen else 1)
                seen.add(c)
            return (
                pd.Series(flags, index=gr.index)
                .rolling(30, min_periods=1)
                .mean()
            )

        df["debit_new_counterparty_30d"] = (
            df.sort_values(["debit_inn", "date", "txn_id"])
            .groupby("debit_inn", group_keys=False)
            .apply(rolling_new_counterparty)
            .fillna(0.0)
        )
        return df

    # ------------------------------------------------------------------
    # 7. ТРАНЗИТ И NET-FLOW
    # ------------------------------------------------------------------
    @staticmethod
    def _add_transit_features(df: pd.DataFrame) -> pd.DataFrame:
        df["transit_flag_same_day"] = (
            (df["days_since_last_cr"] == 0)
            & (df["daily_total_credit"] > 0)
            & (df["daily_total_debit"] > 0)
        ).astype(int)

        df["net_flow_ratio_7d"] = (
            (df["credit_roll_sum_7d"] - df["debit_roll_sum_7d"])
            / (df["credit_roll_sum_7d"] + df["debit_roll_sum_7d"] + 1e-6)
        ).clip(-1, 1)

        return df

    # ------------------------------------------------------------------
    # 8. ПРОФИЛЬ ИНН
    # ------------------------------------------------------------------
    @staticmethod
    def _build_entity_profile(
        df: pd.DataFrame,
        inn_col: str,
        amt_col: str,
    ) -> pd.DataFrame:
        prof = df.groupby(inn_col).agg(
            total_amount=(amt_col, "sum"),
            mean_amount=(amt_col, "mean"),
            std_amount=(amt_col, "std"),
            median_amount=(amt_col, "median"),
            unique_partners=(
                "credit_inn" if inn_col == "debit_inn" else "debit_inn",
                "nunique",
            ),
            txn_count=(amt_col, lambda s: (s > 0).sum()),
            active_days=("date", "nunique"),
        ).fillna(0.0)
        prof.columns = [f"{inn_col}_{c}" for c in prof.columns]
        return prof

    @staticmethod
    def _add_entity_profiles(df: pd.DataFrame) -> pd.DataFrame:
        prof_debit = FeatureEngineer._build_entity_profile(
            df, "debit_inn", "debit_amount"
        )
        prof_credit = FeatureEngineer._build_entity_profile(
            df, "credit_inn", "credit_amount"
        )

        df = df.merge(prof_debit, on="debit_inn", how="left")
        df = df.merge(prof_credit, on="credit_inn", how="left")
        return df

    # ------------------------------------------------------------------
    # 9. ТЕКСТОВЫЕ ПРИЗНАКИ
    # ------------------------------------------------------------------
    def _text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pattern = re.compile("|".join(self.text_patterns), flags=re.IGNORECASE)

        df["purpose_clean"] = df["purpose"].astype(str).apply(self._clean_text)
        df["purpose_stopword_high"] = df["purpose_clean"].str.contains(
            pattern, na=False
        )

        texts: List[str] = df["purpose_clean"].tolist()
        self._tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
        tfidf_matrix = self._tfidf.fit_transform(texts)

        k = min(50, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        k = max(k, 1)
        self._svd = TruncatedSVD(n_components=k, random_state=42)
        svd_matrix = self._svd.fit_transform(tfidf_matrix)

        for idx in range(svd_matrix.shape[1]):
            df[f"purpose_svd_{idx + 1}"] = svd_matrix[:, idx]

        return df

    # ------------------------------------------------------------------
    # SPLIT: DEBIT / CREDIT VIEW
    # ------------------------------------------------------------------
    def _split_roles(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Общие для обеих сторон
        svd_cols = [c for c in df.columns if c.startswith("purpose_svd_")]

        shared_cols = [
            "txn_id",
            "date",
            "purpose",
            "amount",
            "log_amount",
            "day_of_week",
            "is_weekend",
            "is_month_end",
            "month",
            "weekofyear",
            "is_round_10k",
            "is_round_100k",
            "is_round_large",
            "purpose_len",
            "purpose_digits_ratio",
            "purpose_upper_ratio",
            "purpose_clean",
            "purpose_stopword_high",
            "transit_flag_same_day",
            "net_flow_ratio_7d",
        ]

        # DEBIT
        debit = pd.DataFrame(
            {
                "txn_id": df["txn_id"],
                "date": df["date"],
                "inn": df["debit_inn"].astype(str),
                "side": "debit",
                "purpose": df["purpose"].astype(str),
                "amount": df["amount"].astype(float),
                "log_amount": df["log_amount"].astype(float),
                "daily_total": df["daily_total_debit"].astype(float),
                "daily_txn_count": df["daily_debit_txn_count"].astype(float),
                "daily_percent": df["daily_debit_percent"].astype(float),
                "days_since_last_txn": df["days_since_last_db"].astype(float),
            }
        )

        # CREDIT
        credit = pd.DataFrame(
            {
                "txn_id": df["txn_id"],
                "date": df["date"],
                "inn": df["credit_inn"].astype(str),
                "side": "credit",
                "purpose": df["purpose"].astype(str),
                "amount": df["amount"].astype(float),
                "log_amount": df["log_amount"].astype(float),
                "daily_total": df["daily_total_credit"].astype(float),
                "daily_txn_count": df["daily_credit_txn_count"].astype(float),
                "daily_percent": df["daily_credit_percent"].astype(float),
                "days_since_last_txn": df["days_since_last_cr"].astype(float),
            }
        )

        # общие признаки (кроме уже добавленных) + SVD
        for col in shared_cols:
            if col not in debit.columns and col in df.columns:
                debit[col] = df[col]
            if col not in credit.columns and col in df.columns:
                credit[col] = df[col]

        for col in svd_cols:
            debit[col] = df[col]
            credit[col] = df[col]

        # side-specific: все колонки вида debit_* / credit_*, но уже БЕЗ префикса
        for col in df.columns:
            if col.startswith("debit_"):
                base = col[len("debit_") :]
                debit[base] = df[col]
            if col.startswith("credit_"):
                base = col[len("credit_") :]
                credit[base] = df[col]

        # entity profile: debit_inn_* и credit_inn_* – кладём по стороне
        for col in df.columns:
            if col.startswith("debit_inn_"):
                debit[col] = df[col]
            if col.startswith("credit_inn_"):
                credit[col] = df[col]

        return debit, credit

    # ------------------------------------------------------------------
    # ПУБЛИЧНЫЙ МЕТОД
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> FeatureEngineeringResult:
        df = self._prepare_base(df)
        df = self._add_daily_aggregates(df)
        df = self._add_intervals(df)

        windows = (7, 14, 30, 90)
        df = self._add_rolling_side(df, "debit", "debit_amount", windows=windows)
        df = self._add_rolling_side(df, "credit", "credit_amount", windows=windows)

        df = self._add_spikes_and_accel(df)
        df = self._add_new_counterparty(df)
        df = self._add_transit_features(df)
        df = self._add_entity_profiles(df)
        df = self._text_features(df)

        debit, credit = self._split_roles(df)

        return FeatureEngineeringResult(
            dataframe=df,
            debit_table=debit,
            credit_table=credit,
        )
