"""Feature engineering extracted from the original notebook."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .config import SAFE_ZERO_COLS


@dataclass
class FeatureEngineeringResult:
    """Container for feature engineering outputs."""

    dataframe: pd.DataFrame
    debit_table: pd.DataFrame
    credit_table: pd.DataFrame


class FeatureEngineer:
    """Generate features for individual and global tools.

    The logic mirrors the notebook but is packaged as a reusable class.
    """

    def __init__(self, text_patterns: Iterable[str] | None = None) -> None:
        self.text_patterns = list(text_patterns) if text_patterns else [
            r"\bзайм\w*\b", r"\bдолг\w*\b", r"\bобнал\w*\b", r"\bналич\w*\b",
            r"\bперевод\w*\W*карт", r"\bвознагражд\w*\b", r"\bагентск\w*\b",
            r"\bдарен\w*\b", r"\bкрипт\w*\b", r"\bbtc\b", r"\busdt\b",
        ]
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    @staticmethod
    def _clean_text(value: str) -> str:
        value = str(value).lower()
        value = re.sub(r"[^a-zа-я0-9\s]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def _apply_safe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in SAFE_ZERO_COLS:
            if column not in df.columns:
                df[column] = 0.0
        return df

    def _prepare_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["amount"] = df["amount"].astype(float)
        df = self._apply_safe_columns(df)

        if "log_amount" not in df.columns:
            df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

        df["log_days_since_last_txn_debit"] = np.log1p(
            df["days_since_last_txn_debit"].astype(float)
        )
        df["log_days_since_last_txn_credit"] = np.log1p(
            df["days_since_last_txn_credit"].astype(float)
        )
        return df

    def _split_roles(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        debit = pd.DataFrame(
            {
                "txn_id": df["txn_id"],
                "date": df["date"],
                "inn": df["debit_inn"].astype(str),
                "side": "debit",
                "purpose": df["purpose"].astype(str),
                "amount": df["amount"].astype(float),
                "log_amount": df["log_amount"].astype(float),
                "roll_cnt_30d": df["debit_roll_cnt_30d"].astype(float),
                "roll_mean_30d": df["debit_roll_mean_30d"].astype(float),
                "roll_std_30d": df["debit_roll_std_30d"].astype(float),
                "amount_spike_ratio_7d": df["debit_amount_spike_ratio_7d"].astype(float),
                "tx_rate_spike_7d": df["debit_tx_rate_spike_7d"].astype(float),
                "amount_volatility_30d": df["debit_amount_volatility_30d"].astype(float),
                "accel_ratio_7d": df["debit_accel_ratio_7d"].astype(float),
                "to_p95_ratio": df["debit_to_p95_ratio"].astype(float),
                "new_counterparty_ratio": df["debit_new_counterparty_ratio"].astype(float),
                "fan_ratio": df["debit_fan_out_ratio"].astype(float),
                "volatility_z": df["debit_volatility_z"].astype(float),
                "daily_total": df["daily_total_debit"].astype(float),
                "daily_txn_count": df["daily_debit_transaction_count"].astype(float),
                "daily_percent": df.get("daily_debit_percent", 0).astype(float),
                "days_since_last_txn": df["days_since_last_txn_debit"].astype(float),
            }
        )
        credit = pd.DataFrame(
            {
                "txn_id": df["txn_id"],
                "date": df["date"],
                "inn": df["credit_inn"].astype(str),
                "side": "credit",
                "purpose": df["purpose"].astype(str),
                "amount": df["amount"].astype(float),
                "log_amount": df["log_amount"].astype(float),
                "roll_cnt_30d": df["credit_roll_cnt_30d"].astype(float),
                "roll_mean_30d": df["credit_roll_mean_30d"].astype(float),
                "roll_std_30d": df["credit_roll_std_30d"].astype(float),
                "amount_spike_ratio_7d": df["credit_amount_spike_ratio_7d"].astype(float),
                "tx_rate_spike_7d": df["credit_tx_rate_spike_7d"].astype(float),
                "amount_volatility_30d": df["credit_amount_volatility_30d"].astype(float),
                "accel_ratio_7d": df["credit_accel_ratio_7d"].astype(float),
                "to_p95_ratio": df["credit_to_p95_ratio"].astype(float),
                "new_counterparty_ratio": df["credit_new_counterparty_ratio"].astype(float),
                "fan_ratio": df["credit_fan_in_ratio"].astype(float),
                "volatility_z": df["credit_volatility_z"].astype(float),
                "daily_total": df["daily_total_credit"].astype(float),
                "daily_txn_count": df["daily_credit_transaction_count"].astype(float),
                "daily_percent": df.get("daily_credit_percent", 0).astype(float),
                "days_since_last_txn": df["days_since_last_txn_credit"].astype(float),
            }
        )
        return debit, credit

    def _text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pattern = re.compile("|".join(self.text_patterns), flags=re.IGNORECASE)
        df["purpose_clean"] = df["purpose"].astype(str).apply(self._clean_text)
        df["purpose_stopword_high"] = df["purpose_clean"].str.contains(pattern, na=False)

        texts: List[str] = df["purpose_clean"].tolist()
        self._tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
        tfidf_matrix = self._tfidf.fit_transform(texts)

        k = min(50, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=max(k, 1), random_state=42)
        svd_matrix = self._svd.fit_transform(tfidf_matrix)

        for idx in range(svd_matrix.shape[1]):
            df[f"purpose_svd_{idx+1}"] = svd_matrix[:, idx]
        return df

    def transform(self, df: pd.DataFrame) -> FeatureEngineeringResult:
        df = self._prepare_base(df)
        df = self._text_features(df)
        debit, credit = self._split_roles(df)
        return FeatureEngineeringResult(dataframe=df, debit_table=debit, credit_table=credit)
