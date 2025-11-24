"""Time series helper for trend-aware scoring (daily TS anomalies)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesResult:
    scores: pd.DataFrame


class TimeSeriesTool:
    """
    Временной слой для улавливания аномалий по дневной активности ИНН.

    Логика соответствует ноутбучной ячейке:
    1) Строим дневные ряды по debit_inn и credit_inn:
       - daily_total_debit / daily_total_credit
       - daily_debit_transaction_count / daily_credit_transaction_count
       - in_out_ratio_30d
       - net_flow_ratio_7d
       - client_round_ratio_30d
    2) Для каждого ИНН:
       - берём только "активные" дни (есть движение по сумме),
       - считаем робастный z-score по нескольким метрикам,
       - агрегируем в raw TS-аномалию и нормируем в [0,1].
    3) Мержим TS-скор по дням назад к транзакциям.
       - ts_anomaly_score_debit, ts_anomaly_score_credit,
       - ts_anomaly_score_txn = max(debit, credit),
       - ts_anomaly_rank_txn (percent-rank),
       - ts_score = ts_anomaly_score_txn (для ScoreAggregator).
    """

    def __init__(self, min_active_days: int = 10) -> None:
        """
        min_active_days — минимальное количество активных дней по ИНН,
        чтобы считать TS-профиль; иначе TS-скор = 0.
        """
        self.min_active_days = min_active_days

    # ------------------------------------------------------------------
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # ------------------------------------------------------------------
    @staticmethod
    def _robust_zscore(series: pd.Series) -> pd.Series:
        """
        Робастный z-score относительно медианы и MAD.
        |z| = |x - median| / (MAD + eps), MAD = median(|x - median|).
        Возвращаем модуль, т.к. важна величина отклонения.
        """
        x = series.astype(float)
        med = x.median()
        mad = np.median(np.abs(x - med)) + 1e-6
        z = (x - med) / (mad * 1.4826)  # 1.4826 ~ калибровка под std
        return z.abs()

    def _compute_ts_anomaly_for_side(
        self,
        df_side: pd.DataFrame,
        inn_col: str,
        total_col: str,
        cnt_col: str,
        prefix: str,
    ) -> pd.DataFrame:
        """
        df_side: [inn_col, 'date', total_col, cnt_col,
                  'in_out_ratio', 'net_flow_ratio', 'client_round_ratio', 'is_active_day']

        Для каждого ИНН:
          - активные дни: is_active_day == 1
          - если активных дней < min_active_days → TS-скор = 0
          - иначе считаем робастные z-score по:
                total_col, cnt_col, in_out_ratio, net_flow_ratio, client_round_ratio
            агрегируем и нормируем в [0,1] внутри ИНН.
        """
        df_out = df_side.copy()
        raw_col = f"ts_anomaly_raw_{prefix}"
        score_col = f"ts_anomaly_score_{prefix}"

        df_out[raw_col] = 0.0
        df_out[score_col] = 0.0

        for _, idx in df_out.groupby(inn_col).indices.items():
            idx = np.asarray(idx, dtype=int)
            sub = df_out.iloc[idx]

            active_mask = (sub["is_active_day"] == 1)
            active_idx = idx[active_mask.values]

            # мало истории — TS-скор по ИНН = 0
            if active_idx.size < self.min_active_days:
                df_out.loc[idx, raw_col] = 0.0
                df_out.loc[idx, score_col] = 0.0
                continue

            sub_active = df_out.loc[active_idx]

            z_total = self._robust_zscore(sub_active[total_col])
            z_cnt = self._robust_zscore(sub_active[cnt_col])
            z_inout = self._robust_zscore(sub_active["in_out_ratio"])
            z_net = self._robust_zscore(sub_active["net_flow_ratio"])
            z_round = self._robust_zscore(sub_active["client_round_ratio"])

            raw = (z_total + z_cnt + z_inout + z_net + z_round) / 5.0

            r_min, r_max = raw.min(), raw.max()
            if r_max > r_min:
                score = (raw - r_min) / (r_max - r_min)
            else:
                score = pd.Series(0.0, index=raw.index)

            df_out.loc[active_idx, raw_col] = raw.values
            df_out.loc[active_idx, score_col] = score.values

            # неактивные дни считаем нейтральными
            inactive_idx = idx[~active_mask.values]
            if inactive_idx.size > 0:
                df_out.loc[inactive_idx, raw_col] = 0.0
                df_out.loc[inactive_idx, score_col] = 0.0

        return df_out

    # ------------------------------------------------------------------
    # ОСНОВНОЙ МЕТОД
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> TimeSeriesResult:
        """
        Ожидаемые колонки во входном df:
          - 'txn_id', 'date', 'amount'
          - 'debit_inn', 'credit_inn'
          - 'daily_total_debit', 'daily_total_credit'
          - 'daily_debit_transaction_count', 'daily_credit_transaction_count'
          - 'in_out_ratio_30d', 'net_flow_ratio_7d', 'client_round_ratio_30d'

        Возвращает TimeSeriesResult с колонками:
          - 'txn_id'
          - 'ts_anomaly_score_debit'
          - 'ts_anomaly_score_credit'
          - 'ts_anomaly_score_txn'  (max(debit, credit))
          - 'ts_anomaly_rank_txn'   (percent-rank)
          - 'ts_score'              (alias для ts_anomaly_score_txn)
        """
        required = ["txn_id", "date", "amount", "debit_inn", "credit_inn"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"TimeSeriesTool: missing required columns: {missing}")

        df_ts = df.copy()
        df_ts["date"] = pd.to_datetime(df_ts["date"]).dt.normalize()

        # подстраховка по важным фичам
        safe_zero_cols = [
            "daily_total_debit", "daily_total_credit",
            "daily_debit_transaction_count", "daily_credit_transaction_count",
            "in_out_ratio_30d", "net_flow_ratio_7d",
            "client_round_ratio_30d",
        ]
        for c in safe_zero_cols:
            if c not in df_ts.columns:
                df_ts[c] = 0.0

        df_ts["daily_total_debit"] = df_ts["daily_total_debit"].astype(float)
        df_ts["daily_total_credit"] = df_ts["daily_total_credit"].astype(float)
        df_ts["daily_debit_transaction_count"] = df_ts["daily_debit_transaction_count"].astype(float)
        df_ts["daily_credit_transaction_count"] = df_ts["daily_credit_transaction_count"].astype(float)
        df_ts["in_out_ratio_30d"] = df_ts["in_out_ratio_30d"].astype(float)
        df_ts["net_flow_ratio_7d"] = df_ts["net_flow_ratio_7d"].astype(float)
        df_ts["client_round_ratio_30d"] = df_ts["client_round_ratio_30d"].astype(float)

        # --------------------------------------------------------------
        # 1. Дневные ряды по дебету и кредиту
        # --------------------------------------------------------------
        if "debit_inn" not in df_ts.columns:
            raise ValueError("TimeSeriesTool: нет колонки 'debit_inn'.")
        if "credit_inn" not in df_ts.columns:
            raise ValueError("TimeSeriesTool: нет колонки 'credit_inn'.")

        # 1.1. Дебет: дневная активность по debit_inn
        deb_daily = (
            df_ts.groupby(["debit_inn", "date"], as_index=False)
            .agg(
                daily_total_debit=("daily_total_debit", "first"),
                daily_cnt_debit=("daily_debit_transaction_count", "first"),
                in_out_ratio=("in_out_ratio_30d", "first"),
                net_flow_ratio=("net_flow_ratio_7d", "first"),
                client_round_ratio=("client_round_ratio_30d", "first"),
            )
        )

        deb_daily["daily_total_debit"] = deb_daily["daily_total_debit"].astype(float)
        deb_daily["daily_cnt_debit"] = deb_daily["daily_cnt_debit"].astype(float)
        deb_daily["in_out_ratio"] = deb_daily["in_out_ratio"].astype(float)
        deb_daily["net_flow_ratio"] = deb_daily["net_flow_ratio"].astype(float)
        deb_daily["client_round_ratio"] = deb_daily["client_round_ratio"].astype(float)

        deb_daily["is_active_day"] = (deb_daily["daily_total_debit"].abs() > 0).astype(int)

        # 1.2. Кредит: дневная активность по credit_inn
        cred_daily = (
            df_ts.groupby(["credit_inn", "date"], as_index=False)
            .agg(
                daily_total_credit=("daily_total_credit", "first"),
                daily_cnt_credit=("daily_credit_transaction_count", "first"),
                in_out_ratio=("in_out_ratio_30d", "first"),
                net_flow_ratio=("net_flow_ratio_7d", "first"),
                client_round_ratio=("client_round_ratio_30d", "first"),
            )
        )

        cred_daily["daily_total_credit"] = cred_daily["daily_total_credit"].astype(float)
        cred_daily["daily_cnt_credit"] = cred_daily["daily_cnt_credit"].astype(float)
        cred_daily["in_out_ratio"] = cred_daily["in_out_ratio"].astype(float)
        cred_daily["net_flow_ratio"] = cred_daily["net_flow_ratio"].astype(float)
        cred_daily["client_round_ratio"] = cred_daily["client_round_ratio"].astype(float)

        cred_daily["is_active_day"] = (cred_daily["daily_total_credit"].abs() > 0).astype(int)

        # --------------------------------------------------------------
        # 2. TS-аномалии по сторонам
        # --------------------------------------------------------------
        deb_daily_ts = self._compute_ts_anomaly_for_side(
            df_side=deb_daily,
            inn_col="debit_inn",
            total_col="daily_total_debit",
            cnt_col="daily_cnt_debit",
            prefix="debit",
        )

        cred_daily_ts = self._compute_ts_anomaly_for_side(
            df_side=cred_daily,
            inn_col="credit_inn",
            total_col="daily_total_credit",
            cnt_col="daily_cnt_credit",
            prefix="credit",
        )

        # --------------------------------------------------------------
        # 3. Перенос TS-анализa на транзакции
        # --------------------------------------------------------------
        df_ts = df_ts.merge(
            deb_daily_ts[["debit_inn", "date", "ts_anomaly_score_debit"]],
            on=["debit_inn", "date"],
            how="left",
        )

        df_ts = df_ts.merge(
            cred_daily_ts[["credit_inn", "date", "ts_anomaly_score_credit"]],
            on=["credit_inn", "date"],
            how="left",
        )

        df_ts["ts_anomaly_score_debit"] = df_ts["ts_anomaly_score_debit"].fillna(0.0)
        df_ts["ts_anomaly_score_credit"] = df_ts["ts_anomaly_score_credit"].fillna(0.0)

        df_ts["ts_anomaly_score_txn"] = df_ts[
            ["ts_anomaly_score_debit", "ts_anomaly_score_credit"]
        ].max(axis=1)

        # ранговый TS-скор по всей выборке: 0..1, где 1 — верхние TS-анomalies
        df_ts["ts_anomaly_rank_txn"] = df_ts["ts_anomaly_score_txn"].rank(
            method="average", pct=True
        )

        # алиас для совместимости с ScoreAggregator (ожидает 'ts_score')
        df_ts["ts_score"] = df_ts["ts_anomaly_score_txn"]

        scores = df_ts[
            [
                "txn_id",
                "ts_anomaly_score_debit",
                "ts_anomaly_score_credit",
                "ts_anomaly_score_txn",
                "ts_anomaly_rank_txn",
                "ts_score",
            ]
        ].copy()
        scores["txn_id"] = scores["txn_id"].astype(str)

        return TimeSeriesResult(scores=scores)
