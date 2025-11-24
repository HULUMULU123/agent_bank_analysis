"""Combine scores from all tools into a unified dataframe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import sqlite3

from .memory import KB_PATH  # используем тот же путь к БД, что и в memory.py


@dataclass
class AggregationResult:
    dataframe: pd.DataFrame


def _get_col_safe(frame: pd.DataFrame, col: str, default_value: float = 0.0) -> pd.Series:
    """
    Берём колонку как float, все NaN внутри считаем «нет сигнала»
    и заменяем на default_value.
    Если колонки нет — возвращаем константу default_value.
    """
    if col in frame.columns:
        s = pd.to_numeric(frame[col], errors="coerce")
        return s.fillna(default_value).astype(float)
    return pd.Series(default_value, index=frame.index, dtype=float)


def _history_maps(db_path: str) -> tuple[dict, dict, dict, dict]:
    """
    Читает kb_inn_stats и возвращает четыре словаря:
      - hist_deb_map:  inn -> history_risk (debit)
      - hist_crd_map:  inn -> history_risk (credit)
      - cnt_deb_map:   inn -> tx_count (debit)
      - cnt_crd_map:   inn -> tx_count (credit)
    Если таблицы/данных нет — возвращает пустые словари.
    """
    try:
        with sqlite3.connect(db_path) as con:
            stats = pd.read_sql_query(
                "SELECT inn, role, tx_count, history_risk FROM kb_inn_stats",
                con,
            )
    except Exception:
        # БД ещё нет или таблицы нет — считаем, что истории нет
        return {}, {}, {}, {}

    if stats.empty:
        return {}, {}, {}, {}

    pivot_hist = stats.pivot(index="inn", columns="role", values="history_risk")
    pivot_cnt = stats.pivot(index="inn", columns="role", values="tx_count")

    hist_deb = (
        pivot_hist["debit"].dropna().astype(float).to_dict()
        if "debit" in pivot_hist.columns
        else {}
    )
    hist_crd = (
        pivot_hist["credit"].dropna().astype(float).to_dict()
        if "credit" in pivot_hist.columns
        else {}
    )
    cnt_deb = (
        pivot_cnt["debit"].dropna().astype(int).to_dict()
        if "debit" in pivot_cnt.columns
        else {}
    )
    cnt_crd = (
        pivot_cnt["credit"].dropna().astype(int).to_dict()
        if "credit" in pivot_cnt.columns
        else {}
    )

    return hist_deb, hist_crd, cnt_deb, cnt_crd


class ScoreAggregator:
    """
    Агрегатор, повторяющий логику финальной ячейки:

    1) Дедуп по txn_id.
    2) Выбор между global_behavior_score и локальными IsolationForest (iforest_score_*),
       с адаптивным доверием alpha_debit/alpha_credit → base_behavior_risk.
    3) Корректоры: semantic_risk_score, ts_anomaly_score_txn, graph_edge/node →
       correction_score и txn_risk_score / txn_risk_rank.
    4) Переблендинг: base (txn_risk_score) + граф + история ИНН из kb_inn_stats
       → overall_risk и overall_risk_level.
    """

    def __init__(
        self,
        kb_path: str = KB_PATH,
        min_tx_per_inn: int = 10,
        gamma_correction: float = 0.5,
        min_base_weight: float = 0.60,
        graph_weight_cap: float = 0.25,
        hist_cap: float = 0.15,
        hist_hyper: int = 200,
    ) -> None:
        self.kb_path = kb_path
        self.MIN_TX_PER_INN = min_tx_per_inn
        self.GAMMA = float(gamma_correction)

        # параметры переблендинга base / graph / history
        self.MIN_BASE_WEIGHT = float(min_base_weight)
        self.GRAPH_WEIGHT_CAP = float(graph_weight_cap)
        self.HIST_CAP = float(hist_cap)
        self.H_HYPER = int(hist_hyper)

    # ------------------------------------------------------------------
    # ОСНОВНОЙ МЕТОД
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> AggregationResult:
        """
        На вход ожидается df, в котором уже есть:
          - txn_id, date, debit_inn, credit_inn, amount, purpose
          - hbos_score, copod_score, global_behavior_score (или хотя бы hbos/copod)
          - iforest_score_debit, iforest_score_credit
          - semantic_risk_score (по тексту)
          - ts_anomaly_score_txn (TS-аномалии)
          - graph_edge_score, graph_node_score (графовый слой)

        На выходе:
          - df с добавленными колонками:
              global_score,
              alpha_debit, alpha_credit,
              behavior_risk_debit, behavior_risk_credit,
              base_behavior_risk,
              correction_score,
              txn_risk_score, txn_risk_rank,
              history_risk_debit, history_risk_credit, history_risk_combined,
              overall_risk, overall_risk_level.
        """
        df_int = df.copy()

        # --- -1. Защита от раздувания: схлопываем до 1 строки на txn_id ---
        if "txn_id" in df_int.columns:
            # числовые и нечисловые колонки
            num_cols = df_int.select_dtypes(include=[np.number]).columns.tolist()
            if "txn_id" in num_cols:
                num_cols.remove("txn_id")

            cat_cols = [c for c in df_int.columns if c not in num_cols and c != "txn_id"]

            agg_dict: dict[str, str] = {}
            for c in num_cols:
                agg_dict[c] = "max"   # для числовых — максимум (по скорам это безопасно)
            for c in cat_cols:
                agg_dict[c] = "first"  # для категориальных — первая попавшаяся

            df_int = (
                df_int
                .sort_values(["txn_id"])
                .groupby("txn_id", as_index=False)
                .agg(agg_dict)
                .reset_index(drop=True)
            )
        # если нет txn_id — просто работаем как есть

        # --- 0. Параметры и защита ---
        MIN_TX_PER_INN = self.MIN_TX_PER_INN

        # --- 1. Глобальный поведенческий скор (общая модель) ---

        if "global_behavior_score" in df_int.columns:
            # если есть готовый глобальный скор – просто чистим и используем
            g = df_int["global_behavior_score"].astype(float)
        else:
            # иначе считаем как среднее HBOS и COPOD
            hbos = (
                df_int["hbos_score"].astype(float)
                if "hbos_score" in df_int.columns
                else pd.Series(np.nan, index=df_int.index)
            )
            copod = (
                df_int["copod_score"].astype(float)
                if "copod_score" in df_int.columns
                else pd.Series(np.nan, index=df_int.index)
            )
            g = (hbos + copod) / 2.0

        # глобальное среднее по тем, где что-то есть
        global_mean = float(np.nanmean(g)) if not np.all(np.isnan(g)) else 0.5

        # подстраховка: там, где g NaN, ставим global_mean
        g = g.fillna(global_mean).clip(0, 1)
        df_int["global_score"] = g
        glob = df_int["global_score"]

        # --- 2. Индивидуальные скоринги IsolationForest (по ИНН) ---

        if "iforest_score_debit" not in df_int.columns:
            df_int["iforest_score_debit"] = np.nan
        if "iforest_score_credit" not in df_int.columns:
            df_int["iforest_score_credit"] = np.nan

        df_int["iforest_score_debit"] = df_int["iforest_score_debit"].astype(float)
        df_int["iforest_score_credit"] = df_int["iforest_score_credit"].astype(float)

        has_iforest_deb = df_int["iforest_score_debit"].notna()
        has_iforest_cred = df_int["iforest_score_credit"].notna()

        # --- 3. Считаем доверие к индивидуальной модели: alpha_debit / alpha_credit ---

        counts_deb = df_int.groupby("debit_inn").size()
        counts_cred = df_int.groupby("credit_inn").size()

        n_deb = df_int["debit_inn"].map(counts_deb).fillna(0).astype(int)
        n_cred = df_int["credit_inn"].map(counts_cred).fillna(0).astype(int)

        max_n_deb = max(int(counts_deb.max()) if len(counts_deb) else 1, MIN_TX_PER_INN)
        max_n_cred = max(int(counts_cred.max()) if len(counts_cred) else 1, MIN_TX_PER_INN)

        alpha_deb_raw = 0.35 + 0.5 * np.log1p(n_deb) / np.log1p(max_n_deb)
        alpha_cred_raw = 0.35 + 0.5 * np.log1p(n_cred) / np.log1p(max_n_cred)

        alpha_deb = np.clip(alpha_deb_raw, 0.35, 0.85)
        alpha_cred = np.clip(alpha_cred_raw, 0.35, 0.85)

        # где нет индивидуального скора — доверие к индивидуальной модели = 0
        alpha_deb = np.where(has_iforest_deb, alpha_deb, 0.0)
        alpha_cred = np.where(has_iforest_cred, alpha_cred, 0.0)

        df_int["alpha_debit"] = alpha_deb
        df_int["alpha_credit"] = alpha_cred

        # --- 4. Шаг 1: выбираем основу поведения (индивидуальная vs глобальная) ---

        # если индивидуальная модель есть, то individual_score берём как iforest, иначе 0 (alpha=0 → не влияет)
        ind_deb = df_int["iforest_score_debit"].fillna(0.0)
        ind_cred = df_int["iforest_score_credit"].fillna(0.0)
        glob = df_int["global_score"].fillna(0.0)

        # поведенческий риск по роли:
        base_behavior_deb = alpha_deb * ind_deb + (1.0 - alpha_deb) * glob
        base_behavior_cred = alpha_cred * ind_cred + (1.0 - alpha_cred) * glob

        df_int["behavior_risk_debit"] = base_behavior_deb.clip(0, 1)
        df_int["behavior_risk_credit"] = base_behavior_cred.clip(0, 1)

        # итоговый поведенческий риск транзакции:
        # если есть обе роли → пессимистично берём максимум
        # если только одна → берём её
        # если ни одной (крайний случай) → берём global_score
        has_any_side = has_iforest_deb | has_iforest_cred

        base_behavior_txn = np.where(
            has_any_side,
            np.maximum(df_int["behavior_risk_debit"], df_int["behavior_risk_credit"]),
            glob,
        )

        df_int["base_behavior_risk"] = pd.Series(
            base_behavior_txn, index=df_int.index
        ).clip(0, 1)

        # --- 5. Шаг 2: корректоры (семантика, TS, граф) ---

        semantic = _get_col_safe(df_int, "semantic_risk_score", 0.0)
        ts_score = _get_col_safe(df_int, "ts_anomaly_score_txn", 0.0)
        graph_edge = _get_col_safe(df_int, "graph_edge_score", 0.0)
        graph_node = _get_col_safe(df_int, "graph_node_score", 0.0)
        graph_risk = np.maximum(graph_edge, graph_node)

        # веса внутри корректирующего блока (между собой)
        w_semantic = 0.5
        w_ts = 0.3
        w_graph = 0.2

        components_corr = {
            "semantic": (semantic, w_semantic),
            "ts": (ts_score, w_ts),
            "graph": (graph_risk, w_graph),
        }

        weights_used_corr: dict[str, float] = {}
        for name, (series, w) in components_corr.items():
            if series.max() > 0:  # компонент действительно что-то даёт
                weights_used_corr[name] = w

        if not weights_used_corr:
            correction_score = pd.Series(0.0, index=df_int.index)
        else:
            total_w_corr = sum(weights_used_corr.values())
            corr = 0.0
            for name, w in weights_used_corr.items():
                corr += (w / total_w_corr) * components_corr[name][0]
            correction_score = pd.Series(corr, index=df_int.index).clip(0, 1)

        df_int["correction_score"] = correction_score

        # коэффициент влияния корректоров на итоговый риск:
        GAMMA = self.GAMMA

        final_risk = df_int["base_behavior_risk"] + GAMMA * df_int["correction_score"]
        df_int["txn_risk_score"] = final_risk.clip(0, 1)

        # ранговый риск для удобства
        df_int["txn_risk_rank"] = df_int["txn_risk_score"].rank(
            method="average", pct=True
        )

        # ------------------------------------------------------------------
        # 6. Переблендинг с приоритетом: base > graph > history
        # ------------------------------------------------------------------

        # подтягиваем историю из kb_inn_stats
        hist_deb_map, hist_crd_map, cnt_deb_map, cnt_crd_map = _history_maps(
            self.kb_path
        )

        df_int["history_risk_debit"] = df_int["debit_inn"].astype(str).map(
            hist_deb_map
        )
        df_int["history_risk_credit"] = df_int["credit_inn"].astype(str).map(
            hist_crd_map
        )

        # комбинированная история по транзакции (максимум по ролям)
        hist_deb_vals = (
            df_int["history_risk_debit"].astype(float).fillna(-1.0).values
        )
        hist_crd_vals = (
            df_int["history_risk_credit"].astype(float).fillna(-1.0).values
        )
        hist_comb = np.nanmax(
            np.stack([hist_deb_vals, hist_crd_vals], axis=1),
            axis=1,
        )
        df_int["history_risk_combined"] = hist_comb
        # те, где обе NaN, возвращаем в NaN
        mask_no_hist = (
            df_int["history_risk_debit"].isna()
            & df_int["history_risk_credit"].isna()
        )
        df_int.loc[mask_no_hist, "history_risk_combined"] = np.nan

        # веса истории по количеству транзакций ИНН
        n_deb = df_int["debit_inn"].astype(str).map(cnt_deb_map).fillna(0).astype(int)
        n_crd = df_int["credit_inn"].astype(str).map(cnt_crd_map).fillna(0).astype(int)
        n_eff = np.maximum(n_deb, n_crd)

        HIST_CAP = self.HIST_CAP
        H_HYPER = self.H_HYPER
        MIN_BASE_WEIGHT = self.MIN_BASE_WEIGHT
        GRAPH_WEIGHT_CAP = self.GRAPH_WEIGHT_CAP

        w_hist_raw = (n_eff / (n_eff + H_HYPER)).astype(float) * HIST_CAP
        w_hist = np.where(
            df_int["history_risk_combined"].notna(), w_hist_raw, 0.0
        ).astype(float)

        # вес графа: фиксированный потолок, только если есть сигнал
        w_graph_raw = np.where(
            df_int["graph_node_score"].notna(), GRAPH_WEIGHT_CAP, 0.0
        ).astype(float)

        # остаток под базу
        w_base_raw = 1.0 - (w_graph_raw + w_hist)
        w_base = np.clip(w_base_raw, 0.0, 1.0)

        # если базовый вес < MIN_BASE_WEIGHT — поджимаем graph+history пропорционально
        need_boost = w_base < MIN_BASE_WEIGHT
        if np.any(need_boost):
            remain = 1.0 - MIN_BASE_WEIGHT
            combo = w_graph_raw + w_hist
            scale = np.divide(remain, np.maximum(combo, 1e-9))
            scale = np.where(need_boost, scale, 1.0)

            w_graph = w_graph_raw * scale
            w_hist = w_hist * scale
            w_base = 1.0 - (w_graph + w_hist)
        else:
            w_graph = w_graph_raw

        # финальная сборка риска
        base = df_int["txn_risk_score"].astype(float).fillna(0.0)
        graph = df_int["graph_node_score"].astype(float).fillna(base)
        hist = df_int["history_risk_combined"].astype(float).fillna(base)

        overall = (w_base * base + w_graph * graph + w_hist * hist).clip(0, 1)
        df_int["overall_risk"] = overall

        # уровень риска
        def _bucket(x: float) -> Optional[str]:
            if pd.isna(x):
                return None
            if x >= 0.70:
                return "red"
            if x >= 0.31:
                return "yellow"
            return "green"

        df_int["overall_risk_level"] = df_int["overall_risk"].apply(_bucket)

        return AggregationResult(dataframe=df_int)
