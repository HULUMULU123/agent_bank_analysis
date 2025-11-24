"""SQLite-backed AML knowledge base layer.

Хранит историю транзакций и агрегаты по ИНН, даёт компактный контекст для LLM.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Константы и базовые настройки
# -------------------------------------------------------------------

KB_PATH = "aml_kb.sqlite"
BAYES_K = 50          # сглаживание истории риска
TRANSIT_THR = 0.7     # порог транзитности по in_out_ratio_30d
SEM_HIGH_THR = 0.7    # порог "высокой" семантической аномалии
TS_HIGH_THR = 0.7     # порог "высокой" TS-аномалии

# Колонки, которые храним по транзакциям (минимально достаточный контекст)
KB_COLS = [
    # базовые
    "date", "debit_inn", "credit_inn",
    "debit_amount", "credit_amount", "amount",
    "purpose", "purpose_clean", "purpose_stopword_high",

    # глобальный / индивидуальный / поведенческий риск
    "global_score",
    "iforest_score_debit", "iforest_score_credit",
    "behavior_risk_debit", "behavior_risk_credit",
    "base_behavior_risk",
    "txn_risk_score",

    # семантика / TS / граф
    "semantic_risk_score",
    "ts_anomaly_score_txn",
    "graph_edge_score", "graph_node_score",

    # структурные флаги
    "round_large_amount",
    "debit_fan_out_ratio", "credit_fan_in_ratio",
    "in_out_ratio_30d",
]


# -------------------------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------------------------

def _connect(db_path: str = KB_PATH) -> sqlite3.Connection:
    """Создаёт подключение к SQLite, при необходимости создаёт каталог."""
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)


def _hash_txn(row: pd.Series) -> str:
    """Детерминированный txn_id, если нет явного идентификатора."""
    parts = [
        str(row.get("date", "")),
        str(row.get("debit_inn", "")),
        str(row.get("credit_inn", "")),
        str(row.get("amount", "")),
        str(row.get("purpose", "")),
    ]
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит входной df к набору колонок KB_COLS + txn_id.
    Нормализует дату, типы и создаёт txn_id при необходимости.
    """
    out = df.copy()

    # дата → ISO строка
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    else:
        out["date"] = None

    # гарантируем наличие всех колонок
    for c in KB_COLS:
        if c not in out.columns:
            out[c] = np.nan

    # числовые колонки (кроме явных строковых)
    num_like = [
        c
        for c in KB_COLS
        if c not in ["date", "debit_inn", "credit_inn", "purpose", "purpose_clean"]
    ]
    for c in num_like:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # purpose_stopword_high как float (0/1)
    if "purpose_stopword_high" in out.columns:
        out["purpose_stopword_high"] = out["purpose_stopword_high"].astype(float)

    # txn_id
    if "txn_id" not in out.columns:
        out["txn_id"] = out.apply(_hash_txn, axis=1)
    else:
        out["txn_id"] = out["txn_id"].astype(str)

    return out[["txn_id"] + KB_COLS]


def _get_col_safe(frame: pd.DataFrame, col: str, default_value: float = 0.0) -> pd.Series:
    """Возвращает столбец как float, если он есть, иначе серию-константу."""
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(default_value, index=frame.index, dtype=float)


# -------------------------------------------------------------------
# 1) ИНИЦИАЛИЗАЦИЯ БД
# -------------------------------------------------------------------

def kb_init(db_path: str = KB_PATH) -> None:
    """Создаёт таблицы kb_transactions и kb_inn_stats, если их ещё нет."""
    with closing(_connect(db_path)) as con, con:
        cur = con.cursor()

        # Транзакции
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_transactions (
                txn_id TEXT PRIMARY KEY,
                date TEXT,
                debit_inn TEXT,
                credit_inn TEXT,
                debit_amount REAL,
                credit_amount REAL,
                amount REAL,
                purpose TEXT,
                purpose_clean TEXT,
                purpose_stopword_high REAL,

                global_score REAL,
                iforest_score_debit REAL,
                iforest_score_credit REAL,
                behavior_risk_debit REAL,
                behavior_risk_credit REAL,
                base_behavior_risk REAL,
                txn_risk_score REAL,

                semantic_risk_score REAL,
                ts_anomaly_score_txn REAL,
                graph_edge_score REAL,
                graph_node_score REAL,

                round_large_amount INTEGER,
                debit_fan_out_ratio REAL,
                credit_fan_in_ratio REAL,
                in_out_ratio_30d REAL
            );
            """
        )

        # Агрегаты по ИНН
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_inn_stats (
                inn TEXT,
                role TEXT CHECK(role IN ('debit','credit')),

                tx_count INTEGER,
                first_tx_date TEXT,
                last_tx_date TEXT,
                uniq_counterparties INTEGER,

                amount_mean REAL,
                amount_p95 REAL,

                risk_mean REAL,
                risk_p95 REAL,
                risk_mean_30d REAL,
                risk_mean_90d REAL,
                max_txn_risk REAL,

                green_cnt INTEGER,
                yellow_cnt INTEGER,
                red_cnt INTEGER,

                fan_ratio_mean REAL,
                transit_ratio REAL,
                loan_like_share REAL,
                semantic_high_share REAL,
                ts_high_share REAL,

                graph_node_mean REAL,
                graph_node_p95 REAL,
                graph_edge_p95 REAL,

                history_risk REAL,
                suspicious_cp_cnt INTEGER,
                suspicious_cp_ratio REAL,
                cp_review_flag INTEGER,

                updated_at TEXT,

                PRIMARY KEY (inn, role)
            );
            """
        )
    print(f"Инициализирована БД памяти агента: {db_path}")


# -------------------------------------------------------------------
# 2) UPSERT ТРАНЗАКЦИЙ + ПЕРЕСЧЁТ ПРОФИЛЕЙ ИНН
# -------------------------------------------------------------------

def kb_upsert_transactions(
    df_new: pd.DataFrame,
    db_path: str = KB_PATH,
    verbose: bool = True,
    recompute_stats: bool = True,
) -> List[str]:
    """
    Добавляет/обновляет транзакции по txn_id и пересчитывает профили ИНН.
    Возвращает список затронутых ИНН.
    """
    if df_new is None or len(df_new) == 0:
        if verbose:
            print("Пустой df_new — нечего загружать в память.")
        return []

    df_ins = _ensure_df_columns(df_new).copy()

    # нормализация txn_id
    df_ins["txn_id"] = (
        df_ins["txn_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.lower()
    )

    # выбрасываем пустые txn_id
    df_ins = df_ins[df_ins["txn_id"].notna() & (df_ins["txn_id"] != "")]
    if df_ins.empty:
        if verbose:
            print("После очистки txn_id данных не осталось.")
        return []

    # дубликаты внутри входного df_new → оставляем последнюю версию
    df_ins = df_ins.drop_duplicates(subset=["txn_id"], keep="last")

    with closing(_connect(db_path)) as con, con:
        cur = con.cursor()

        # список существующих txn_id в БД
        ids = df_ins["txn_id"].tolist()
        existing_set = set()
        if ids:
            CHUNK = 800
            for i in range(0, len(ids), CHUNK):
                chunk = ids[i : i + CHUNK]
                q = f"""
                    SELECT txn_id FROM kb_transactions
                    WHERE txn_id IN ({",".join(["?"] * len(chunk))})
                """
                ex = pd.read_sql_query(q, con, params=chunk)
                existing_set.update(
                    ex["txn_id"].astype(str).str.lower().tolist()
                )

        # разбиваем на новые и обновляемые
        mask_existing = df_ins["txn_id"].isin(existing_set)
        df_update = df_ins[mask_existing].copy()
        df_insert = df_ins[~mask_existing].copy()

        # INSERT
        if not df_insert.empty:
            cols = df_insert.columns.tolist()
            placeholders = ",".join(["?"] * len(cols))
            collist = ",".join(cols)
            rows = [tuple(r) for r in df_insert.itertuples(index=False, name=None)]
            cur.executemany(
                f"INSERT INTO kb_transactions ({collist}) VALUES ({placeholders})",
                rows,
            )
            if verbose:
                print(f"Вставлено новых транзакций: {len(df_insert):,}")

        # UPDATE
        if not df_update.empty:
            cols = [c for c in df_update.columns if c != "txn_id"]
            set_expr = ",".join([f"{c}=?" for c in cols])
            rows = [
                tuple(row[cols].tolist() + [row["txn_id"]])
                for _, row in df_update.iterrows()
            ]
            cur.executemany(
                f"UPDATE kb_transactions SET {set_expr} WHERE txn_id = ?",
                rows,
            )
            if verbose:
                print(f"Обновлено существующих транзакций: {len(df_update):,}")

        con.commit()

        # затронутые ИНН
        affected_inn = (
            pd.concat(
                [df_ins["debit_inn"], df_ins["credit_inn"]],
                ignore_index=True,
            )
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if recompute_stats and affected_inn:
            kb_recompute_inn_stats(
                affected_inn, db_path=db_path, verbose=verbose
            )

    return affected_inn


# -------------------------------------------------------------------
# 3) ПЕРЕСЧЁТ СТАТИСТИК ПО ИНН
# -------------------------------------------------------------------

def kb_recompute_inn_stats(
    inn_list: List[str],
    db_path: str = KB_PATH,
    verbose: bool = True,
) -> None:
    """Пересчитывает агрегаты по заданным ИНН (по дебету и кредиту)."""
    if not inn_list:
        return

    with closing(_connect(db_path)) as con, con:
        tx = pd.read_sql_query(
            "SELECT * FROM kb_transactions",
            con,
            parse_dates=["date"],
        )
        if tx.empty:
            if verbose:
                print("Таблица kb_transactions пуста, пересчитывать нечего.")
            return

        tx["txn_risk_score"] = pd.to_numeric(
            tx["txn_risk_score"], errors="coerce"
        )
        global_mean = (
            float(np.nanmean(tx["txn_risk_score"])) if len(tx) else 0.5
        )
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        all_stats: List[Dict[str, Any]] = []

        for role in ["debit", "credit"]:
            if role == "debit":
                inn_col = "debit_inn"
                cp_col = "credit_inn"
                fan_col = "debit_fan_out_ratio"
            else:
                inn_col = "credit_inn"
                cp_col = "debit_inn"
                fan_col = "credit_fan_in_ratio"

            tx_role = tx[tx[inn_col].isin(inn_list)].copy()
            if tx_role.empty:
                continue

            for inn, grp in tx_role.groupby(inn_col):
                grp_sorted = grp.sort_values("date")

                tx_count = len(grp_sorted)
                first_dt = grp_sorted["date"].min()
                last_dt = grp_sorted["date"].max()
                uniq_cp = grp_sorted[cp_col].nunique(dropna=True)

                amt = pd.to_numeric(grp_sorted["amount"], errors="coerce")
                amount_mean = (
                    float(np.nanmean(amt)) if len(amt) else np.nan
                )
                amount_p95 = (
                    float(np.nanquantile(amt, 0.95)) if len(amt) else np.nan
                )

                risk = pd.to_numeric(
                    grp_sorted["txn_risk_score"], errors="coerce"
                )
                risk_mean = (
                    float(np.nanmean(risk)) if len(risk) else np.nan
                )
                risk_p95 = (
                    float(np.nanquantile(risk, 0.95)) if len(risk) else np.nan
                )
                max_txn_risk = (
                    float(np.nanmax(risk)) if len(risk) else np.nan
                )

                if pd.isna(last_dt):
                    risk_mean_30d = risk_mean_90d = np.nan
                else:
                    cutoff_30 = last_dt - pd.Timedelta(days=30)
                    cutoff_90 = last_dt - pd.Timedelta(days=90)
                    r30 = grp_sorted.loc[
                        grp_sorted["date"] >= cutoff_30, "txn_risk_score"
                    ].astype(float)
                    r90 = grp_sorted.loc[
                        grp_sorted["date"] >= cutoff_90, "txn_risk_score"
                    ].astype(float)
                    risk_mean_30d = (
                        float(np.nanmean(r30)) if len(r30) else np.nan
                    )
                    risk_mean_90d = (
                        float(np.nanmean(r90)) if len(r90) else np.nan
                    )

                green_cnt = int((risk < 0.31).sum())
                yellow_cnt = int(
                    ((risk >= 0.31) & (risk < 0.70)).sum()
                )
                red_cnt = int((risk >= 0.70).sum())

                # fan-out / fan-in
                fan = pd.to_numeric(
                    grp_sorted[fan_col], errors="coerce"
                )
                fan_ratio_mean = (
                    float(np.nanmean(fan)) if len(fan) else np.nan
                )

                # транзитность
                in_out = pd.to_numeric(
                    grp_sorted["in_out_ratio_30d"], errors="coerce"
                ).values
                if len(in_out):
                    transit_mask = np.abs(in_out) > TRANSIT_THR
                    transit_ratio = float(transit_mask.mean())
                else:
                    transit_ratio = np.nan

                # "займы/долги"
                if "purpose_clean" in grp_sorted.columns:
                    loan_like = grp_sorted["purpose_clean"].astype(
                        str
                    ).str.contains(
                        r"\bзайм|\bдолг|\bкредит",
                        flags=re.IGNORECASE,
                        regex=True,
                    )
                    loan_like_share = (
                        float(loan_like.mean()) if len(loan_like) else np.nan
                    )
                else:
                    ph = pd.to_numeric(
                        grp_sorted["purpose_stopword_high"],
                        errors="coerce",
                    )
                    loan_like_share = (
                        float((ph > 0.5).mean()) if len(ph) else np.nan
                    )

                # семантика и TS
                sem = pd.to_numeric(
                    grp_sorted["semantic_risk_score"], errors="coerce"
                )
                ts_ = pd.to_numeric(
                    grp_sorted["ts_anomaly_score_txn"], errors="coerce"
                )
                semantic_high_share = (
                    float((sem > SEM_HIGH_THR).mean()) if len(sem) else np.nan
                )
                ts_high_share = (
                    float((ts_ > TS_HIGH_THR).mean()) if len(ts_) else np.nan
                )

                # граф
                node = pd.to_numeric(
                    grp_sorted["graph_node_score"], errors="coerce"
                )
                edge = pd.to_numeric(
                    grp_sorted["graph_edge_score"], errors="coerce"
                )
                graph_node_mean = (
                    float(np.nanmean(node)) if len(node) else np.nan
                )
                graph_node_p95 = (
                    float(np.nanquantile(node, 0.95))
                    if len(node)
                    else np.nan
                )
                graph_edge_p95 = (
                    float(np.nanquantile(edge, 0.95))
                    if len(edge)
                    else np.nan
                )

                # подозрительные контрагенты (по max(txn_risk_score) >= 0.60)
                cp_group = (
                    grp_sorted.groupby(cp_col)["txn_risk_score"]
                    .max()
                    .dropna()
                )
                suspicious_cp_mask = cp_group >= 0.60
                suspicious_cp_cnt = int(suspicious_cp_mask.sum())
                suspicious_cp_ratio = (
                    float(
                        suspicious_cp_cnt / max(1, int(uniq_cp))
                    )
                    if uniq_cp
                    else 0.0
                )
                cp_review_flag = int(suspicious_cp_ratio > (4.0 / 10.0))

                # байесовское сглаживание исторического риска
                hist_base = risk_mean
                if not np.isnan(risk_mean_30d):
                    hist_base = float(
                        np.nanmean([risk_mean, risk_mean_30d])
                    )
                if np.isnan(hist_base):
                    history_risk = global_mean
                else:
                    w = tx_count / (tx_count + BAYES_K)
                    history_risk = float(
                        w * hist_base + (1 - w) * global_mean
                    )

                all_stats.append(
                    dict(
                        inn=str(inn),
                        role=role,
                        tx_count=int(tx_count),
                        first_tx_date=str(first_dt.date())
                        if pd.notna(first_dt)
                        else None,
                        last_tx_date=str(last_dt.date())
                        if pd.notna(last_dt)
                        else None,
                        uniq_counterparties=int(uniq_cp),
                        amount_mean=amount_mean,
                        amount_p95=amount_p95,
                        risk_mean=risk_mean,
                        risk_p95=risk_p95,
                        risk_mean_30d=risk_mean_30d,
                        risk_mean_90d=risk_mean_90d,
                        max_txn_risk=max_txn_risk,
                        green_cnt=green_cnt,
                        yellow_cnt=yellow_cnt,
                        red_cnt=red_cnt,
                        fan_ratio_mean=fan_ratio_mean,
                        transit_ratio=transit_ratio,
                        loan_like_share=loan_like_share,
                        semantic_high_share=semantic_high_share,
                        ts_high_share=ts_high_share,
                        graph_node_mean=graph_node_mean,
                        graph_node_p95=graph_node_p95,
                        graph_edge_p95=graph_edge_p95,
                        history_risk=history_risk,
                        suspicious_cp_cnt=suspicious_cp_cnt,
                        suspicious_cp_ratio=suspicious_cp_ratio,
                        cp_review_flag=cp_review_flag,
                        updated_at=now,
                    )
                )

        if not all_stats:
            if verbose:
                print(
                    "Для переданных ИНН нет транзакций в kb_transactions."
                )
            return

        stats_df = pd.DataFrame(all_stats)
        cur = con.cursor()
        for _, r in stats_df.iterrows():
            cur.execute(
                """
                INSERT INTO kb_inn_stats
                (inn,role,tx_count,first_tx_date,last_tx_date,uniq_counterparties,
                 amount_mean,amount_p95,
                 risk_mean,risk_p95,risk_mean_30d,risk_mean_90d,max_txn_risk,
                 green_cnt,yellow_cnt,red_cnt,
                 fan_ratio_mean,transit_ratio,loan_like_share,
                 semantic_high_share,ts_high_share,
                 graph_node_mean,graph_node_p95,graph_edge_p95,
                 history_risk,
                 suspicious_cp_cnt,suspicious_cp_ratio,cp_review_flag,
                 updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(inn,role) DO UPDATE SET
                    tx_count=excluded.tx_count,
                    first_tx_date=excluded.first_tx_date,
                    last_tx_date=excluded.last_tx_date,
                    uniq_counterparties=excluded.uniq_counterparties,
                    amount_mean=excluded.amount_mean,
                    amount_p95=excluded.amount_p95,
                    risk_mean=excluded.risk_mean,
                    risk_p95=excluded.risk_p95,
                    risk_mean_30d=excluded.risk_mean_30d,
                    risk_mean_90d=excluded.risk_mean_90d,
                    max_txn_risk=excluded.max_txn_risk,
                    green_cnt=excluded.green_cnt,
                    yellow_cnt=excluded.yellow_cnt,
                    red_cnt=excluded.red_cnt,
                    fan_ratio_mean=excluded.fan_ratio_mean,
                    transit_ratio=excluded.transit_ratio,
                    loan_like_share=excluded.loan_like_share,
                    semantic_high_share=excluded.semantic_high_share,
                    ts_high_share=excluded.ts_high_share,
                    graph_node_mean=excluded.graph_node_mean,
                    graph_node_p95=excluded.graph_node_p95,
                    graph_edge_p95=excluded.graph_edge_p95,
                    history_risk=excluded.history_risk,
                    suspicious_cp_cnt=excluded.suspicious_cp_cnt,
                    suspicious_cp_ratio=excluded.suspicious_cp_ratio,
                    cp_review_flag=excluded.cp_review_flag,
                    updated_at=excluded.updated_at
                ;
                """,
                tuple(r.values),
            )
        con.commit()
        if verbose:
            print(f"Пересчитано профилей ИНН: {len(stats_df):,}")


# -------------------------------------------------------------------
# 4) VIEW ДЛЯ LLM ПО КОНКРЕТНОМУ ИНН
# -------------------------------------------------------------------

def kb_get_inn_llm_view(
    inn: str,
    db_path: str = KB_PATH,
    last_n: int = 20,
) -> Dict[str, Any]:
    """
    Возвращает компактный контекст для LLM по ИНН:
      - агрегированные профили по дебету/кредиту из kb_inn_stats
      - список текстовых флагов (risk_flags)
      - последние N самых рискованных операций по этому ИНН
    """
    inn = str(inn)

    with closing(_connect(db_path)) as con:
        stats = pd.read_sql_query(
            "SELECT * FROM kb_inn_stats WHERE inn = ? ORDER BY role",
            con,
            params=[inn],
        )
        tx = pd.read_sql_query(
            """
            SELECT *
            FROM kb_transactions
            WHERE debit_inn = ? OR credit_inn = ?
            ORDER BY txn_risk_score DESC, date DESC
            LIMIT ?
            """,
            con,
            params=[inn, inn, int(last_n)],
        )

    profiles = stats.to_dict(orient="records")

    # комбинированный исторический риск
    hist_vals = []
    for _, row in stats.iterrows():
        v = row.get("history_risk", np.nan)
        if v is not None and not pd.isna(v):
            hist_vals.append(float(v))
    combined_hist = float(np.nanmax(hist_vals)) if hist_vals else None

    # текстовые флаги
    flags: List[str] = []
    for _, row in stats.iterrows():
        role = row["role"]
        prefix = "как плательщик" if role == "debit" else "как получатель"

        hr = row.get("history_risk", None)
        if hr is not None and not pd.isna(hr) and hr >= 0.7:
            flags.append(
                f"ИНН {prefix}: устойчиво высокий исторический риск ({hr:.2f})."
            )

        if row.get("tx_count", 0):
            frac_red = row.get("red_cnt", 0) / max(1, row["tx_count"])
            if frac_red > 0.3:
                flags.append(
                    f"ИНН {prefix}: более 30% операций в красной зоне риска."
                )

        ls = row.get("loan_like_share", None)
        if ls is not None and not pd.isna(ls) and ls > 0.5:
            flags.append(
                f"ИНН {prefix}: преобладают операции с займами/долгами (доля около {ls:.0%})."
            )

        tr = row.get("transit_ratio", None)
        if tr is not None and not pd.isna(tr) and tr > 0.4:
            flags.append(
                f"ИНН {prefix}: выраженные транзитные потоки (доля транзитных операций около {tr:.0%})."
            )

        sh = row.get("semantic_high_share", None)
        if sh is not None and not pd.isna(sh) and sh > 0.4:
            flags.append(
                f"ИНН {prefix}: частые семантические аномалии в назначениях (~{sh:.0%} операций)."
            )

        th = row.get("ts_high_share", None)
        if th is not None and not pd.isna(th) and th > 0.4:
            flags.append(
                f"ИНН {prefix}: частые аномальные дни по объёмам/активности (~{th:.0%} операций)."
            )

        gn = row.get("graph_node_mean", None)
        if gn is not None and not pd.isna(gn) and gn > 0.7:
            flags.append(
                f"ИНН {prefix}: высокий графовый риск узла (средний графовый скор около {gn:.2f})."
            )

    # убираем дубликаты фраз, сохраняя порядок
    flags = list(dict.fromkeys(flags))

    # последние рискованные операции
    recent_risky_tx: List[Dict[str, Any]] = []
    for _, r in tx.iterrows():
        role_tx = "unknown"
        cp_inn = None
        if str(r.get("debit_inn", "")) == inn:
            role_tx = "debit"
            cp_inn = str(r.get("credit_inn", ""))
        elif str(r.get("credit_inn", "")) == inn:
            role_tx = "credit"
            cp_inn = str(r.get("debit_inn", ""))

        recent_risky_tx.append(
            {
                "date": str(r.get("date")),
                "role": role_tx,
                "counterparty_inn": cp_inn,
                "amount": float(r["amount"])
                if not pd.isna(r.get("amount", np.nan))
                else None,
                "purpose": r.get("purpose"),
                "txn_risk_score": float(r["txn_risk_score"])
                if not pd.isna(r.get("txn_risk_score", np.nan))
                else None,
                "semantic_risk_score": float(r["semantic_risk_score"])
                if not pd.isna(r.get("semantic_risk_score", np.nan))
                else None,
                "ts_anomaly_score_txn": float(r["ts_anomaly_score_txn"])
                if not pd.isna(r.get("ts_anomaly_score_txn", np.nan))
                else None,
                "graph_edge_score": float(r["graph_edge_score"])
                if not pd.isna(r.get("graph_edge_score", np.nan))
                else None,
            }
        )

    return {
        "inn": inn,
        "combined_history_risk": combined_hist,
        "profiles": profiles,
        "risk_flags": flags,
        "recent_risky_tx": recent_risky_tx,
    }
