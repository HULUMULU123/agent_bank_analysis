"""
Configuration constants for the transaction risk agent.
"""
from pathlib import Path

DATA_DIR = Path("data")
# Default input patterns support both CSV and Excel bank statements.
DEFAULT_INPUT_PATTERNS = ("*.csv", "*.xlsx", "*.xls")
DEFAULT_OUTPUT_EXCEL = Path("llm_scored_transactions.xlsx")

# Columns that should be present; missing ones are created with zeros during preprocessing
SAFE_ZERO_COLS = [
    "debit_roll_cnt_30d", "debit_roll_mean_30d", "debit_roll_std_30d",
    "debit_amount_spike_ratio_7d", "debit_tx_rate_spike_7d", "debit_amount_volatility_30d",
    "credit_roll_cnt_30d", "credit_roll_mean_30d", "credit_roll_std_30d",
    "credit_amount_spike_ratio_7d", "credit_tx_rate_spike_7d", "credit_amount_volatility_30d",
    "daily_total_debit", "daily_total_credit",
    "daily_debit_transaction_count", "daily_credit_transaction_count",
    "debit_accel_ratio_7d", "credit_accel_ratio_7d",
    "debit_to_p95_ratio", "credit_to_p95_ratio",
    "debit_new_counterparty_ratio", "credit_new_counterparty_ratio",
    "debit_fan_out_ratio", "credit_fan_in_ratio",
    "debit_volatility_z", "credit_volatility_z",
    "in_out_ratio_30d", "net_flow_ratio_7d",
    "transit_same_day_flag", "round_large_amount", "client_round_ratio_30d",
    "purpose_len", "purpose_digits_ratio", "purpose_upper_ratio",
    "is_weekend", "is_month_end", "is_quarter_end", "is_year_end",
    "days_since_last_txn_debit", "days_since_last_txn_credit",
]
