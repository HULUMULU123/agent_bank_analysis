
+14
-4

"""CLI entry point for the transaction risk agent."""
from pathlib import Path

import pandas as pd

from agent_bank_analysis.agents.pipeline import TransactionRiskAgent


def main() -> None:
    agent = TransactionRiskAgent()
    outputs = agent.run()

    # Save enriched table
    output_path = Path("llm_scored_transactions.csv")
    outputs.enriched.to_csv(output_path, index=False)
    # Save enriched table (all engineered features + model scores)
    csv_path = Path("llm_scored_transactions.csv")
    outputs.enriched.to_csv(csv_path, index=False)

    print(f"Processed {len(outputs.enriched)} transactions. Results saved to {output_path}.")
    # Save full feature set per transaction as Excel
    features_xlsx = Path("transaction_features.xlsx")
    outputs.enriched.to_excel(features_xlsx, index=False)

    # Save raw LLM responses per txn_id as a separate Excel file
    llm_xlsx = Path("llm_responses.xlsx")
    outputs.llm_responses.to_excel(llm_xlsx, index=False)

    print(f"Processed {len(outputs.enriched)} transactions.")
    print(f"Full results saved to {csv_path} and {features_xlsx}.")
    print(f"LLM responses saved to {llm_xlsx}.")
    print("LLM prompt preview:\n", outputs.prompt[:500], "...")


if __name__ == "__main__":
    main()