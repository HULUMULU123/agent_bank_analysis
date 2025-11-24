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

    print(f"Processed {len(outputs.enriched)} transactions. Results saved to {output_path}.")
    print("LLM prompt preview:\n", outputs.prompt[:500], "...")


if __name__ == "__main__":
    main()
