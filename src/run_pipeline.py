"""CLI entrypoint:  python -m src.run_pipeline [--simulate-failure]"""
import argparse
import os
import sys

from src import config, db
from src.agent import run_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agentic books pipeline.")
    parser.add_argument("--simulate-failure", action="store_true",
                        help="fail the first request to page 2 once (also: SIMULATE_FAILURE=1)")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    simulate = args.simulate_failure or config.simulate_failure_from_env()
    result = run_agent(simulate_failure=simulate)

    silver = db.query("SELECT COUNT(*) AS n FROM silver_books")[0]["n"]
    summary = db.query("SELECT summary FROM gold_summary LIMIT 1")
    print("\n" + "=" * 70)
    print(f"Run:     {result['run_id']}")
    print(f"Outcome: {result['reason']}")
    print(f"Silver rows: {silver}")
    if summary:
        print(f"\nAI summary:\n{summary[0]['summary']}")
    return 0 if result["finished"] else 2


if __name__ == "__main__":
    sys.exit(main())
