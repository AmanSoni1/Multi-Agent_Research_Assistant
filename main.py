"""
CLI entrypoint: run the full Planner -> Retriever -> Writer -> Critic pipeline
on a research question from the command line.

Usage:
    python main.py "What are the main approaches to reducing hallucination in RAG systems?"
    python main.py "..." --max-revisions 1
    PLANNER_BACKEND=local python main.py "..."   # use Gemma 3 4B via Ollama for the Planner
"""
import argparse
import logging
import sys
import time

from graph.graph import run_research
from graph.config import PLANNER_BACKEND

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument("question", type=str, help="The research question to investigate")
    parser.add_argument("--max-revisions", type=int, default=2, help="Max critic revision cycles (default: 2)")
    args = parser.parse_args()

    print(f"\n{'='*70}\nResearch question: {args.question}")
    print(f"Planner backend: {PLANNER_BACKEND}   |   Max revisions: {args.max_revisions}\n{'='*70}\n")

    start = time.time()
    try:
        final_state = run_research(args.question, max_revisions=args.max_revisions)
    except Exception as exc:
        logger.exception("Pipeline failed with an unhandled error: %s", exc)
        sys.exit(1)
    total_time = time.time() - start

    print("\n--- STEP LOG ---")
    for line in final_state.get("step_log", []):
        print(f"  - {line}")

    print(f"\n--- FINAL REPORT ---\n")
    print(final_state.get("final_report") or "(no final report -- pipeline did not converge)")

    total_tokens_in = sum(t["input_tokens"] for t in final_state.get("token_usage", []))
    total_tokens_out = sum(t["output_tokens"] for t in final_state.get("token_usage", []))
    print(f"\n--- RUN STATS ---")
    print(f"  Total wall time: {total_time:.1f}s")
    print(f"  Revision cycles: {final_state.get('revision_count', 0)}")
    print(f"  Total tokens: {total_tokens_in} in / {total_tokens_out} out")


if __name__ == "__main__":
    main()
