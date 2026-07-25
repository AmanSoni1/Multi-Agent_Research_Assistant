"""
Eval Stage A -- run in the MAIN venv (has langgraph/chromadb/tavily).

Runs the full pipeline on a fixed set of test questions and dumps everything
RAGAS needs (question, answer, retrieved contexts) plus our own operational
metrics (revision cycles, per-node latency, token-based cost estimate) to a
JSON file. Stage B (eval/ragas_eval.py, run in the separate eval_venv) reads
that file and computes faithfulness / answer relevance.

Two-venv split exists because ragas 0.4.3's dependency chain (via
langchain-community) currently conflicts with the langgraph/langchain-core 1.x
stack this project's main pipeline depends on -- see README "Known Limitations".

Usage: python eval/generate_results.py
"""
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.graph import run_research

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval.generate_results")

# OpenRouter per-token pricing used for the cost estimate (USD per token).
# Update these if you change WRITER_MODEL / CRITIC_MODEL / RERANK_MODEL in graph/config.py.
MODEL_PRICING = {
    "deepseek/deepseek-v4-flash": {"input": 0.10e-6, "output": 0.20e-6},
    "planner-api": {"input": 0.10e-6, "output": 0.20e-6},
    "planner-local": {"input": 0.0, "output": 0.0},  # local inference, no per-token cost
}

TEST_QUESTIONS = [
    "What are the main approaches to reducing hallucination in RAG systems?",
    "How do multi-agent LLM systems differ from single-prompt chain-of-thought approaches?",
    "What are the tradeoffs between local and API-hosted LLM inference for agentic systems?",
    "How does hybrid retrieval combining web search and vector stores improve grounding?",
    "What role does a critic/reflection loop play in improving LLM-generated reports?",
    "What are the current limitations of retrieval-augmented generation?",
    "How do tool-use capable language models decide when to call an external tool?",
    "What is the state of the art in multi-step research agent planning?",
]


def estimate_cost(token_usage: list[dict]) -> float:
    total = 0.0
    for entry in token_usage:
        pricing = MODEL_PRICING.get(entry["model"])
        if not pricing:
            continue
        total += entry["input_tokens"] * pricing["input"] + entry["output_tokens"] * pricing["output"]
    return total


def main(n_questions: int = None, output_path: str = None):
    questions = TEST_QUESTIONS[:n_questions] if n_questions else TEST_QUESTIONS
    output_path = output_path or os.path.join(os.path.dirname(__file__), "eval_results.json")

    results = []
    for i, q in enumerate(questions, 1):
        logger.info("Running question %d/%d: %s", i, len(questions), q)
        start = time.time()
        try:
            final_state = run_research(q, max_revisions=2)
        except Exception as exc:
            logger.error("Question failed entirely: %s -- %s", q, exc)
            results.append({"question": q, "error": str(exc)})
            continue
        wall_time = time.time() - start

        contexts = [c["text"] for c in final_state.get("evidence", [])]
        results.append({
            "question": q,
            "answer": final_state.get("final_report", ""),
            "contexts": contexts if contexts else ["(no evidence retrieved)"],
            "revision_count": final_state.get("revision_count", 0),
            "wall_time_seconds": round(wall_time, 2),
            "node_timings": final_state.get("node_timings", []),
            "token_usage": final_state.get("token_usage", []),
            "estimated_cost_usd": round(estimate_cost(final_state.get("token_usage", [])), 6),
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %d results to %s", len(results), output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Limit to first N test questions")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(n_questions=args.n, output_path=args.output)
