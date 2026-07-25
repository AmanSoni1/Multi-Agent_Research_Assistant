"""
Eval Stage B -- run in the isolated `eval_venv` (see eval/requirements.txt),
NOT the main project venv. See eval/generate_results.py for why these are split.

Reads eval/eval_results.json (produced by generate_results.py), scores each
question with RAGAS faithfulness + answer relevance, and writes a combined
results table (RAGAS scores + our own operational metrics: revision cycles,
latency, cost) to eval/eval_report.md and eval/eval_report.csv.

Usage (from project root):
    python eval/generate_results.py                 # Stage A, main venv
    ./eval_venv/bin/python eval/ragas_eval.py        # Stage B, eval venv
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")
REPORT_MD_PATH = os.path.join(os.path.dirname(__file__), "eval_report.md")
REPORT_CSV_PATH = os.path.join(os.path.dirname(__file__), "eval_report.csv")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RAGAS_JUDGE_MODEL = "deepseek/deepseek-v4-flash"  # cheap judge model for RAGAS's own LLM-based scoring

# RAGAS's answer-relevancy metric needs an embeddings model. We use OpenAI's
# via OpenRouter isn't supported for embeddings, so this falls back to a
# direct OpenAI key if set -- otherwise this metric is skipped gracefully.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def load_results():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run `python eval/generate_results.py` "
            f"in the MAIN venv first (this script must run in eval_venv)."
        )
    with open(RESULTS_PATH) as f:
        return json.load(f)


def main():
    raw_results = load_results()
    valid_results = [r for r in raw_results if "error" not in r]
    failed_count = len(raw_results) - len(valid_results)

    if not valid_results:
        print("No valid results to evaluate (all questions errored in Stage A). Exiting.")
        return

    judge_llm = ChatOpenAI(
        model=RAGAS_JUDGE_MODEL, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, temperature=0,
    )

    metrics = [Faithfulness()]
    embeddings = None
    if OPENAI_API_KEY:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        metrics.append(AnswerRelevancy())
    else:
        print(
            "NOTE: OPENAI_API_KEY not set -- skipping answer_relevancy (needs an embeddings model). "
            "Faithfulness only (uses the judge LLM, no embeddings needed)."
        )

    dataset = Dataset.from_list([
        {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]}
        for r in valid_results
    ])

    print(f"Scoring {len(valid_results)} question(s) with RAGAS ({[m.name for m in metrics]})...")
    ragas_result = evaluate(
        dataset, metrics=metrics, llm=judge_llm, embeddings=embeddings, raise_exceptions=False,
    )
    scores_df = ragas_result.to_pandas()

    # --- merge in our own operational metrics ---
    for i, r in enumerate(valid_results):
        scores_df.loc[i, "revision_count"] = r["revision_count"]
        scores_df.loc[i, "wall_time_seconds"] = r["wall_time_seconds"]
        scores_df.loc[i, "estimated_cost_usd"] = r["estimated_cost_usd"]

    scores_df.to_csv(REPORT_CSV_PATH, index=False)

    avg_faithfulness = scores_df["faithfulness"].mean() if "faithfulness" in scores_df else float("nan")
    avg_relevance = scores_df["answer_relevancy"].mean() if "answer_relevancy" in scores_df else float("nan")
    avg_revisions = scores_df["revision_count"].mean()
    avg_latency = scores_df["wall_time_seconds"].mean()
    avg_cost = scores_df["estimated_cost_usd"].mean()
    total_cost = scores_df["estimated_cost_usd"].sum()

    with open(REPORT_MD_PATH, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write(f"Evaluated {len(valid_results)} question(s)")
        if failed_count:
            f.write(f" ({failed_count} failed during pipeline run and were excluded)")
        f.write(".\n\n")
        f.write("## Summary\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Avg. Faithfulness | {avg_faithfulness:.3f} |\n")
        if "answer_relevancy" in scores_df:
            f.write(f"| Avg. Answer Relevance | {avg_relevance:.3f} |\n")
        f.write(f"| Avg. Revision Cycles | {avg_revisions:.2f} |\n")
        f.write(f"| Avg. Latency (s) | {avg_latency:.1f} |\n")
        f.write(f"| Avg. Cost per Query (USD) | ${avg_cost:.5f} |\n")
        f.write(f"| Total Cost for Eval Run (USD) | ${total_cost:.4f} |\n\n")
        f.write("## Per-question results\n\n")
        f.write(scores_df.to_markdown(index=False))
        f.write("\n")

    print(f"\nWrote {REPORT_MD_PATH} and {REPORT_CSV_PATH}")
    print(f"Avg faithfulness: {avg_faithfulness:.3f}  |  Avg revisions: {avg_revisions:.2f}  |  Avg cost/query: ${avg_cost:.5f}")


if __name__ == "__main__":
    main()
