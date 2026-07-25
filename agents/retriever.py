"""
Retriever/Search Agent: hybrid retrieval per sub-question.

For each sub-question, pulls from both a web search API (Tavily -- recency,
breadth) and a local ChromaDB vector store (curated arXiv corpus -- reliable
grounding). Combining both is the "hybrid retrieval" design point: web
covers things the curated corpus can't (recent news, long-tail facts),
the vector store covers things web search is noisy for (precise, citeable
technical claims from a controlled corpus).

After combining, a small model reranks/filters the combined pool down to the
most relevant chunks per sub-question, so the Writer isn't drowning in
loosely-related evidence.
"""
import logging

from graph.config import get_openrouter_llm, RERANK_MODEL
from graph.state import ResearchState, EvidenceChunk
from agents.utils import call_llm_json, dedupe_evidence
from tools.web_search import web_search
from tools.vector_store import vector_search

logger = logging.getLogger(__name__)

RERANK_SYSTEM_PROMPT = """You filter and rerank retrieved evidence chunks for relevance \
to a specific sub-question. You will be given the sub-question and a numbered list of \
candidate chunks (title + text snippet). Return ONLY a JSON object naming the indices of \
the chunks worth keeping (most relevant first), dropping any that are off-topic, \
redundant, or too vague to cite. Keep at most 5 chunks. If fewer than 5 are relevant, \
return fewer -- don't pad with irrelevant ones.

Respond with ONLY:
{"keep_indices": [<int>, <int>, ...]}"""


def _rerank(llm, sub_question: str, chunks: list[EvidenceChunk]) -> tuple[list[EvidenceChunk], float, dict]:
    if not chunks:
        return [], 0.0, {"input_tokens": 0, "output_tokens": 0}

    listing = "\n".join(
        f"[{i}] ({c['source_type']}) {c['title']}: {c['text'][:300]}"
        for i, c in enumerate(chunks)
    )
    try:
        parsed, elapsed, tokens = call_llm_json(
            llm,
            RERANK_SYSTEM_PROMPT,
            f"Sub-question: {sub_question}\n\nCandidate chunks:\n{listing}",
            node_name="retriever-rerank",
        )
        keep = parsed.get("keep_indices", [])
        kept = [chunks[i] for i in keep if isinstance(i, int) and 0 <= i < len(chunks)]
        return (kept if kept else chunks[:5]), elapsed, tokens
    except RuntimeError as exc:
        logger.warning("Rerank failed, keeping all chunks unfiltered: %s", exc)
        return chunks[:5], 0.0, {"input_tokens": 0, "output_tokens": 0}


def retriever_node(state: ResearchState) -> dict:
    plan = state["plan"]
    llm = get_openrouter_llm(RERANK_MODEL, temperature=0.0)

    all_evidence: list[EvidenceChunk] = []
    step_log = list(state.get("step_log", []))
    timings = list(state.get("node_timings", []))
    usage = list(state.get("token_usage", []))

    for sq in plan:
        web_chunks = web_search(sq["question"], sub_question_id=sq["id"])
        vec_chunks = vector_search(sq["question"], sub_question_id=sq["id"])
        combined = dedupe_evidence(web_chunks + vec_chunks)

        if not combined:
            step_log.append(f"[{sq['id']}] No evidence found from web or vector store.")
            continue

        kept, elapsed, tokens = _rerank(llm, sq["question"], combined)
        all_evidence.extend(kept)
        step_log.append(
            f"[{sq['id']}] Retrieved {len(web_chunks)} web + {len(vec_chunks)} vector "
            f"results, kept {len(kept)} after reranking."
        )
        timings.append({"node": "retriever-rerank", "seconds": elapsed})
        usage.append({"node": "retriever-rerank", "model": RERANK_MODEL, **tokens})

    return {
        "evidence": all_evidence,
        "step_log": step_log,
        "node_timings": timings,
        "token_usage": usage,
    }
