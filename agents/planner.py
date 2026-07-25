"""
Planner Agent: decomposes the raw research question into 3-5 sub-questions.

Backend is pluggable (see graph/config.py: PLANNER_BACKEND=local|api).
This is intentionally the cheapest node in the graph -- decomposition is a
lower-reasoning-burden task than synthesis or critique, so it's the natural
place to demonstrate cost-aware model routing without hurting quality where
it matters most (the Writer).
"""
import logging

from graph.config import get_planner_llm, PLANNER_BACKEND
from graph.state import ResearchState, SubQuestion
from agents.utils import call_llm_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research planning assistant. Given a research question, \
decompose it into 3 to 5 distinct sub-questions that together give thorough coverage \
of the topic. Each sub-question should probe a different angle (e.g. definitions/background, \
current approaches, tradeoffs/limitations, applications, open problems) -- avoid overlapping \
sub-questions that would retrieve the same evidence.

Respond with ONLY a JSON object, no other text, in this exact shape:
{
  "sub_questions": [
    {"question": "...", "rationale": "one sentence on why this angle matters"},
    ...
  ]
}"""


def planner_node(state: ResearchState) -> dict:
    question = state["question"]
    llm = get_planner_llm()

    try:
        parsed, elapsed, tokens = call_llm_json(
            llm,
            SYSTEM_PROMPT,
            f"Research question: {question}",
            node_name="planner",
        )
        raw_subqs = parsed.get("sub_questions", [])
    except RuntimeError as exc:
        logger.error("Planner failed, falling back to a single-question plan: %s", exc)
        raw_subqs = [{"question": question, "rationale": "Fallback: planner failed, using original question directly."}]
        elapsed, tokens = 0.0, {"input_tokens": 0, "output_tokens": 0}

    plan: list[SubQuestion] = [
        SubQuestion(id=f"sq{i+1}", question=sq["question"], rationale=sq.get("rationale", ""))
        for i, sq in enumerate(raw_subqs[:5])
    ]
    if not plan:
        plan = [SubQuestion(id="sq1", question=question, rationale="Fallback plan.")]

    step_log = state.get("step_log", []) + [
        f"Planner ({PLANNER_BACKEND} backend) produced {len(plan)} sub-questions."
    ]
    timings = state.get("node_timings", []) + [{"node": "planner", "seconds": elapsed}]
    usage = state.get("token_usage", []) + [
        {"node": "planner", "model": f"planner-{PLANNER_BACKEND}", **tokens}
    ]

    return {
        "plan": plan,
        "step_log": step_log,
        "node_timings": timings,
        "token_usage": usage,
        "revision_count": state.get("revision_count", 0),
        "max_revisions": state.get("max_revisions", 2),
    }
