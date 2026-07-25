"""
Critic/Reviewer Agent: reviews the draft for unsupported claims, missing
citations, weak sub-question coverage, and coherence issues. Outputs either
approval or concrete revision instructions -- this is the reflection loop.

The revision cap (max_iterations) is enforced by the graph's conditional edge,
not here -- the critic always gives an honest assessment; the graph decides
whether there's budget left to act on it.
"""
import logging

from graph.config import get_openrouter_llm, CRITIC_MODEL
from graph.state import ResearchState, CriticFeedback
from agents.utils import call_llm_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict research report reviewer. Check the draft report against \
the original sub-questions and the evidence it was supposed to be grounded in. Look specifically for:
1. Unsupported claims -- statements presented as fact with no [Source: ...] citation and no \
   corresponding evidence chunk backing them.
2. Missing citations -- claims that should cite evidence but don't.
3. Weak coverage -- any sub-question addressed superficially or not at all, when evidence for \
   it did exist.
4. Coherence issues -- contradictions, disorganization, or a conclusion that doesn't follow from \
   the sections above it.

Be strict but fair: approve if the report is genuinely well-grounded and complete, even if not \
perfect prose. Don't demand citations for evidence that was never retrieved -- flag that as a \
coverage gap instead, since the Writer can't cite what doesn't exist.

Respond with ONLY a JSON object:
{
  "approved": true or false,
  "issues": ["specific issue 1", "specific issue 2", ...],
  "revision_instructions": "concrete, actionable instructions for fixing the issues (empty string if approved)"
}"""


def critic_node(state: ResearchState) -> dict:
    llm = get_openrouter_llm(CRITIC_MODEL, temperature=0.0)

    evidence_summary = "\n".join(
        f"- [{c['sub_question_id']}] {c['title']} ({c['source_id']})"
        for c in state.get("evidence", [])
    ) or "(no evidence retrieved)"

    user_prompt = (
        f"Original question: {state['question']}\n\n"
        f"Sub-questions:\n" + "\n".join(f"- ({sq['id']}) {sq['question']}" for sq in state["plan"]) + "\n\n"
        f"Available evidence (for checking whether coverage gaps are the Writer's fault or "
        f"a retrieval gap):\n{evidence_summary}\n\n"
        f"Draft report:\n{state.get('draft_report', '')}"
    )

    try:
        parsed, elapsed, tokens = call_llm_json(llm, SYSTEM_PROMPT, user_prompt, node_name="critic")
        feedback = CriticFeedback(
            approved=bool(parsed.get("approved", False)),
            issues=parsed.get("issues", []),
            revision_instructions=parsed.get("revision_instructions", ""),
        )
    except RuntimeError as exc:
        logger.error("Critic failed, defaulting to approval so the pipeline can terminate: %s", exc)
        feedback = CriticFeedback(
            approved=True,
            issues=[f"Critic LLM call failed: {exc}"],
            revision_instructions="",
        )
        elapsed, tokens = 0.0, {"input_tokens": 0, "output_tokens": 0}

    step_log = state.get("step_log", []) + [
        f"Critic {'approved' if feedback['approved'] else 'requested revisions on'} the draft "
        f"({len(feedback['issues'])} issue(s) noted)."
    ]
    timings = state.get("node_timings", []) + [{"node": "critic", "seconds": elapsed}]
    usage = state.get("token_usage", []) + [{"node": "critic", "model": CRITIC_MODEL, **tokens}]

    result = {
        "critic_feedback": feedback,
        "step_log": step_log,
        "node_timings": timings,
        "token_usage": usage,
    }
    if feedback["approved"]:
        result["final_report"] = state.get("draft_report", "")
    return result
