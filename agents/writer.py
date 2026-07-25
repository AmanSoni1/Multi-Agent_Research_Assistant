"""
Synthesizer/Writer Agent: turns retrieved evidence into a structured report.

Runs the same node on both the first pass and every revision pass -- if
critic_feedback exists and isn't approved, the revision instructions are
appended to the prompt so the Writer sees exactly what to fix.
"""
import logging
import time

from graph.config import get_openrouter_llm, WRITER_MODEL
from graph.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research report writer. You will be given a research question, \
a set of sub-questions with rationale, and retrieved evidence chunks tagged by which \
sub-question they support and their source (web URL or arXiv paper).

Write a structured Markdown report with:
- A short introduction framing the overall question
- One section per sub-question, titled with the sub-question itself, synthesizing the \
relevant evidence into a coherent explanation. Cite sources inline like [Source: <source_id>] \
after any claim drawn from a specific piece of evidence. Do not invent facts not present in \
the evidence -- if evidence for a sub-question is thin or absent, say so explicitly rather \
than filling the gap with unstated assumptions.
- A conclusion synthesizing across all sub-questions.

If you are given revision instructions from a prior critique, address every point raised \
before returning the report -- do not ignore any of them."""


def _format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "(No evidence was retrieved.)"
    lines = []
    for c in evidence:
        lines.append(
            f"- [{c['sub_question_id']} | {c['source_type']} | {c['source_id']}] "
            f"{c['title']}: {c['text'][:500]}"
        )
    return "\n".join(lines)


def _format_plan(plan: list[dict]) -> str:
    return "\n".join(f"- ({sq['id']}) {sq['question']} -- {sq['rationale']}" for sq in plan)


def writer_node(state: ResearchState) -> dict:
    llm = get_openrouter_llm(WRITER_MODEL, temperature=0.4)

    prompt_parts = [
        f"Research question: {state['question']}",
        f"\nSub-questions:\n{_format_plan(state['plan'])}",
        f"\nEvidence:\n{_format_evidence(state.get('evidence', []))}",
    ]

    feedback = state.get("critic_feedback")
    if feedback and not feedback.get("approved", True):
        prompt_parts.append(
            f"\nThis is a REVISION. Prior draft was not approved. "
            f"Revision instructions:\n{feedback.get('revision_instructions', '')}\n"
            f"Specific issues to fix:\n" + "\n".join(f"- {i}" for i in feedback.get("issues", []))
        )
        prompt_parts.append(f"\nPrevious draft:\n{state.get('draft_report', '')}")

    start = time.time()
    try:
        response = llm.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(prompt_parts)}]
        )
        report = response.content if isinstance(response.content, str) else str(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        tokens = {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}
    except Exception as exc:  # noqa: BLE001
        logger.error("Writer LLM call failed: %s", exc)
        report = (
            f"# Report generation failed\n\nThe Writer agent could not reach the model: {exc}\n\n"
            f"Retrieved evidence is still available below for manual review.\n\n{_format_evidence(state.get('evidence', []))}"
        )
        tokens = {"input_tokens": 0, "output_tokens": 0}
    elapsed = time.time() - start

    revision_count = state.get("revision_count", 0)
    is_revision = bool(feedback and not feedback.get("approved", True))

    step_log = state.get("step_log", []) + [
        f"Writer produced {'a revised' if is_revision else 'the initial'} draft "
        f"({len(report)} chars)."
    ]
    timings = state.get("node_timings", []) + [{"node": "writer", "seconds": elapsed}]
    usage_log = state.get("token_usage", []) + [{"node": "writer", "model": WRITER_MODEL, **tokens}]

    return {
        "draft_report": report,
        "revision_count": revision_count + 1 if is_revision else revision_count,
        "step_log": step_log,
        "node_timings": timings,
        "token_usage": usage_log,
    }
