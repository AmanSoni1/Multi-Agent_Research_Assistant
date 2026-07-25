"""
Wires the four agent nodes into a LangGraph StateGraph.

Flow:
    planner -> retriever -> writer -> critic -> (loop back to writer, or END)

The critic->writer edge is conditional: if the critic didn't approve AND
we're under the revision cap, go back to the writer with feedback. Otherwise
end -- either because it's approved, or because we've hit max_revisions and
have to ship what we have (this is the hardcoded max_iterations guard from
the brief, preventing an infinite critique loop).
"""
import logging
from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from graph.config import MAX_REVISIONS_DEFAULT
from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.writer import writer_node
from agents.critic import critic_node

logger = logging.getLogger(__name__)


def _route_after_critic(state: ResearchState) -> str:
    feedback = state.get("critic_feedback", {})
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", MAX_REVISIONS_DEFAULT)

    if feedback.get("approved", False):
        return "end"
    if revision_count >= max_revisions:
        logger.warning(
            "Hit max_revisions (%d) without approval -- shipping best draft as final_report.",
            max_revisions,
        )
        return "end_unapproved"
    return "revise"


def _finalize_unapproved(state: ResearchState) -> dict:
    """Reached only when we exhaust the revision budget without approval."""
    step_log = state.get("step_log", []) + [
        f"Revision cap ({state.get('max_revisions', MAX_REVISIONS_DEFAULT)}) reached without "
        f"approval -- shipping the last draft as final_report with known limitations noted."
    ]
    return {"final_report": state.get("draft_report", ""), "step_log": step_log}


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize_unapproved", _finalize_unapproved)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "writer")
    graph.add_edge("writer", "critic")

    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"end": END, "revise": "writer", "end_unapproved": "finalize_unapproved"},
    )
    graph.add_edge("finalize_unapproved", END)

    return graph.compile()


def run_research(question: str, max_revisions: int = MAX_REVISIONS_DEFAULT) -> ResearchState:
    """Convenience entrypoint used by main.py, the FastAPI layer, and the eval suite."""
    app = build_graph()
    initial_state: ResearchState = {
        "question": question,
        "revision_count": 0,
        "max_revisions": max_revisions,
        "step_log": [],
        "node_timings": [],
        "token_usage": [],
    }
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})
    return final_state
