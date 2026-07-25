"""
Smoke tests for the graph's control flow: state merging across nodes, the
critic->writer reflection loop, and the max_revisions cap. These mock every
LLM call and retrieval call, so they run with no API keys and no network --
they test wiring/logic correctness, not real model quality (that's what the
RAGAS eval suite in eval/ is for).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from unittest.mock import patch, MagicMock

from graph.graph import run_research
from graph.state import EvidenceChunk


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 10}


def make_fake_llm(responses):
    """Returns a MagicMock LLM whose .invoke() yields `responses` in order, then repeats the last."""
    llm = MagicMock()
    call_count = {"n": 0}

    def _invoke(messages):
        i = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return FakeResponse(responses[i])

    llm.invoke.side_effect = _invoke
    return llm


PLANNER_JSON = json.dumps({
    "sub_questions": [
        {"question": "What is X?", "rationale": "background"},
        {"question": "How does X compare to Y?", "rationale": "tradeoffs"},
    ]
})

RERANK_JSON = json.dumps({"keep_indices": [0]})

CRITIC_APPROVE_JSON = json.dumps({"approved": True, "issues": [], "revision_instructions": ""})
CRITIC_REJECT_JSON = json.dumps({
    "approved": False,
    "issues": ["Missing citation in section 2"],
    "revision_instructions": "Add a citation to the claim in section 2.",
})


def _fake_evidence(sub_question_id):
    return [EvidenceChunk(
        sub_question_id=sub_question_id, source_type="web", source_id="http://example.com",
        title="Example", text="Some evidence text.", relevance_score=0.9,
    )]


@patch("agents.retriever.vector_search", return_value=[])
@patch("agents.retriever.web_search")
@patch("agents.critic.get_openrouter_llm")
@patch("agents.writer.get_openrouter_llm")
@patch("agents.retriever.get_openrouter_llm")
@patch("agents.planner.get_planner_llm")
def test_happy_path_approves_first_draft(
    mock_planner_llm, mock_retriever_llm, mock_writer_llm, mock_critic_llm,
    mock_web_search, mock_vector_search,
):
    mock_planner_llm.return_value = make_fake_llm([PLANNER_JSON])
    mock_retriever_llm.return_value = make_fake_llm([RERANK_JSON])
    mock_writer_llm.return_value = make_fake_llm(["# Report\n\nThis is a draft."])
    mock_critic_llm.return_value = make_fake_llm([CRITIC_APPROVE_JSON])
    mock_web_search.side_effect = lambda q, sub_question_id, **kw: _fake_evidence(sub_question_id)

    final_state = run_research("What is X?", max_revisions=2)

    assert final_state["final_report"] == "# Report\n\nThis is a draft."
    assert final_state["revision_count"] == 0
    assert len(final_state["plan"]) == 2
    assert final_state["critic_feedback"]["approved"] is True
    print("PASS: happy path approves on first draft")


@patch("agents.retriever.vector_search", return_value=[])
@patch("agents.retriever.web_search")
@patch("agents.critic.get_openrouter_llm")
@patch("agents.writer.get_openrouter_llm")
@patch("agents.retriever.get_openrouter_llm")
@patch("agents.planner.get_planner_llm")
def test_revision_loop_then_approves(
    mock_planner_llm, mock_retriever_llm, mock_writer_llm, mock_critic_llm,
    mock_web_search, mock_vector_search,
):
    mock_planner_llm.return_value = make_fake_llm([PLANNER_JSON])
    mock_retriever_llm.return_value = make_fake_llm([RERANK_JSON])
    mock_writer_llm.return_value = make_fake_llm(["# Draft v1", "# Draft v2 (revised)"])
    mock_critic_llm.return_value = make_fake_llm([CRITIC_REJECT_JSON, CRITIC_APPROVE_JSON])
    mock_web_search.side_effect = lambda q, sub_question_id, **kw: _fake_evidence(sub_question_id)

    final_state = run_research("What is X?", max_revisions=2)

    assert final_state["revision_count"] == 1
    assert final_state["final_report"] == "# Draft v2 (revised)"
    print("PASS: revision loop runs once then approves")


@patch("agents.retriever.vector_search", return_value=[])
@patch("agents.retriever.web_search")
@patch("agents.critic.get_openrouter_llm")
@patch("agents.writer.get_openrouter_llm")
@patch("agents.retriever.get_openrouter_llm")
@patch("agents.planner.get_planner_llm")
def test_max_revisions_cap_terminates(
    mock_planner_llm, mock_retriever_llm, mock_writer_llm, mock_critic_llm,
    mock_web_search, mock_vector_search,
):
    mock_planner_llm.return_value = make_fake_llm([PLANNER_JSON])
    mock_retriever_llm.return_value = make_fake_llm([RERANK_JSON])
    mock_writer_llm.return_value = make_fake_llm(["# v1", "# v2", "# v3"])
    # Critic NEVER approves -- must terminate via the max_revisions cap, not loop forever.
    mock_critic_llm.return_value = make_fake_llm([CRITIC_REJECT_JSON])
    mock_web_search.side_effect = lambda q, sub_question_id, **kw: _fake_evidence(sub_question_id)

    final_state = run_research("What is X?", max_revisions=1)

    assert final_state["revision_count"] == 1, "should stop growing once cap is hit"
    assert final_state.get("final_report"), "must still ship a final_report even when unapproved"
    assert "Revision cap" in final_state["step_log"][-1]
    print("PASS: max_revisions cap prevents infinite loop and still ships a report")


if __name__ == "__main__":
    test_happy_path_approves_first_draft()
    test_revision_loop_then_approves()
    test_max_revisions_cap_terminates()
    print("\nAll smoke tests passed.")
