"""
Shared state object that flows through every node in the LangGraph pipeline.

Design note: we use a TypedDict (not a Pydantic model) because LangGraph's
StateGraph merges partial updates from each node into this dict automatically.
Pydantic works too, but TypedDict keeps node return values simple ({"key": value})
without needing to reconstruct the whole model on every update.
"""
from __future__ import annotations
from typing import TypedDict, List, Optional, Literal


class SubQuestion(TypedDict):
    id: str                 # e.g. "sq1"
    question: str
    rationale: str          # why this sub-question matters / what it covers


class EvidenceChunk(TypedDict):
    sub_question_id: str
    source_type: Literal["web", "vector_store"]
    source_id: str           # URL for web, doc_id for vector store
    title: str
    text: str
    relevance_score: Optional[float]   # filled in by the reranker step


class CriticFeedback(TypedDict):
    approved: bool
    issues: List[str]           # e.g. ["Sub-question 3 has no citations", ...]
    revision_instructions: str  # concrete instructions for the Writer


class NodeTiming(TypedDict):
    node: str
    seconds: float


class TokenUsage(TypedDict):
    node: str
    model: str
    input_tokens: int
    output_tokens: int


class ResearchState(TypedDict, total=False):
    # input
    question: str

    # planner output
    plan: List[SubQuestion]

    # retriever output
    evidence: List[EvidenceChunk]

    # writer output
    draft_report: str
    final_report: str

    # critic output
    critic_feedback: CriticFeedback
    revision_count: int
    max_revisions: int

    # observability (used by Streamlit UI + eval suite)
    node_timings: List[NodeTiming]
    token_usage: List[TokenUsage]
    step_log: List[str]   # human-readable trace of what happened, for live UI display
