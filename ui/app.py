"""
Streamlit UI for the Multi-Agent Research Assistant.

Runs the LangGraph pipeline in-process (via .stream()) so the UI can render
each agent's output as it completes, rather than waiting for a single final
blob of text -- this is the "show the multi-agent process working" demo
requirement.

Run with: streamlit run ui/app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
# Add the project's parent directory to Python's module search path for imports
# research_assistant/ui/app.py -> research_assistant/ui  -> research_assistant

import streamlit as st

from graph.graph import build_graph
from graph.config import PLANNER_BACKEND, MAX_REVISIONS_DEFAULT

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")
st.title("🔎 Multi-Agent Research Assistant")
st.caption(
    f"Planner → Hybrid Retriever → Writer → Critic (reflection loop) · "
    f"Planner backend: `{PLANNER_BACKEND}`"
)

with st.sidebar:
    st.header("Settings")
    max_revisions = st.slider("Max revision cycles", 0, 5, MAX_REVISIONS_DEFAULT)
    st.markdown("---")
    st.markdown(
        "**Architecture**\n\n"
        "1. Planner decomposes the question\n"
        "2. Retriever pulls web + vector-store evidence per sub-question, reranks it\n"
        "3. Writer synthesizes a cited report\n"
        "4. Critic checks it; loops back to Writer if not approved (capped)"
    )

question = st.text_input(
    "Research question",
    placeholder="e.g. What are the main approaches to reducing hallucination in RAG systems?",
)
run_button = st.button("Run research", type="primary", disabled=not question)

if run_button and question:
    plan_box = st.empty()
    evidence_box = st.empty()
    draft_box = st.empty()
    critic_box = st.empty()
    final_box = st.empty()
    log_box = st.expander("Full step log", expanded=False)

    graph = build_graph()
    initial_state = {
        "question": question,
        "revision_count": 0,
        "max_revisions": max_revisions,
        "step_log": [],
        "node_timings": [],
        "token_usage": [],
    }

    accumulated_log = []
    final_state = None

    with st.spinner("Running pipeline..."):
        try:
            for step in graph.stream(initial_state, config={"recursion_limit": 50}):
                for node_name, node_output in step.items():
                    final_state = {**(final_state or {}), **node_output}
                    accumulated_log.extend(node_output.get("step_log", [])[len(accumulated_log):])

                    if node_name == "planner":
                        with plan_box.container():
                            st.subheader("📋 Plan")
                            for sq in node_output.get("plan", []):
                                st.markdown(f"**{sq['id']}. {sq['question']}**  \n*{sq['rationale']}*")

                    elif node_name == "retriever":
                        with evidence_box.container():
                            st.subheader("📚 Retrieved evidence")
                            evidence = node_output.get("evidence", [])
                            if not evidence:
                                st.warning("No evidence retrieved for any sub-question.")
                            for c in evidence:
                                st.markdown(
                                    f"`{c['sub_question_id']}` **{c['title']}** "
                                    f"({c['source_type']}) — {c['source_id']}"
                                )

                    elif node_name == "writer":
                        with draft_box.container():
                            st.subheader(
                                f"✍️ Draft (revision {node_output.get('revision_count', 0)})"
                            )
                            st.markdown(node_output.get("draft_report", ""))

                    elif node_name == "critic":
                        fb = node_output.get("critic_feedback", {})
                        with critic_box.container():
                            st.subheader("🧐 Critic review")
                            if fb.get("approved"):
                                st.success("Approved")
                            else:
                                st.warning("Revisions requested")
                                for issue in fb.get("issues", []):
                                    st.markdown(f"- {issue}")

                    elif node_name == "finalize_unapproved":
                        st.info(
                            "Revision cap reached — shipping the last draft as final, "
                            "with the critic's outstanding issues visible above."
                        )

        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.exception(exc)

    if final_state and final_state.get("final_report"):
        with final_box.container():
            st.subheader("✅ Final report")
            st.markdown(final_state["final_report"])

            total_in = sum(t["input_tokens"] for t in final_state.get("token_usage", []))
            total_out = sum(t["output_tokens"] for t in final_state.get("token_usage", []))
            c1, c2, c3 = st.columns(3)
            c1.metric("Revision cycles", final_state.get("revision_count", 0))
            c2.metric("Total tokens", f"{total_in + total_out:,}")
            c3.metric("Nodes executed", len(final_state.get("node_timings", [])))

    with log_box:
        for line in accumulated_log:
            st.text(line)
