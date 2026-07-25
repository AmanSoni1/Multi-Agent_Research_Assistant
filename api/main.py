"""
FastAPI layer over the LangGraph pipeline -- lets the system be driven from
Postman, curl, or any other client, independent of the Streamlit UI.

Run with: uvicorn api.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph.graph import run_research
from graph.config import PLANNER_BACKEND, MAX_REVISIONS_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="Multi-Agent Research Assistant API",
    description="Planner -> hybrid Retriever -> Writer -> Critic pipeline, built on LangGraph.",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The research question to investigate")
    max_revisions: int = Field(MAX_REVISIONS_DEFAULT, ge=0, le=5)


class ResearchResponse(BaseModel):
    question: str
    final_report: str
    revision_count: int
    plan: list
    step_log: list
    node_timings: list
    token_usage: list
    wall_time_seconds: float


@app.get("/health")
def health():
    return {"status": "ok", "planner_backend": PLANNER_BACKEND}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    start = time.time()
    try:
        final_state = run_research(req.question, max_revisions=req.max_revisions)
    except Exception as exc:
        logger.exception("Pipeline failed for question: %s", req.question)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    if not final_state.get("final_report"):
        raise HTTPException(
            status_code=502,
            detail="Pipeline completed without producing a final report -- check logs.",
        )

    return ResearchResponse(
        question=req.question,
        final_report=final_state["final_report"],
        revision_count=final_state.get("revision_count", 0),
        plan=final_state.get("plan", []),
        step_log=final_state.get("step_log", []),
        node_timings=final_state.get("node_timings", []),
        token_usage=final_state.get("token_usage", []),
        wall_time_seconds=round(time.time() - start, 2),
    )
