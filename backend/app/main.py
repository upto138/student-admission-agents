"""
FastAPI Application Entry Point.

Routes:
  POST /chat - Submit your question, receive the result from the Researcher Agent
  GET  /health — Health check
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load the .env file before importing agent modules
load_dotenv()

from app.workflows.chat_workflow import ChatWorkflow
# from app.schemas.agent_state import AgentState

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Admission Agents API",
    description="Multi-Agent System for university admission assistance (Vietnam)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify a particular domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for endpoint /chat."""
    message: str
    student_name: str = ""
    urls: List[str] = []
    preferred_universities: List[str] = []
    preferred_majors: List[str] = []
    gpa: Optional[float] = None
    target_scores: dict = {}


class ResearchResponse(BaseModel):
    """Response cho endpoint /chat."""
    session_id: str
    # ── Researcher fields ──
    raw_summary: str
    universities_count: int
    universities: list
    general_requirements: list
    important_deadlines: dict
    admission_methods: list
    sources: list
    # ── Planner fields ──
    plan_result: Optional[Dict[str, Any]] = None
    missing_questions: List[str] = Field(default_factory=list)
    needs_human_confirmation: bool = False
    # ── Metadata ──
    errors: list
    completed_agents: list


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "agents": ["researcher", "planner (coming)", "advisor (coming)", "application (coming)"],
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", "NOT SET"),
        "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "NOT SET"),
    }


@app.post("/chat", response_model=ResearchResponse)
async def chat(request: ChatRequest):
    """
    The main endpoint - Receive requests from student and run the workflow.

    Currently only running Researcher Agent.
    Will be expanded to include Planner, Advisor, Application Agents.
    """
    logger.info(f"[API] /chat received: '{request.message[:80]}...' | URLs: {len(request.urls)}")

    try:
        # Create initial state
        state = ChatWorkflow.create_initial_state(
            user_message=request.message,
            student_name=request.student_name,
            urls=request.urls,
            preferred_universities=request.preferred_universities,
            preferred_majors=request.preferred_majors,
            gpa=request.gpa,
            target_scores=request.target_scores,
        )

        # Run workflow
        workflow = ChatWorkflow(verbose=True)
        state = await workflow.run(state)

        # Format response
        result = state.research_result
        if result is None:
            raise HTTPException(status_code=500, detail="Researcher returned no result.")

        universities_data = []
        for u in result.universities:
            universities_data.append(u.model_dump(exclude_none=True))

        return ResearchResponse(
            session_id=state.session_id,
            raw_summary=result.raw_summary,
            universities_count=len(result.universities),
            universities=universities_data,
            general_requirements=result.general_requirements,
            important_deadlines=result.important_deadlines,
            admission_methods=result.admission_methods,
            sources=result.sources,
            # ── Planner fields ──
            plan_result=state.plan_result.model_dump() if state.plan_result else None,
            missing_questions=(
                state.plan_result.missing_questions if state.plan_result else []
            ),
            needs_human_confirmation=(
                state.plan_result.needs_human_confirmation if state.plan_result else False
            ),
            # ── Metadata ──
            errors=state.errors,
            completed_agents=state.completed_agents,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[API] Unhandled error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point when running live ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)