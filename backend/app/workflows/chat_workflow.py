"""
Chat Workflow — Entry point for conversational flow.

Orchestrates the agents following order:
    Researcher → Planner → Advisor → Application

Currently, this file fully implements the Researcher step.
The subsequent agents (Planner, Advisor, Application) will be added later.
"""
from __future__ import annotations

import logging
import uuid

from app.agents.researcher import ResearcherAgent
from app.schemas.agent_state import AgentState, StudentProfile

logger = logging.getLogger(__name__)


class ChatWorkflow:
    """
    Orchestrator coordinates the entire pipeline recruitment pipeline.
    Currently: only running Researcher Agent.
    Will be expanded to include Planner, Advisor, Application.
    """

    def __init__(self, verbose: bool = False):
        self.researcher = ResearcherAgent(verbose=verbose)
        self.verbose = verbose

    async def run(self, state: AgentState) -> AgentState:
        """
        Run the entire workflow from input to output.

        Current Pipeline:
        1. Researcher Agent — Gather information
        (2. Planner Agent — Plan) → TODO
        (3. Advisor Agent — Explain)    → TODO
        (4. Application Agent — Submit application) → TODO if user confirm

        Args:
            state: AgentState has student_profile and user_message.

        Returns:
            AgentState has been fully completed by all agents.
        """
        if not state.session_id:
            state.session_id = str(uuid.uuid4())

        logger.info(f"[Workflow] Starting session: {state.session_id}")

        # ── Step 1: Researcher ─────────────────────────────────────────────────
        logger.info("[Workflow] → Running Researcher Agent")
        state = await self.researcher.run(state)

        if state.errors:
            logger.warning(f"[Workflow] Errors after Researcher: {state.errors}")

        # ── Step 2: Planner (TODO) ─────────────────────────────────────────────
        # from app.agents.planner import PlannerAgent
        # state = await PlannerAgent(verbose=self.verbose).run(state)

        # ── Step 3: Advisor (TODO) ─────────────────────────────────────────────
        # from app.agents.advisor import AdvisorAgent
        # state = await AdvisorAgent(verbose=self.verbose).run(state)

        logger.info(f"[Workflow] Session {state.session_id} completed. Agents: {state.completed_agents}")
        return state

    @classmethod
    def create_initial_state(
        cls,
        user_message: str,
        student_name: str = "",
        urls: list[str] | None = None,
        preferred_universities: list[str] | None = None,
        preferred_majors: list[str] | None = None,
        gpa: float | None = None,
        target_scores: dict | None = None,
    ) -> AgentState:
        """
        Helper creates AgentState from user input data

        Args:
            user_message: Student's question / request.
            student_name: Student's name.
            urls: List of URL's of articles / admission pages to research.
            preferred_universities: List of desired universities.
            preferred_majors: List of desired majors.
            gpa: Current GPA.
            target_scores: Target scores for subjects (e.g. {"Math": 9.0, "Physics": 8.5}).

        Returns:
            AgentState is ready to run the workflow.
        """
        profile = StudentProfile(
            name=student_name,
            gpa=gpa,
            target_scores=target_scores or {},
            preferred_universities=preferred_universities or [],
            preferred_majors=preferred_majors or [],
        )

        return AgentState(
            session_id=str(uuid.uuid4()),
            user_message=user_message,
            student_profile=profile,
            input_urls=urls or [],
        )