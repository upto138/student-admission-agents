"""
Chat Workflow — Entry point cho conversational flow.

Orchestrates các agent theo thứ tự:
    Researcher → Planner → Advisor → Application

Hiện tại:
    Bước 1 (Researcher)  ✔ Hoàn chỉnh
    Bước 2 (Planner)    ✔ Hoàn chỉnh
    Bước 3 (Advisor)    TODO
    Bước 4 (Application) TODO
"""
from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator

from app.agents.researcher import ResearcherAgent
from app.agents.planner import PlannerAgent
from app.schemas.agent_state import AgentState, StudentProfile

logger = logging.getLogger(__name__)


class ChatWorkflow:
    """
    Orchestrator điều phối toàn bộ pipeline tuyển sinh.

    Pipeline hiện tại:
      Bước 1: ResearcherAgent  — Thu thập và trích xuất thông tin tuyển sinh
      Bước 2: PlannerAgent     — Lập checklist + timeline + phát hiện thiếu thông tin
      (Bước 3: AdvisorAgent)  — TODO
      (Bước 4: ApplicationAgent) — TODO nếu user xác nhận
    """

    def __init__(self, verbose: bool = False):
        self.researcher = ResearcherAgent(verbose=verbose)
        self.planner = PlannerAgent(verbose=verbose)
        self.verbose = verbose

    async def run(self, state: AgentState) -> AgentState:
        """
        Chạy toàn bộ workflow từ input đến output.

        Pipeline:
          1. ResearcherAgent — thu thập thông tin tuyển sinh từ URL/query
          2. PlannerAgent    — lập checklist + timeline, chặn Agent 3 nếu thiếu thông tin
          3. AdvisorAgent    — TODO
          4. ApplicationAgent — TODO (chỉ chạy sau khi học sinh xác nhận)

        Args:
            state: AgentState có student_profile và user_message.

        Returns:
            AgentState đã được cập nhật bởi các agent.
        """
        if not state.session_id:
            state.session_id = str(uuid.uuid4())

        logger.info(f"[Workflow] Bắt đầu session: {state.session_id}")

        # ── Bước 1: Researcher ────────────────────────────────────────────
        logger.info("[Workflow] → Chạy ResearcherAgent")
        state = await self.researcher.run(state)

        if state.errors:
            logger.warning(f"[Workflow] Lỗi sau Researcher: {state.errors}")

        # ── Bước 2: Planner ───────────────────────────────────────────
        if state.research_result and not state.errors:
            logger.info("[Workflow] → Chạy PlannerAgent")
            state = await self.planner.run(state)
        else:
            logger.warning(
                "[Workflow] Bỏ qua PlannerAgent do Researcher không có kết quả "
                "hoặc có lỗi."
            )

        # ── Bước 3: Advisor (TODO) ───────────────────────────────────
        # from app.agents.advisor import AdvisorAgent
        # state = await AdvisorAgent(verbose=self.verbose).run(state)

        # ── Bước 4: Application (TODO) ───────────────────────────────
        # Chỉ chạy sau khi học sinh xác nhận (state.requires_confirmation = False)
        # from app.agents.application_agent import ApplicationAgent
        # if not state.requires_confirmation:
        #     state = await ApplicationAgent(verbose=self.verbose).run(state)

        logger.info(
            f"[Workflow] Session {state.session_id} hoàn tất. "
            f"Agents đã chạy: {state.completed_agents}"
        )
        return state
    
    async def run_stream(self, state: AgentState) -> AsyncIterator[str]:
        """
        Streaming version — yield text chunks from Researcher Agent in real time.
        Used for WebSocket endpoint.
        """
        if not state.session_id:
            state.session_id = str(uuid.uuid4())
 
        async for chunk in self.researcher.run_stream(state):
            yield chunk

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