"""
Unit Tests — PlannerAgent (Agent 2)

These tests do NOT call any real LLM, RAG, or network services.
All external dependencies (FoundryChatClient, query_knowledge_base) are mocked.

Run:
    cd backend
    .venv\\Scripts\\python.exe -m pytest tests/unit/test_planner_unit.py -v

Coverage areas:
    - _parse_plan_result()   : JSON parsing strategies (4 strategies + fallback)
    - _dict_to_plan_result() : dict → PlanResult conversion with coercion
    - _build_user_message()  : message construction from AgentState
    - PlannerAgent.run()     : guard, HITL gate, LLM mock, state update
    - PlannerAgent toolset   : RAG tool wiring
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.planner import (
    ALL_PLANNER_TOOLS,
    PlannerAgent,
    PlannerToolset,
    _build_user_message,
    _dict_to_plan_result,
    _parse_plan_result,
)
from app.schemas.agent_state import (
    AgentState,
    ChecklistItem,
    PlanResult,
    ResearchResult,
    StudentProfile,
    TimelineTask,
    UniversityInfo,
)
from app.tools.rag_tool import ALL_RAG_TOOLS, query_knowledge_base


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def sample_research_result() -> ResearchResult:
    """Simulated ResearcherAgent output — does not call any real service."""
    return ResearchResult(
        query="tuyển sinh ĐHBK HCM 2026",
        universities=[
            UniversityInfo(
                university_name="Đại học Bách Khoa TP.HCM",
                major="Công nghệ thông tin",
                admission_method="xét điểm thi THPT",
                deadline="2026-05-20",
                required_documents=["CCCD", "Học bạ THPT", "Ảnh 3x4"],
                benchmark_score=24.5,
                source_url="https://vnexpress.net/test",
            )
        ],
        general_requirements=["Tốt nghiệp THPT"],
        important_deadlines={"Hạn nộp hồ sơ": "2026-05-20"},
        admission_methods=["Xét điểm thi THPT", "Xét học bạ"],
        raw_summary="Thông tin tuyển sinh ĐHBK HCM 2026",
        sources=["https://vnexpress.net/test"],
    )


@pytest.fixture
def full_student_profile() -> StudentProfile:
    """Complete StudentProfile — all required fields filled."""
    return StudentProfile(
        name="Nguyễn Văn An",
        gpa=8.5,
        preferred_majors=["Công nghệ thông tin"],
        preferred_universities=["Đại học Bách Khoa TP.HCM"],
        target_scores={"Toán": 9.0, "Lý": 8.5, "Hóa": 8.0},
    )


@pytest.fixture
def incomplete_student_profile() -> StudentProfile:
    """StudentProfile missing name and GPA — should trigger missing_questions."""
    return StudentProfile(
        name="",
        gpa=None,
        preferred_majors=["Công nghệ thông tin"],
    )


@pytest.fixture
def valid_plan_json() -> dict:
    """A valid PlanResult payload as a raw dict — represents expected LLM output."""
    return {
        "missing_questions": [],
        "checklist": [
            {
                "title": "CCCD bản scan màu",
                "status": "pending",
                "required": True,
                "reason": "Tin tuyển sinh yêu cầu giấy tờ định danh.",
                "source": "admission_news",
            },
            {
                "title": "Học bạ THPT bản sao công chứng",
                "status": "ready",
                "required": True,
                "reason": "Tin tuyển sinh yêu cầu kết quả học tập.",
                "source": "admission_news",
            },
            {
                "title": "Chứng chỉ IELTS",
                "status": "need_review",
                "required": False,
                "reason": "Không rõ có bắt buộc không.",
                "source": "inferred",
            },
        ],
        "timeline": [
            {
                "date": "2026-05-15",
                "task": "Hoàn tất scan hồ sơ",
                "priority": "high",
                "reason": "Deadline nộp là 20/05.",
            },
            {
                "date": None,
                "task": "Xin xác nhận tốt nghiệp THPT as soon as possible",
                "priority": "medium",
                "reason": "Deadline chưa rõ từ Researcher.",
            },
        ],
        "risks": ["Nếu thiếu học bạ sẽ không nộp được hồ sơ đúng hạn."],
        "needs_human_confirmation": False,
    }


# =============================================================================
# Group 1: _parse_plan_result — 4-strategy JSON parsing
# =============================================================================

class TestParsePlanResult:
    """Tests for the 4-strategy JSON extraction logic."""

    def test_strategy1_pure_json(self, valid_plan_json):
        """Strategy 1: entire text is pure JSON — must parse directly."""
        raw = json.dumps(valid_plan_json, ensure_ascii=False)
        result = _parse_plan_result(raw)

        assert isinstance(result, PlanResult)
        assert len(result.checklist) == 3
        assert result.checklist[0].title == "CCCD bản scan màu"
        assert result.checklist[0].source == "admission_news"
        assert result.checklist[2].source == "inferred"
        assert result.checklist[2].status == "need_review"
        assert len(result.timeline) == 2
        assert result.timeline[0].date == "2026-05-15"
        assert result.timeline[1].date is None          # null deadline preserved
        assert result.needs_human_confirmation is False
        assert result.missing_questions == []

    def test_strategy2_fenced_json_block(self, valid_plan_json):
        """Strategy 2: JSON wrapped in ```json ... ``` fenced code block."""
        json_str = json.dumps(valid_plan_json, ensure_ascii=False)
        raw = f"Đây là kế hoạch:\n\n```json\n{json_str}\n```\n\nHy vọng hữu ích!"
        result = _parse_plan_result(raw)

        assert isinstance(result, PlanResult)
        assert len(result.checklist) == 3
        assert result.checklist[0].source == "admission_news"

    def test_strategy2_fenced_block_no_language_tag(self, valid_plan_json):
        """Strategy 2: fenced block without the 'json' language tag."""
        json_str = json.dumps(valid_plan_json, ensure_ascii=False)
        raw = f"Kết quả:\n```\n{json_str}\n```"
        result = _parse_plan_result(raw)

        assert isinstance(result, PlanResult)
        assert len(result.checklist) == 3

    def test_strategy3_json_embedded_in_text(self, valid_plan_json):
        """Strategy 3: JSON object embedded mid-text with no fences."""
        json_str = json.dumps(valid_plan_json, ensure_ascii=False)
        raw = f"Agent phân tích xong rồi. Đây là output: {json_str} Hết rồi."
        result = _parse_plan_result(raw)

        assert isinstance(result, PlanResult)
        assert len(result.checklist) == 3

    def test_strategy4_fallback_returns_empty_plan(self):
        """Strategy 4 (fallback): unparseable text must return empty PlanResult, not crash."""
        raw = "Xin lỗi, tôi không thể phân tích thông tin này vào lúc này."
        result = _parse_plan_result(raw)

        assert isinstance(result, PlanResult)
        assert result.checklist == []
        assert result.timeline == []
        assert result.missing_questions == []
        assert result.risks == []
        assert result.needs_human_confirmation is False

    def test_empty_string_returns_empty_plan(self):
        """Empty input must return empty PlanResult without crashing."""
        assert isinstance(_parse_plan_result(""), PlanResult)
        assert isinstance(_parse_plan_result("   "), PlanResult)


# =============================================================================
# Group 2: _dict_to_plan_result — coercion and robustness
# =============================================================================

class TestDictToPlanResult:
    """Tests for the dict → PlanResult conversion with type coercion."""

    def test_all_fields_present(self, valid_plan_json):
        """All fields present — must map 1:1 without data loss."""
        result = _dict_to_plan_result(valid_plan_json)

        assert len(result.checklist) == 3
        assert len(result.timeline) == 2
        assert len(result.risks) == 1
        assert result.needs_human_confirmation is False

    def test_checklist_items_have_source(self, valid_plan_json):
        """Every ChecklistItem must have a non-empty source field."""
        result = _dict_to_plan_result(valid_plan_json)

        for item in result.checklist:
            assert item.source != "", (
                f"ChecklistItem '{item.title}' is missing source. "
                f"Valid values: admission_news | rag | student_profile | inferred"
            )

    def test_null_timeline_date_preserved(self, valid_plan_json):
        """TimelineTask with date=None must not be coerced to a string."""
        result = _dict_to_plan_result(valid_plan_json)

        null_tasks = [t for t in result.timeline if t.date is None]
        assert len(null_tasks) >= 1, (
            "Expected at least one TimelineTask with date=None. "
            "Planner must not invent dates when Researcher returns null deadline."
        )

    def test_malformed_checklist_item_is_skipped(self):
        """A checklist item that is not a dict must be silently skipped."""
        data = {
            "checklist": [
                "not a dict",                              # invalid — must be skipped
                {"title": "CCCD", "source": "admission_news"},  # valid
            ],
            "timeline": [],
            "risks": [],
            "needs_human_confirmation": False,
        }
        result = _dict_to_plan_result(data)
        assert len(result.checklist) == 1
        assert result.checklist[0].title == "CCCD"

    def test_missing_optional_fields_use_defaults(self):
        """Missing optional fields must fall back to schema defaults."""
        data = {
            "checklist": [{"title": "Học bạ", "source": "admission_news"}],
        }
        result = _dict_to_plan_result(data)

        assert result.missing_questions == []
        assert result.timeline == []
        assert result.risks == []
        assert result.needs_human_confirmation is False
        assert result.checklist[0].status == "pending"      # default
        assert result.checklist[0].required is True         # default

    def test_needs_human_confirmation_string_coercion(self):
        """String 'true'/'false' must be coerced to bool correctly."""
        data_true = {"needs_human_confirmation": "true", "checklist": []}
        data_false = {"needs_human_confirmation": "false", "checklist": []}

        assert _dict_to_plan_result(data_true).needs_human_confirmation is True
        assert _dict_to_plan_result(data_false).needs_human_confirmation is False

    def test_missing_questions_as_string_coerced_to_list(self):
        """missing_questions as a bare string must be wrapped into a list."""
        data = {
            "missing_questions": "Bạn tên gì?",
            "checklist": [],
        }
        result = _dict_to_plan_result(data)
        assert result.missing_questions == ["Bạn tên gì?"]


# =============================================================================
# Group 3: _build_user_message — content coverage
# =============================================================================

class TestBuildUserMessage:
    """Tests for the user message builder that feeds the LLM prompt."""

    def test_includes_university_and_profile(
        self, sample_research_result, full_student_profile
    ):
        """Message must include key research fields and student name."""
        state = AgentState(
            session_id="test-msg",
            research_result=sample_research_result,
            student_profile=full_student_profile,
        )
        message = _build_user_message(state)

        assert "Bách Khoa" in message
        assert "Nguyễn Văn An" in message
        assert "PlanResult" in message
        assert "missing_questions" in message
        assert "source" in message

    def test_includes_deadline_info(self, sample_research_result, full_student_profile):
        """Message must include deadline data from research_result."""
        state = AgentState(
            research_result=sample_research_result,
            student_profile=full_student_profile,
        )
        message = _build_user_message(state)
        assert "2026-05-20" in message

    def test_marks_missing_name_as_placeholder(
        self, sample_research_result, incomplete_student_profile
    ):
        """Incomplete profile must have [CHƯA CÓ] placeholder for missing fields."""
        state = AgentState(
            research_result=sample_research_result,
            student_profile=incomplete_student_profile,
        )
        message = _build_user_message(state)
        assert "CHƯA CÓ" in message

    def test_empty_universities_does_not_crash(self):
        """_build_user_message must not crash when research_result has no universities."""
        state = AgentState(
            research_result=ResearchResult(raw_summary="Không tìm thấy trường nào.", sources=[]),
            student_profile=StudentProfile(),
        )
        message = _build_user_message(state)
        assert isinstance(message, str)
        assert len(message) > 0


# =============================================================================
# Group 4: PlannerAgent.run() — with mocked LLM
# =============================================================================

class TestPlannerAgentRun:
    """Tests for PlannerAgent.run() — LLM is mocked, no real API calls."""

    def _make_state(self, research_result=None, profile=None) -> AgentState:
        return AgentState(
            session_id="unit-test",
            user_message="Em muốn nộp hồ sơ ĐHBK",
            research_result=research_result,
            student_profile=profile or StudentProfile(name="Test Student"),
        )

    @pytest.mark.asyncio
    async def test_guard_no_research_result(self):
        """run() must return early with an error if research_result is None."""
        agent = PlannerAgent(verbose=False)
        state = self._make_state(research_result=None)

        result = await agent.run(state)

        assert result.plan_result is None
        assert len(result.errors) > 0
        assert "planner" not in result.completed_agents

    @pytest.mark.asyncio
    async def test_successful_run_populates_plan_result(
        self, sample_research_result, valid_plan_json
    ):
        """
        run() with a mocked LLM returning valid JSON must populate plan_result
        and mark the agent as done.
        """
        agent = PlannerAgent(verbose=False)
        state = self._make_state(research_result=sample_research_result)

        # Mock the entire Foundry Agent — return valid JSON as response text
        mock_response = MagicMock()
        mock_response.text = json.dumps(valid_plan_json, ensure_ascii=False)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_response)

        with patch.object(agent, "_get_agent", return_value=mock_agent):
            result = await agent.run(state)

        assert result.plan_result is not None
        assert len(result.plan_result.checklist) == 3
        assert len(result.plan_result.timeline) == 2
        assert "planner" in result.completed_agents
        assert result.requires_confirmation is False

    @pytest.mark.asyncio
    async def test_hitl_gate_missing_questions(self, sample_research_result):
        """
        When LLM returns missing_questions, run() must set:
          state.requires_confirmation = True
          state.application_status = 'pending_info'
        """
        agent = PlannerAgent(verbose=False)
        state = self._make_state(research_result=sample_research_result)

        plan_with_questions = {
            "missing_questions": ["Bạn vui lòng cho biết họ và tên đầy đủ?"],
            "checklist": [],
            "timeline": [],
            "risks": [],
            "needs_human_confirmation": True,
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(plan_with_questions, ensure_ascii=False)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_response)

        with patch.object(agent, "_get_agent", return_value=mock_agent):
            result = await agent.run(state)

        assert result.requires_confirmation is True
        assert result.application_status == "pending_info"
        assert len(result.plan_result.missing_questions) == 1

    @pytest.mark.asyncio
    async def test_hitl_gate_needs_confirmation_only(self, sample_research_result):
        """
        When missing_questions is empty but needs_human_confirmation=True,
        application_status must be 'pending_confirmation' (not 'pending_info').
        """
        agent = PlannerAgent(verbose=False)
        state = self._make_state(research_result=sample_research_result)

        payload = {
            "missing_questions": [],
            "checklist": [{"title": "CCCD", "source": "admission_news"}],
            "timeline": [],
            "risks": [],
            "needs_human_confirmation": True,
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(payload, ensure_ascii=False)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_response)

        with patch.object(agent, "_get_agent", return_value=mock_agent):
            result = await agent.run(state)

        assert result.requires_confirmation is True
        assert result.application_status == "pending_confirmation"

    @pytest.mark.asyncio
    async def test_llm_returns_garbage_gives_empty_plan(self, sample_research_result):
        """
        When LLM returns unstructured text (no JSON), run() must not crash.
        plan_result must be an empty PlanResult.
        """
        agent = PlannerAgent(verbose=False)
        state = self._make_state(research_result=sample_research_result)

        mock_response = MagicMock()
        mock_response.text = "Xin lỗi, tôi không hiểu yêu cầu."
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_response)

        with patch.object(agent, "_get_agent", return_value=mock_agent):
            result = await agent.run(state)

        assert result.plan_result is not None
        assert result.plan_result.checklist == []
        assert result.plan_result.timeline == []
        # Agent is still marked done even with fallback result
        assert "planner" in result.completed_agents

    @pytest.mark.asyncio
    async def test_exception_during_llm_call_adds_error(self, sample_research_result):
        """
        If the LLM call raises an exception, run() must catch it,
        add an error to state, and return a default empty PlanResult.
        """
        agent = PlannerAgent(verbose=False)
        state = self._make_state(research_result=sample_research_result)

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Connection timeout"))

        with patch.object(agent, "_get_agent", return_value=mock_agent):
            result = await agent.run(state)

        assert len(result.errors) > 0
        assert "PlannerAgent" in result.errors[0]
        assert result.plan_result is not None   # must not be None — default empty plan


# =============================================================================
# Group 5: Toolset wiring
# =============================================================================

class TestPlannerToolset:
    """Tests verifying the RAG tool is correctly wired into PlannerAgent."""

    def test_toolset_contains_query_knowledge_base(self):
        """PlannerToolset.tools must include query_knowledge_base (v2 — RAG connected)."""
        assert query_knowledge_base in PlannerToolset.tools, (
            "PlannerToolset.tools must contain query_knowledge_base. "
            "Check: PlannerToolset.tools = ALL_RAG_TOOLS in planner.py."
        )

    def test_all_planner_tools_equals_all_rag_tools(self):
        """ALL_PLANNER_TOOLS module-level export must equal ALL_RAG_TOOLS."""
        assert ALL_PLANNER_TOOLS == ALL_RAG_TOOLS

    def test_extra_tools_merged_on_top_of_rag_defaults(self):
        """PlannerAgent must merge extra_tools ON TOP of the default RAG toolset."""
        dummy = lambda: None
        dummy.__name__ = "dummy_tool"

        agent = PlannerAgent(verbose=False, extra_tools=[dummy])

        assert dummy in agent._tools
        assert len(agent._tools) == len(ALL_PLANNER_TOOLS) + 1
        for rag_tool in ALL_RAG_TOOLS:
            assert rag_tool in agent._tools, (
                f"Default RAG tool '{rag_tool.__name__}' missing after extra_tools injection."
            )

    def test_agent_without_extra_tools_has_only_rag(self):
        """PlannerAgent with no extra_tools must have exactly ALL_PLANNER_TOOLS."""
        agent = PlannerAgent(verbose=False)
        assert agent._tools == ALL_PLANNER_TOOLS
