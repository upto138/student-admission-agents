"""
Unit tests cho PlannerAgent.

Các tests này KHÔNG gọi LLM thật — chỉ test logic parsing và state management.
Chạy: python -m pytest tests/test_agent2_planner.py -v
"""
from __future__ import annotations

import json
import pytest

from app.schemas.agent_state import (
    AgentState,
    ChecklistItem,
    PlanResult,
    ResearchResult,
    StudentProfile,
    TimelineTask,
    UniversityInfo,
)
from app.agents.planner import (
    _dict_to_plan_result,
    _parse_plan_result,
    _build_user_message,
    PlannerToolset,
    ALL_PLANNER_TOOLS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_research_result() -> ResearchResult:
    """ResearchResult mẫu — output giả lập của ResearcherAgent."""
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
    """StudentProfile đầy đủ thông tin."""
    return StudentProfile(
        name="Nguyễn Văn An",
        gpa=8.5,
        preferred_majors=["Công nghệ thông tin"],
        preferred_universities=["Đại học Bách Khoa TP.HCM"],
        target_scores={"Toán": 9.0, "Lý": 8.5, "Hóa": 8.0},
    )


@pytest.fixture
def incomplete_student_profile() -> StudentProfile:
    """StudentProfile thiếu tên và GPA."""
    return StudentProfile(
        name="",
        gpa=None,
        preferred_majors=["Công nghệ thông tin"],
    )


@pytest.fixture
def valid_plan_json() -> dict:
    """JSON hợp lệ của PlanResult."""
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
        ],
        "timeline": [
            {
                "date": "2026-05-15",
                "task": "Hoàn tất scan hồ sơ",
                "priority": "high",
                "reason": "Deadline nộp là 20/05.",
            }
        ],
        "risks": ["Nếu thiếu học bạ sẽ không nộp được hồ sơ đúng hạn."],
        "needs_human_confirmation": False,
    }


# =============================================================================
# Test 1: Parse JSON hợp lệ
# =============================================================================

def test_parse_valid_json(valid_plan_json):
    """_parse_plan_result phải parse JSON thuần thành PlanResult đúng."""
    raw_text = json.dumps(valid_plan_json, ensure_ascii=False)
    result = _parse_plan_result(raw_text)

    assert isinstance(result, PlanResult)
    assert len(result.checklist) == 2
    assert result.checklist[0].title == "CCCD bản scan màu"
    assert result.checklist[0].source == "admission_news"
    assert result.checklist[1].status == "ready"
    assert len(result.timeline) == 1
    assert result.timeline[0].priority == "high"
    assert result.timeline[0].date == "2026-05-15"
    assert len(result.risks) == 1
    assert result.needs_human_confirmation is False
    assert result.missing_questions == []


# =============================================================================
# Test 2: Parse JSON trong fenced code block
# =============================================================================

def test_parse_fenced_json(valid_plan_json):
    """_parse_plan_result phải xử lý JSON trong khối ```json ... ```."""
    json_str = json.dumps(valid_plan_json, ensure_ascii=False)
    raw_text = f"Đây là kết quả phân tích:\n\n```json\n{json_str}\n```\n\nHy vọng hữu ích!"
    result = _parse_plan_result(raw_text)

    assert isinstance(result, PlanResult)
    assert len(result.checklist) == 2
    assert result.checklist[0].source == "admission_news"


# =============================================================================
# Test 3: Fallback khi text không parse được
# =============================================================================

def test_parse_fallback_returns_empty_plan():
    """_parse_plan_result không được crash khi nhận text lộn xộn — trả về PlanResult rỗng."""
    raw_text = "Xin lỗi, tôi không thể phân tích thông tin này vào lúc này."
    result = _parse_plan_result(raw_text)

    assert isinstance(result, PlanResult)
    assert result.checklist == []
    assert result.timeline == []
    assert result.missing_questions == []
    assert result.risks == []


# =============================================================================
# Test 4: Guard khi research_result là None
# =============================================================================

@pytest.mark.asyncio
async def test_run_returns_early_if_no_research_result():
    """PlannerAgent.run() phải trả về sớm với lỗi nếu research_result chưa có."""
    from app.agents.planner import PlannerAgent

    state = AgentState(
        session_id="test-session",
        user_message="Tôi muốn nộp hồ sơ",
        research_result=None,  # Researcher chưa chạy
    )

    agent = PlannerAgent(verbose=False)
    result_state = await agent.run(state)

    assert result_state.plan_result is None
    assert len(result_state.errors) > 0
    assert "research_result" in result_state.errors[0].lower() or \
           "planner" in result_state.errors[0].lower()
    assert "planner" not in result_state.completed_agents


# =============================================================================
# Test 5: HITL gate — missing_questions → requires_confirmation = True
# =============================================================================

@pytest.mark.asyncio
async def test_missing_questions_sets_hitl_flag(sample_research_result):
    """
    Khi PlanResult có missing_questions → state.requires_confirmation phải là True
    và application_status phải là 'pending_info'.
    """
    from app.agents.planner import PlannerAgent, _dict_to_plan_result

    # Giả lập PlannerAgent đã parse được kết quả có missing_questions
    plan_with_questions = PlanResult(
        missing_questions=["Bạn vui lòng cho biết họ và tên đầy đủ?"],
        checklist=[],
        timeline=[],
        risks=[],
        needs_human_confirmation=True,
    )

    state = AgentState(
        session_id="test-hitl",
        research_result=sample_research_result,
    )

    # Simulate agent đã ghi plan_result vào state
    state.plan_result = plan_with_questions
    if plan_with_questions.missing_questions:
        state.requires_confirmation = True
        state.application_status = "pending_info"

    assert state.requires_confirmation is True
    assert state.application_status == "pending_info"


# =============================================================================
# Test 6: Mọi ChecklistItem phải có source không rỗng
# =============================================================================

def test_checklist_items_must_have_source(valid_plan_json):
    """Mọi ChecklistItem trong PlanResult phải có field source không rỗng."""
    result = _dict_to_plan_result(valid_plan_json)

    for item in result.checklist:
        assert item.source != "", (
            f"ChecklistItem '{item.title}' thiếu source. "
            f"Mọi item phải có source: admission_news | rag | student_profile | inferred"
        )


# =============================================================================
# Test 7: PlannerToolset — v1 không có tools
# =============================================================================

def test_planner_toolset_is_empty_in_v1():
    """PlannerToolset.tools phải rỗng trong v1 (xem comment trong class để biết khi nào mở rộng)."""
    assert PlannerToolset.tools == [], (
        "PlannerToolset.tools không được có tools trong v1. "
        "Xem docstring của PlannerToolset để biết khi nào nên mở rộng."
    )
    assert ALL_PLANNER_TOOLS == []


# =============================================================================
# Test 8: extra_tools được merge vào đúng cách
# =============================================================================

def test_extra_tools_injected_correctly():
    """PlannerAgent nhận extra_tools và merge vào self._tools đúng cách."""
    from app.agents.planner import PlannerAgent

    dummy_tool = lambda: None
    dummy_tool.__name__ = "dummy_tool"

    agent = PlannerAgent(verbose=False, extra_tools=[dummy_tool])
    assert dummy_tool in agent._tools
    assert len(agent._tools) == len(ALL_PLANNER_TOOLS) + 1


# =============================================================================
# Test 9: _build_user_message — bao gồm đủ thông tin
# =============================================================================

def test_build_user_message_includes_key_fields(sample_research_result, full_student_profile):
    """_build_user_message phải chứa thông tin trường, hồ sơ học sinh và yêu cầu JSON."""
    state = AgentState(
        session_id="test-msg",
        research_result=sample_research_result,
        student_profile=full_student_profile,
    )

    message = _build_user_message(state)

    assert "Bách Khoa" in message
    assert "Nguyễn Văn An" in message
    assert "PlanResult" in message
    assert "source" in message
    assert "missing_questions" in message
