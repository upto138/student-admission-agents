"""
Integration Test — PlannerAgent (Agent 2) — Manual Runner

These tests call REAL services:
  - Azure AI Foundry (LLM: FOUNDRY_PROJECT_ENDPOINT + FOUNDRY_MODEL)
  - RAG knowledge base (query_knowledge_base via ChromaDB or Azure AI Search)

Required environment variables in .env:
    FOUNDRY_PROJECT_ENDPOINT=...
    FOUNDRY_MODEL=gpt-4o-admission
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_ENDPOINT=...
    AZURE_EMBEDDING_DEPLOYMENT=embedding-admission
    RAG_BACKEND=chroma   (or "azure" for production)

How to run:
    cd backend
    .venv\\Scripts\\python.exe -m tests.integration.test_planner_integration         # default (test 2)
    .venv\\Scripts\\python.exe -m tests.integration.test_planner_integration 1       # full profile, known deadline
    .venv\\Scripts\\python.exe -m tests.integration.test_planner_integration 2       # incomplete profile → HITL gate
    .venv\\Scripts\\python.exe -m tests.integration.test_planner_integration 3       # planner only (skip researcher)

WARNING:
    Each run makes real LLM API calls and consumes tokens.
    Do NOT run this in CI/CD pipelines.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

# Allow running as: python -m tests.integration.test_planner_integration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from app.agents.planner import PlannerAgent
from app.schemas.agent_state import AgentState, ResearchResult, StudentProfile, UniversityInfo
from app.workflows.chat_workflow import ChatWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Integration Test 1: Full pipeline — complete student profile
# =============================================================================

async def test_full_pipeline_complete_profile():
    """
    Test 1: Run the full Researcher → Planner pipeline with a complete student profile.

    Expected:
    - plan_result.checklist is non-empty
    - plan_result.missing_questions is empty (profile is complete)
    - state.requires_confirmation is False
    - Each ChecklistItem has a valid source
    """
    print("\n" + "=" * 60)
    print("INTEGRATION TEST 1: Full pipeline — complete profile")
    print("=" * 60)

    state = ChatWorkflow.create_initial_state(
        user_message=(
            "Em tên Nguyễn Văn An, GPA 8.5. "
            "Em muốn xét tuyển ngành Công nghệ thông tin tại ĐHBK HCM năm 2026. "
            "Tìm thông tin hồ sơ cần chuẩn bị và timeline cụ thể."
        ),
        student_name="Nguyễn Văn An",
        gpa=8.5,
        target_scores={"Toán": 9.0, "Lý": 8.5, "Hóa": 8.0},
        preferred_universities=["Đại học Bách Khoa TP.HCM"],
        preferred_majors=["Công nghệ thông tin"],
    )

    workflow = ChatWorkflow(verbose=True)
    result_state = await workflow.run(state)
    _print_result(result_state)


# =============================================================================
# Integration Test 2: HITL gate — incomplete student profile
# =============================================================================

async def test_hitl_gate_incomplete_profile():
    """
    Test 2: Run the full pipeline with an incomplete profile (no name, no GPA).

    Expected:
    - plan_result.missing_questions is NOT empty
    - state.requires_confirmation is True
    - state.application_status == 'pending_info'
    """
    print("\n" + "=" * 60)
    print("INTEGRATION TEST 2: HITL gate — incomplete profile")
    print("=" * 60)

    state = ChatWorkflow.create_initial_state(
        user_message=(
            "Em muốn xét tuyển ngành Kỹ thuật phần mềm tại ĐHQG TP.HCM. "
            "Cho em biết hồ sơ cần chuẩn bị."
            # No name, no GPA, no scores → should trigger missing_questions
        ),
        preferred_universities=["Đại học Quốc gia TP.HCM"],
        preferred_majors=["Kỹ thuật phần mềm"],
    )

    workflow = ChatWorkflow(verbose=True)
    result_state = await workflow.run(state)
    _print_result(result_state)

    # Manual check — developer verifies console output
    if result_state.plan_result:
        if result_state.plan_result.missing_questions:
            print("\n✅ HITL gate triggered correctly — missing_questions detected.")
        else:
            print("\n⚠️  WARNING: Profile was incomplete but missing_questions is empty.")

        if result_state.requires_confirmation:
            print("✅ state.requires_confirmation = True")
        else:
            print("⚠️  WARNING: requires_confirmation is False despite missing questions.")


# =============================================================================
# Integration Test 3: PlannerAgent only (skip Researcher, inject mock research data)
# =============================================================================

async def test_planner_only_with_injected_research():
    """
    Test 3: Run PlannerAgent directly with pre-built research data (no Researcher call).

    Use this test to:
    - Isolate Planner behavior from Researcher quality
    - Verify RAG tool (query_knowledge_base) is called when appropriate
    - Debug Planner prompt/output issues without consuming Researcher tokens

    Expected:
    - plan_result.checklist is non-empty
    - Each ChecklistItem has source set (admission_news | rag | student_profile | inferred)
    - Timeline tasks with null deadline have date = None
    """
    print("\n" + "=" * 60)
    print("INTEGRATION TEST 3: Planner only — injected research data")
    print("=" * 60)

    # Simulated Researcher output — inject directly into state
    research_result = ResearchResult(
        query="tuyển sinh ĐHBK HCM 2026 CNTT",
        universities=[
            UniversityInfo(
                university_name="Đại học Bách Khoa TP.HCM",
                major="Công nghệ thông tin",
                admission_method="Xét điểm thi THPT Quốc gia",
                deadline="2026-05-20",
                required_documents=["CCCD bản sao công chứng", "Học bạ THPT", "Ảnh 3x4 cm"],
                benchmark_score=24.5,
                website="https://hcmut.edu.vn",
                source_url="https://vnexpress.net/tuyen-sinh-bk-2026",
                notes="Điểm chuẩn trên thang 30",
            ),
        ],
        general_requirements=["Tốt nghiệp THPT năm 2026"],
        important_deadlines={"Hạn nộp hồ sơ": "2026-05-20"},
        admission_methods=["Xét điểm thi THPT", "Xét học bạ THPT"],
        raw_summary=(
            "ĐHBK HCM tuyển sinh ngành CNTT năm 2026. "
            "Điểm chuẩn 2025 là 24.5. Hạn nộp hồ sơ 20/05/2026."
        ),
        sources=["https://vnexpress.net/tuyen-sinh-bk-2026"],
    )

    profile = StudentProfile(
        name="Trần Thị B",
        gpa=8.7,
        target_scores={"Toán": 9.0, "Lý": 8.5, "Hóa": 8.5},
        preferred_majors=["Công nghệ thông tin"],
        preferred_universities=["Đại học Bách Khoa TP.HCM"],
    )

    state = AgentState(
        session_id="integration-test-3",
        user_message="Em muốn chuẩn bị hồ sơ vào ĐHBK HCM CNTT 2026.",
        research_result=research_result,
        student_profile=profile,
    )

    planner = PlannerAgent(verbose=True)
    result_state = await planner.run(state)
    _print_result(result_state)


# =============================================================================
# Shared: pretty-print helper
# =============================================================================

def _print_result(state: AgentState) -> None:
    """Print AgentState result to console in human-readable format."""
    print(f"\n{'─' * 55}")
    print(f"Session ID     : {state.session_id}")
    print(f"Agents done    : {state.completed_agents}")
    print(f"HITL flag      : requires_confirmation = {state.requires_confirmation}")
    print(f"App status     : {state.application_status or '(empty)'}")

    if state.errors:
        print(f"\n⛔ Errors: {state.errors}")

    plan = state.plan_result
    if not plan:
        print("\n⚠️  No plan_result in state.")
        return

    print(f"\nPlanResult:")
    print(f"  Checklist items  : {len(plan.checklist)}")
    print(f"  Timeline tasks   : {len(plan.timeline)}")
    print(f"  Missing questions: {len(plan.missing_questions)}")
    print(f"  Risks            : {len(plan.risks)}")
    print(f"  Needs confirmation: {plan.needs_human_confirmation}")

    if plan.missing_questions:
        print("\n  ❓ Missing questions:")
        for q in plan.missing_questions:
            print(f"    - {q}")

    if plan.checklist:
        print("\n  📋 Checklist (first 5):")
        for item in plan.checklist[:5]:
            status_icon = {"ready": "✅", "pending": "⏳", "need_review": "🔍"}.get(
                item.status, "❓"
            )
            req = "required" if item.required else "optional"
            print(f"    {status_icon} [{item.source}] {item.title} ({req})")

    if plan.timeline:
        print("\n  📅 Timeline (first 5):")
        for task in plan.timeline[:5]:
            date_str = task.date or "ASAP (null deadline)"
            print(f"    [{task.priority.upper()}] {date_str} — {task.task}")

    if plan.risks:
        print("\n  ⚠️  Risks:")
        for risk in plan.risks:
            print(f"    - {risk}")

    print(f"{'─' * 55}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("PlannerAgent Integration Test Suite")
    print(f"  FOUNDRY_PROJECT_ENDPOINT : {os.environ.get('FOUNDRY_PROJECT_ENDPOINT', 'NOT SET')}")
    print(f"  FOUNDRY_MODEL            : {os.environ.get('FOUNDRY_MODEL', 'NOT SET')}")
    print(f"  RAG_BACKEND              : {os.environ.get('RAG_BACKEND', 'chroma (default)')}")
    print(f"  AZURE_SEARCH_ENDPOINT    : {os.environ.get('AZURE_SEARCH_ENDPOINT', 'NOT SET')}")

    test_num = sys.argv[1] if len(sys.argv) > 1 else "2"
    tests = {
        "1": test_full_pipeline_complete_profile,
        "2": test_hitl_gate_incomplete_profile,
        "3": test_planner_only_with_injected_research,
    }

    selected = tests.get(test_num, test_hitl_gate_incomplete_profile)
    asyncio.run(selected())
