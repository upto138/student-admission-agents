"""
Test suite for Advisor Agent (Phase 1 MVP)

Verifies the full pipeline: Researcher → Planner → Advisor

How to run (from backend/ directory):
    python -m tests.test_advisor          # Test 1 (default) — full profile
    python -m tests.test_advisor 1        # Test 1 — full profile with URLs
    python -m tests.test_advisor 2        # Test 2 — minimal profile (missing info)
    python -m tests.test_advisor 3        # Test 3 — check only AdvisorResult fields
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.workflows.chat_workflow import ChatWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test cases
# =============================================================================

async def test_full_profile():
    """Test 1: Full student profile — Advisor should return high/medium confidence."""
    print("\n" + "=" * 60)
    print("TEST 1: Full pipeline — full student profile")
    print("=" * 60)

    state = ChatWorkflow.create_initial_state(
        user_message=(
            "Em muốn xét tuyển ngành Công nghệ thông tin. "
            "Em cần lời khuyên về kế hoạch nộp hồ sơ và các rủi ro cần chú ý."
        ),
        student_name="Nguyen Van A",
        gpa=8.5,
        target_scores={"Math": 9.0, "Physics": 8.5, "Chemistry": 8.0},
        preferred_universities=["Ho Chi Minh City University of Technology", "University of Science"],
        preferred_majors=["Computer Science", "Software Engineering"],
    )

    workflow = ChatWorkflow(verbose=True)
    result_state = await workflow.run(state)
    _print_result(result_state)


async def test_missing_profile():
    """Test 2: Minimal profile — Advisor should run but note missing info in caveats."""
    print("\n" + "=" * 60)
    print("TEST 2: Minimal profile (no GPA, no scores)")
    print("=" * 60)

    state = ChatWorkflow.create_initial_state(
        user_message="Em muốn học ngành Kỹ thuật phần mềm. Cho em lời khuyên.",
        student_name="Tran Thi B",
        # No GPA, no target_scores — Advisor should note this in caveats
        preferred_majors=["Software Engineering"],
    )

    workflow = ChatWorkflow(verbose=True)
    result_state = await workflow.run(state)
    _print_result(result_state)


async def test_advisor_result_structure():
    """Test 3: Verify AdvisorResult fields are correctly populated."""
    print("\n" + "=" * 60)
    print("TEST 3: Verify AdvisorResult structure")
    print("=" * 60)

    state = ChatWorkflow.create_initial_state(
        user_message="Cho em biết em cần chuẩn bị gì để nộp hồ sơ đại học.",
        student_name="Le Van C",
        gpa=9.0,
        target_scores={"Math": 9.5, "Literature": 8.0, "English": 9.0},
        preferred_majors=["Information Technology"],
        preferred_universities=["Vietnam National University"],
    )

    workflow = ChatWorkflow(verbose=True)
    result_state = await workflow.run(state)

    print("\n" + "─" * 50)
    print("AdvisorResult field check:")

    advisor = result_state.advisor_result
    if advisor is None:
        print("  ❌ advisor_result is None — Advisor did not run or failed!")
        if result_state.errors:
            print(f"  Errors: {result_state.errors}")
        return

    print(f"  ✅ advisor_result exists")
    print(f"  confidence_level : {advisor.confidence_level}")
    print(f"  narrative        : {len(advisor.narrative)} chars")
    print(f"  personalized_advice : {len(advisor.personalized_advice)} items")
    print(f"  risk_analysis    : {len(advisor.risk_analysis)} items")
    print(f"  action_steps     : {len(advisor.action_steps)} items")
    print(f"  caveats          : {len(advisor.caveats)} items")
    print(f"  advised_at       : {advisor.advised_at}")

    if advisor.risk_analysis:
        print("\n  First risk:")
        r = advisor.risk_analysis[0]
        print(f"    description : {r.description[:80]}")
        print(f"    severity    : {r.severity}")
        print(f"    mitigation  : {r.mitigation[:80]}")

    if advisor.action_steps:
        print("\n  First action step:")
        a = advisor.action_steps[0]
        print(f"    step         : {a.step[:80]}")
        print(f"    priority     : {a.priority}")
        print(f"    deadline_hint: {a.deadline_hint}")

    # Also check backward-compat field
    print(f"\n  advisor_response (backward compat): {len(result_state.advisor_response)} chars")
    print("─" * 50)


# =============================================================================
# Print helper
# =============================================================================

def _print_result(state):
    """Print the full pipeline result in readable format."""
    print(f"\n{'─'*60}")
    print(f"Session ID     : {state.session_id}")
    print(f"Agents done    : {state.completed_agents}")

    if state.errors:
        print(f"\n⚠️  Errors: {state.errors}")

    # ── Researcher result ──────────────────────────────────────
    result = state.research_result
    if result:
        print(f"\n[Researcher]")
        print(f"  Universities found : {len(result.universities)}")
        print(f"  Summary            : {result.raw_summary[:150]}...")
    else:
        print("\n[Researcher] ❌ No result")

    # ── Planner result ─────────────────────────────────────────
    plan = state.plan_result
    if plan:
        print(f"\n[Planner]")
        print(f"  Checklist items    : {len(plan.checklist)}")
        print(f"  Timeline tasks     : {len(plan.timeline)}")
        print(f"  Missing questions  : {plan.missing_questions or 'None'}")
        print(f"  Needs confirmation : {plan.needs_human_confirmation}")
    else:
        print("\n[Planner] ❌ No result")

    # ── Advisor result ─────────────────────────────────────────
    advisor = state.advisor_result
    if advisor:
        print(f"\n[Advisor]")
        print(f"  Confidence level   : {advisor.confidence_level}")
        print(f"  Risks identified   : {len(advisor.risk_analysis)}")
        print(f"  Action steps       : {len(advisor.action_steps)}")
        print(f"  Caveats            : {len(advisor.caveats)}")
        print(f"\n  --- Narrative (first 400 chars) ---")
        print(f"  {advisor.narrative[:400]}...")

        if advisor.caveats:
            print(f"\n  --- Caveats ---")
            for c in advisor.caveats:
                print(f"    • {c}")
    else:
        print("\n[Advisor] ❌ No result — check errors above")

    print(f"\n{'─'*60}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("Advisor Agent Test Suite — Phase 1 MVP")
    print(f"  FOUNDRY_PROJECT_ENDPOINT : {os.environ.get('FOUNDRY_PROJECT_ENDPOINT', 'NOT SET')}")
    print(f"  FOUNDRY_MODEL            : {os.environ.get('FOUNDRY_MODEL', 'NOT SET')}")
    print(f"  AZURE_SEARCH_ENDPOINT    : {os.environ.get('AZURE_SEARCH_ENDPOINT', 'NOT SET')}")

    test_num = sys.argv[1] if len(sys.argv) > 1 else "1"
    tests = {
        "1": test_full_profile,
        "2": test_missing_profile,
        "3": test_advisor_result_structure,
    }
    asyncio.run(tests.get(test_num, test_full_profile)())
