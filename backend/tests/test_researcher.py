"""
Test suite for Researcher Agent

How to run:
    cd backend
    python -m tests.test_researcher        # default test
    python -m tests.test_researcher 1      # test search index
    python -m tests.test_researcher 2      # test query benchmarks
    python -m tests.test_researcher 3      # full test profile + stream
"""
from __future__ import annotations

import asyncio
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


async def test_search_index():
    """Test 1: Researcher searches within Azure AI Search index."""
    print("\n" + "=" * 60)
    print("TEST 1: Search in Azure AI Search index")
    print("=" * 60)

    state = ChatWorkflow.create_initial_state(
        user_message=(
            "Tìm thông tin điểm chuẩn và phương thức xét tuyển "
            "ngành Công nghệ thông tin năm 2024 từ index."
        ),
        student_name="Nguyễn Văn A",
        preferred_majors=["Công nghệ thông tin"],
        preferred_universities=["Đại học Bách Khoa Hà Nội"],
    )

    workflow = ChatWorkflow(verbose=True)
    result_state = await workflow.run(state)
    _print_result(result_state)


async def test_query_with_profile():
    """Test 2: Query the benchmarks with complete profile student."""
    print("\n" + "=" * 60)
    print("TEST 2: Query with student profile")
    print("=" * 60)

    state = ChatWorkflow.create_initial_state(
        user_message=(
            "Em có điểm Toán 9.0, Lý 8.5, Hóa 8.0. "
            "Tìm thông tin điểm chuẩn ngành CNTT và Kỹ thuật phần mềm "
            "tại Đại học Khoa học tự nhiên và ĐHQG TP.HCM năm 2025."
        ),
        student_name="Trần Thị B",
        gpa=8.5,
        target_scores={"Toán": 9.0, "Lý": 8.5, "Hóa": 8.0},
        preferred_universities=["Đại học Khoa học tự nhiên", "Đại học Bách khoa"],
        preferred_majors=["Công nghệ thông tin", "Khoa học máy tính"],
    )

    workflow = ChatWorkflow(verbose=True)
    result_state = await workflow.run(state)
    _print_result(result_state)


async def test_stream():
    """Test 3: Streaming response from Researcher Agent."""
    print("\n" + "=" * 60)
    print("TEST 3: Streaming response")
    print("=" * 60)

    state = ChatWorkflow.create_initial_state(
        user_message=(
            "Tìm thông tin tuyển sinh ngành Y khoa năm 2024 "
            "tại Đại học Y Hà Nội và Đại học Y Dược TP.HCM."
        ),
        student_name="Lê Văn C",
        gpa=9.0,
        target_scores={"Toán": 9.0, "Hóa": 9.5, "Sinh": 9.0},
        preferred_majors=["Y khoa"],
        preferred_universities=["Đại học Y Hà Nội", "Đại học Y Dược TP.HCM"],
    )

    workflow = ChatWorkflow(verbose=True)

    print("\nStreaming response:\n")
    async for chunk in workflow.run_stream(state):
        print(chunk, end="", flush=True)
    print("\n\nStream complete.")


def _print_result(state):
    """Print the results to the console in an easy-to-read format."""
    print(f"\n{'─'*50}")
    print(f"Session ID  : {state.session_id}")
    print(f"Agents done : {state.completed_agents}")

    if state.errors:
        print(f"Errors: {state.errors}")

    result = state.research_result
    if not result:
        print("No research result.")
        return

    print(f"\nResearchResult:")
    print(f"   Query          : {result.query[:80] if result.query else 'N/A'}")
    if result.universities:
        uni_names = ', '.join([u.university_name for u in result.universities if u.university_name])
        print(f"   Universities   : {uni_names} ({len(result.universities)})")
    else:
        print(f"   Universities   : 0")
    print(f"   Sources        : {len(result.sources)}")
    print(f"   Methods        : {result.admission_methods}")

    for i, uni in enumerate(result.universities[:3], 1):
        print(f"\n   [{i}] {uni.university_name} — {uni.major}")
        if uni.benchmark_score:
            print(f"       Điểm chuẩn : {uni.benchmark_score}")
        if uni.admission_method:
            print(f"       Phương thức: {uni.admission_method}")
        if uni.deadline:
            print(f"       Deadline   : {uni.deadline}")

    if result.important_deadlines:
        print(f"\n   Deadlines:")
        for k, v in list(result.important_deadlines.items())[:3]:
            print(f"   - {k}: {v}")

    print(f"\n   Summary: {result.raw_summary[:300]}...")
    print(f"{'─'*50}")


if __name__ == "__main__":
    print("Researcher Agent Test Suite")
    print(f"   FOUNDRY_PROJECT_ENDPOINT : {os.environ.get('FOUNDRY_PROJECT_ENDPOINT', 'NOT SET')}")
    print(f"   FOUNDRY_MODEL            : {os.environ.get('FOUNDRY_MODEL', 'NOT SET')}")
    print(f"   AZURE_SEARCH_ENDPOINT    : {os.environ.get('AZURE_SEARCH_ENDPOINT', 'NOT SET')}")
    print(f"   AZURE_SEARCH_INDEX_NAME  : {os.environ.get('AZURE_SEARCH_INDEX_NAME', 'NOT SET')}")

    test_num = sys.argv[1] if len(sys.argv) > 1 else "2"
    tests = {"1": test_search_index, "2": test_query_with_profile, "3": test_stream}
    asyncio.run(tests.get(test_num, test_query_with_profile)())