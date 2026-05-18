"""
Unit Tests — AdvisorAgent (Agent 3)

These tests check the advisor's internal helper functions:
- _build_user_message() : handles edge cases including empty/None profiles, empty/None research and plan results.
- _parse_advisor_result() : handles various formats of LLM outputs (pure JSON, fenced JSON, raw text).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import pytest

from app.agents.advisor import _build_user_message, _parse_advisor_result
from app.schemas.agent_state import (
    AgentState,
    StudentProfile,
    ResearchResult,
    PlanResult,
    UniversityInfo,
    ChecklistItem,
    TimelineTask,
    AdvisorResult,
)


class TestAdvisorBuildUserMessage:
    """Tests for _build_user_message edge cases."""

    def test_edge_case_all_fields_none(self):
        """
        Edge case 1: StudentProfile contains empty or None fields.
        Must not crash and should render default placeholder '[NOT PROVIDED]'.
        """
        # Pydantic initializes these with default factories, but let's explicitly pass empty values or None where allowed.
        profile = StudentProfile(
            name="",
            gpa=None,
            target_scores={},
            preferred_majors=[],
            preferred_universities=[],
            extra_activities=[],
            notes=""
        )
        
        state = AgentState(
            student_profile=profile,
            user_message=""
        )
        
        message = _build_user_message(state)
        
        # Verify placeholders exist
        assert "## STUDENT PROFILE" in message
        assert "Full name: [NOT PROVIDED]" in message
        assert "GPA: [NOT PROVIDED]" in message
        assert "Target scores: [NOT PROVIDED]" in message
        assert "Preferred majors: [NOT PROVIDED]" in message
        assert "Preferred universities: [NOT PROVIDED]" in message
        assert "Extracurricular activities: [NOT PROVIDED]" in message
        assert "Notes: None" in message

    def test_edge_case_research_result_empty(self):
        """
        Edge case 2: research_result exists but is empty (default values/no fields populated).
        Must not crash and should render reasonable empty indicators.
        """
        profile = StudentProfile(name="Nguyen Van A")
        research_result = ResearchResult(
            query="",
            universities=[],
            general_requirements=[],
            important_deadlines={},
            admission_methods=[],
            raw_summary="",
            sources=[]
        )
        
        state = AgentState(
            student_profile=profile,
            research_result=research_result
        )
        
        message = _build_user_message(state)
        
        assert "## RESEARCH RESULTS" in message
        assert "Summary: (No summary available)" in message
        assert "Universities/Majors found:\n  (None found)" in message
        assert "Admission methods: (No information available)" in message
        assert "Key deadlines:\n  (No information available)" in message

    def test_edge_case_plan_result_empty_but_exists(self):
        """
        Edge case 3: plan_result exists (not None) but contains empty lists and default values.
        Must not crash and should render correct defaults.
        """
        profile = StudentProfile(name="Nguyen Van A")
        plan_result = PlanResult(
            missing_questions=[],
            checklist=[],
            timeline=[],
            risks=[],
            needs_human_confirmation=False
        )
        
        state = AgentState(
            student_profile=profile,
            plan_result=plan_result
        )
        
        message = _build_user_message(state)
        
        assert "## ADMISSION PLAN" in message
        assert "Document checklist:\n  (None)" in message
        assert "Timeline:\n  (None)" in message
        assert "Risks: None" in message
        assert "Unanswered questions: None" in message


class TestAdvisorParseResult:
    """Tests for parsing of Advisor responses."""

    def test_parse_pure_json(self):
        """Pure JSON payload must parse successfully."""
        data = {
            "narrative": "Tư vấn tổng quan cho học sinh.",
            "personalized_advice": ["Hãy tập trung học Toán"],
            "risk_analysis": [
                {"description": "Thiếu chứng chỉ tiếng Anh", "severity": "high", "mitigation": "Học thi IELTS"}
            ],
            "action_steps": [
                {"step": "Đăng ký thi IELTS", "priority": "high", "deadline_hint": "2026-06-01"}
            ],
            "confidence_level": "high",
            "caveats": []
        }
        raw = json.dumps(data, ensure_ascii=False)
        result = _parse_advisor_result(raw)
        
        assert isinstance(result, AdvisorResult)
        assert result.narrative == "Tư vấn tổng quan cho học sinh."
        assert len(result.risk_analysis) == 1
        assert result.risk_analysis[0].description == "Thiếu chứng chỉ tiếng Anh"
        assert result.risk_analysis[0].severity == "high"
        assert len(result.action_steps) == 1
        assert result.action_steps[0].step == "Đăng ký thi IELTS"
        assert result.confidence_level == "high"

    def test_parse_fenced_json(self):
        """Fenced JSON block must parse successfully."""
        raw = """
        Đây là phân tích của tôi:
        ```json
        {
            "narrative": "Tư vấn",
            "confidence_level": "medium"
        }
        ```
        """
        result = _parse_advisor_result(raw)
        assert isinstance(result, AdvisorResult)
        assert result.narrative == "Tư vấn"
        assert result.confidence_level == "medium"

    def test_parse_fallback_raw_text(self):
        """If response is not JSON, it should fall back to using the raw text as narrative."""
        raw = "Đây không phải là JSON. Đây là lời khuyên viết thường."
        result = _parse_advisor_result(raw)
        
        assert isinstance(result, AdvisorResult)
        assert result.narrative == raw
        assert result.confidence_level == "low"
        assert "using raw text as narrative" in result.caveats[0]
