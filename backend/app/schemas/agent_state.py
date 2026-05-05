"""
Shared state object are passed to all agents in the workflow
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    """Basic student infomation"""
    name: str = ""
    gpa: Optional[float] = None
    target_scores: Dict[str, float] = Field(default_factory=dict)
    preferred_majors: List[str] = Field(default_factory=list)
    preferred_universities: List[str] = Field(default_factory=list)
    extra_activities: List[str] = Field(default_factory=list)
    notes: str = ""


class UniversityInfo(BaseModel):
    """Information about a university/program"""
    university_name: str
    major: str
    benchmark_score: Optional[float] = None
    admission_method: str = ""
    required_documents: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    tuition_fee: Optional[str] = None
    website: Optional[str] = None
    source_url: Optional[str] = None
    notes: str = ""


class ResearchResult(BaseModel):
    """Output of the Researcher Agent"""
    query: str = ""
    universities: List[UniversityInfo] = Field(default_factory=list)
    general_requirements: List[str] = Field(default_factory=list)
    important_deadlines: Dict[str, str] = Field(default_factory=dict)
    admission_methods: List[str] = Field(default_factory=list)
    raw_summary: str = ""
    sources: List[str] = Field(default_factory=list)
    researched_at: Optional[datetime] = None


class AdmissionStep(BaseModel):
    """One step in the selection process"""
    step_number: int
    title: str
    description: str
    deadline: Optional[str] = None
    required_documents: List[str] = Field(default_factory=list)
    is_completed: bool = False


class AdmissionPlan(BaseModel):
    """Output of the Planner Agent."""
    student_name: str = ""
    recommended_universities: List[UniversityInfo] = Field(default_factory=list)
    steps: List[AdmissionStep] = Field(default_factory=list)
    document_checklist: List[str] = Field(default_factory=list)
    risk_level: str = "medium"  # low / medium / high
    notes: str = ""


class AgentState(BaseModel):
    """
    The shared state is passed through all agenst in the pipeline
    Each agent reads the state, executes the logic and then updates state before returning
    """
    # Session info
    session_id: str = ""
    user_message: str = ""

    # Input data
    student_profile: StudentProfile = Field(default_factory=StudentProfile)
    input_urls: List[str] = Field(default_factory=list)

    # Agent outputs (filled in gradually along the pipeline)
    research_result: Optional[ResearchResult] = None
    admission_plan: Optional[AdmissionPlan] = None
    advisor_response: str = ""
    application_status: str = ""  # draft / pending_confirmation / submitted / failed

    # HITL (Human-in-the-Loop)
    requires_confirmation: bool = False
    confirmation_payload: Optional[Dict[str, Any]] = None

    # Metadata
    errors: List[str] = Field(default_factory=list)
    current_agent: str = ""
    completed_agents: List[str] = Field(default_factory=list)

    def mark_agent_done(self, agent_name: str) -> None:
        """Mark an agent as completed"""
        if agent_name not in self.completed_agents:
            self.completed_agents.append(agent_name)

    def add_error(self, error: str) -> None:
        """Record errors during processing"""
        self.errors.append(error)