"""
Shared state object are passed to all agents in the workflow.

Schemas:
  StudentProfile    — thông tin học sinh (input)
  UniversityInfo    — thông tin một trường/ngành
  ResearchResult    — output của ResearcherAgent
  ChecklistItem     — một mục trong danh sách hồ sơ (output Planner)
  TimelineTask      — một mốc trong timeline chuẩn bị (output Planner)
  PlanResult        — toàn bộ output của PlannerAgent
  RiskItem          — một rủi ro cụ thể (output Advisor)
  ActionStep        — một bước hành động cụ thể (output Advisor)
  AdvisorResult     — toàn bộ output của AdvisorAgent
  AdmissionStep     — một bước trong quy trình tuyển sinh
  AdmissionPlan     — kế hoạch tổng thể (legacy, đơn giản hơn PlanResult)
  AgentState        — state dùng chung cho toàn bộ pipeline
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


# =============================================================================
# Planner Agent schemas
# =============================================================================

class ChecklistItem(BaseModel):
    """
    Một mục trong danh sách hồ sơ cần chuẩn bị.

    source phải là một trong:
      admission_news   — lấy từ bài báo tuyển sinh
      rag              — lấy từ knowledge base (ChromaDB / Azure Search)
      student_profile  — suy ra từ hồ sơ học sinh
      inferred         — suy luận, không có nguồn rõ ràng → status nên là need_review
    """
    title: str
    status: str = "pending"      # ready | pending | need_review
    required: bool = True
    reason: str = ""
    source: str = ""             # admission_news | rag | student_profile | inferred


class TimelineTask(BaseModel):
    """
    Một mốc trong timeline chuẩn bị hồ sơ.
    Nếu deadline từ bài báo là null → date = None, mô tả dùng 'as soon as possible'.
    """
    date: Optional[str] = None   # ISO date hoặc None
    task: str
    priority: str = "medium"     # high | medium | low
    reason: str = ""


class PlanResult(BaseModel):
    """
    Output đầy đủ của PlannerAgent.
    Khớp chính xác với schema PlanResult trong project-docs.md.

    Lưu ý:
      - Agent 3 bị chặn bởi Orchestrator nếu missing_questions không rỗng.
      - Mọi ChecklistItem phải có source; nếu source = inferred thì status = need_review.
      - Nếu deadline từ Researcher là null, các TimelineTask không được chứa ngày cụ thể.
    """
    missing_questions: List[str] = Field(default_factory=list)
    checklist: List[ChecklistItem] = Field(default_factory=list)
    timeline: List[TimelineTask] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    needs_human_confirmation: bool = False


# =============================================================================
# Advisor Agent schemas
# =============================================================================

class RiskItem(BaseModel):
    """Một rủi ro cụ thể được phân tích bởi AdvisorAgent."""
    description: str                              # Mô tả rủi ro
    severity: str = "medium"                      # high | medium | low
    mitigation: str = ""                          # Cách giảm thiểu


class ActionStep(BaseModel):
    """Một bước hành động cụ thể do AdvisorAgent đề xuất."""
    step: str                                     # Mô tả hành động
    priority: str = "medium"                      # high | medium | low
    deadline_hint: Optional[str] = None           # Gợi ý thời hạn (YYYY-MM-DD hoặc None)


class AdvisorResult(BaseModel):
    """
    Output đầy đủ của AdvisorAgent.

    Hybrid approach: structured schema + free-text narrative.
    - narrative: phần tư vấn dạng văn bản tự nhiên (phần chính, hiển thị cho user)
    - personalized_advice: lời khuyên dạng bullet points
    - risk_analysis: phân tích rủi ro có cấu trúc
    - action_steps: các bước hành động cụ thể, ưu tiên theo deadline
    - confidence_level: đánh giá độ đầy đủ của dữ liệu đầu vào
    - caveats: giới hạn, mâu thuẫn dữ liệu, hoặc thông tin thiếu
    """
    narrative: str = ""                           # Tư vấn dạng free-text (phần chính)
    personalized_advice: List[str] = Field(default_factory=list)
    risk_analysis: List[RiskItem] = Field(default_factory=list)
    action_steps: List[ActionStep] = Field(default_factory=list)
    confidence_level: str = "medium"              # high | medium | low
    caveats: List[str] = Field(default_factory=list)
    advised_at: Optional[datetime] = None         # Timestamp

### ================= End advisor schemas =================

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
    plan_result: Optional[PlanResult] = None
    admission_plan: Optional[AdmissionPlan] = None  # legacy — dùng plan_result thay thế
    advisor_result: Optional[AdvisorResult] = None   # Structured output from AdvisorAgent
    advisor_response: str = ""                        # Backward compat — = advisor_result.narrative
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