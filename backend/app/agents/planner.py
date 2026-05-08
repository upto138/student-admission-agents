"""
Planner Agent — Analyzes admission data and generates a structured preparation plan.

Position in pipeline:
    ResearcherAgent → [PlannerAgent] → AdvisorAgent → ApplicationAgent

Input (from AgentState):
    state.research_result   — ResearcherAgent output (structured admission info)
    state.student_profile   — student profile (may have missing fields)

Output (written to AgentState):
    state.plan_result       — PlanResult: checklist + timeline + missing_questions + risks

How it works (mirrors ResearcherAgent pattern):
    1. _get_agent()          → Lazy-init FoundryChatClient agent
    2. _build_user_message() → Format research_result + student_profile → LLM message
    3. agent.run(message)    → Framework handles tool calls automatically
    4. response.text         → raw text from LLM
    5. _parse_plan_result()  → Extract JSON → PlanResult
    6. Write to state and return

Tools (via PlannerToolset):
    query_knowledge_base — Queries ChromaDB / Azure AI Search for additional
    admission regulations, scholarship conditions, or school-specific policies
    that the Researcher may not have covered for the specific university/major.

Extensibility (Strategy pattern):
    See PlannerToolset below. To add more tools in the future,
    only modify PlannerToolset.tools — no changes to agent logic needed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import ast
from pathlib import Path
from typing import Optional

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from app.tools.rag_tool import ALL_RAG_TOOLS
from azure.identity import DefaultAzureCredential, AzureCliCredential

from app.agents.base import BaseAgent
from app.schemas.agent_state import (
    AgentState,
    ChecklistItem,
    PlanResult,
    TimelineTask,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PlannerToolset — Strategy object (Extensibility point)
# =============================================================================

class PlannerToolset:
    """
    Strategy object: controls the list of tools available to the Planner Agent.

    --- CURRENT (v2) ---
    Uses query_knowledge_base to supplement the Researcher's findings:
      - Researcher already scraped URLs and did an initial RAG query.
      - Planner can do a focused, planning-specific RAG query to verify:
          * Document requirements for a specific university/major
          * Scholarship eligibility conditions
          * Ministry of Education (MOET) regulations
          * Edge cases the Researcher may not have covered

    --- FUTURE: uncomment when the following modules are ready ---
    # from app.tools.regulation_tool import ALL_REGULATION_TOOLS  # not built yet
    # → Specialized tool for MOET regulations (when KB is populated with official docs)
    """
    # query_knowledge_base: tra cứu thêm quy định, học bổng, điều kiện đặc biệt
    tools: list = ALL_RAG_TOOLS


ALL_PLANNER_TOOLS = PlannerToolset.tools


# =============================================================================
# Helper functions
# =============================================================================

def _load_prompt() -> str:
    """Đọc system prompt từ file prompts/planner.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "planner.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    logger.warning(f"Prompt file not found: {prompt_path}. Using fallback prompt.")
    return (
        "Bạn là Planner Agent trong hệ thống tư vấn tuyển sinh đại học Việt Nam. "
        "Nhiệm vụ của bạn là phân tích thông tin tuyển sinh và hồ sơ học sinh, "
        "sau đó sinh ra danh sách hồ sơ cần chuẩn bị và timeline. "
        "Luôn trả về JSON theo schema PlanResult."
    )


def _build_user_message(state: AgentState) -> str:
    """
    Tạo user message từ AgentState để gửi cho LLM.

    Bao gồm:
      - Thông tin tuyển sinh (từ research_result)
      - Hồ sơ học sinh (student_profile)
      - Yêu cầu output JSON theo schema PlanResult
    """
    parts = []
    result = state.research_result
    profile = state.student_profile

    # ── Thông tin tuyển sinh từ Researcher ──────────────────────────────────
    if result:
        uni_summaries = []
        for u in result.universities:
            summary_parts = []
            if u.university_name:
                summary_parts.append(f"Trường: {u.university_name}")
            if u.major:
                summary_parts.append(f"Ngành: {u.major}")
            if u.admission_method:
                summary_parts.append(f"Phương thức: {u.admission_method}")
            if u.deadline:
                summary_parts.append(f"Hạn nộp: {u.deadline}")
            if u.required_documents:
                docs = ", ".join(u.required_documents)
                summary_parts.append(f"Giấy tờ cần: {docs}")
            if u.benchmark_score is not None:
                summary_parts.append(f"Điểm chuẩn: {u.benchmark_score}")
            if u.notes:
                summary_parts.append(f"Ghi chú: {u.notes}")
            if summary_parts:
                uni_summaries.append("\n  ".join(summary_parts))

        if uni_summaries:
            parts.append(
                "**Thông tin tuyển sinh (từ Researcher Agent):**\n"
                + "\n---\n".join(uni_summaries)
            )

        if result.general_requirements:
            parts.append(
                "**Yêu cầu chung:**\n"
                + "\n".join(f"- {r}" for r in result.general_requirements)
            )

        if result.important_deadlines:
            deadlines = "\n".join(
                f"- {k}: {v}" for k, v in result.important_deadlines.items()
            )
            parts.append(f"**Các mốc thời gian quan trọng:**\n{deadlines}")

        if result.admission_methods:
            parts.append(
                "**Phương thức xét tuyển:**\n"
                + "\n".join(f"- {m}" for m in result.admission_methods)
            )

        if result.sources:
            parts.append(
                "**Nguồn thông tin:**\n"
                + "\n".join(f"- {s}" for s in result.sources)
            )

    # ── Hồ sơ học sinh ──────────────────────────────────────────────────────
    profile_lines = []
    if profile.name:
        profile_lines.append(f"- Họ tên: {profile.name}")
    else:
        profile_lines.append("- Họ tên: [CHƯA CÓ]")

    profile_lines.append(f"- GPA: {profile.gpa if profile.gpa is not None else '[CHƯA CÓ]'}")

    if profile.preferred_majors:
        profile_lines.append(f"- Ngành mong muốn: {', '.join(profile.preferred_majors)}")
    else:
        profile_lines.append("- Ngành mong muốn: [CHƯA CÓ]")

    if profile.preferred_universities:
        profile_lines.append(f"- Trường mong muốn: {', '.join(profile.preferred_universities)}")

    if profile.target_scores:
        scores = json.dumps(profile.target_scores, ensure_ascii=False)
        profile_lines.append(f"- Điểm mục tiêu: {scores}")

    if profile.extra_activities:
        profile_lines.append(f"- Hoạt động ngoại khóa: {', '.join(profile.extra_activities)}")

    if profile.notes:
        profile_lines.append(f"- Ghi chú: {profile.notes}")

    parts.append("**Hồ sơ học sinh:**\n" + "\n".join(profile_lines))

    # ── Yêu cầu output ──────────────────────────────────────────────────────
    parts.append(
        "**Yêu cầu output:** Trả về JSON hợp lệ theo schema PlanResult:\n"
        "{\n"
        '  "missing_questions": ["câu hỏi 1 nếu thiếu thông tin học sinh"],\n'
        '  "checklist": [\n'
        '    {\n'
        '      "title": "Tên giấy tờ",\n'
        '      "status": "pending",\n'
        '      "required": true,\n'
        '      "reason": "Lý do cần giấy tờ này",\n'
        '      "source": "admission_news"\n'
        '    }\n'
        '  ],\n'
        '  "timeline": [\n'
        '    {\n'
        '      "date": "2026-05-15",\n'
        '      "task": "Mô tả việc cần làm",\n'
        '      "priority": "high",\n'
        '      "reason": "Lý do ưu tiên cao"\n'
        '    }\n'
        '  ],\n'
        '  "risks": ["Rủi ro 1 nếu có"],\n'
        '  "needs_human_confirmation": true\n'
        "}"
    )

    return "\n\n".join(parts)


def _parse_plan_result(raw_text: str) -> PlanResult:
    """
    Parse JSON từ response của LLM thành PlanResult.

    Chiến lược (giống _parse_research_result):
      1. Thử parse toàn bộ text là JSON thuần
      2. Thử tìm fenced code block (```json ... ```)
      3. Quét tìm JSON object trong text
      4. Fallback: trả về PlanResult rỗng, không crash
    """
    text = (raw_text or "").strip()
    if not text:
        return PlanResult()

    EXPECTED_KEYS = frozenset({
        "missing_questions", "checklist", "timeline", "risks", "needs_human_confirmation",
    })

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _coerce(obj) -> Optional[dict]:
        if isinstance(obj, dict):
            return obj
        return None

    def _try_parse(candidate: str) -> Optional[dict]:
        """JSON thuần → ast.literal_eval fallback."""
        if not candidate:
            return None
        try:
            return _coerce(json.loads(candidate))
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return _coerce(ast.literal_eval(candidate))
        except Exception:
            return None

    def _best_json_in_text(candidate: str) -> Optional[dict]:
        """Quét tìm JSON object tốt nhất trong text."""
        if not candidate:
            return None
        decoder = json.JSONDecoder()
        best: Optional[dict] = None
        best_score = -1
        for m in re.finditer(r"[{]", candidate):
            try:
                obj, _ = decoder.raw_decode(candidate, m.start())
            except json.JSONDecodeError:
                continue
            as_dict = _coerce(obj)
            if as_dict is None:
                continue
            score = len(EXPECTED_KEYS & as_dict.keys())
            if score > best_score:
                best, best_score = as_dict, score
                if score >= 2 and "checklist" in as_dict:
                    return best
        return best

    # ── Thử 1: toàn bộ text là JSON thuần ──────────────────────────────────
    decoded = _try_parse(text)
    if decoded is not None:
        return _dict_to_plan_result(decoded)

    # ── Thử 2: fenced code block ─────────────────────────────────────────────
    best_from_fence: Optional[dict] = None
    best_fence_score = -1
    for fence_match in re.finditer(r"```(?:json|JSON)?\s*(.*?)\s*```", text, re.DOTALL):
        fenced = fence_match.group(1).strip()
        candidate = _try_parse(fenced) or _best_json_in_text(fenced)
        if candidate is None:
            continue
        score = len(EXPECTED_KEYS & candidate.keys())
        if score > best_fence_score:
            best_from_fence, best_fence_score = candidate, score
    if best_from_fence is not None:
        return _dict_to_plan_result(best_from_fence)

    # ── Thử 3: quét raw text ────────────────────────────────────────────────
    decoded = _best_json_in_text(text)
    if decoded is not None:
        return _dict_to_plan_result(decoded)

    # ── Fallback: trả về PlanResult rỗng ───────────────────────────────────
    logger.warning(
        "PlannerAgent: không parse được JSON từ response. "
        "Trả về PlanResult rỗng. text[:400]=%r",
        text[:400].replace("\n", "\\n"),
    )
    return PlanResult()


def _dict_to_plan_result(data: dict) -> PlanResult:
    """Convert dict → PlanResult, xử lý dữ liệu thiếu/sai định dạng."""

    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        return [value]

    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    # ── Parse checklist ──────────────────────────────────────────────────────
    checklist: list[ChecklistItem] = []
    for item in _as_list(data.get("checklist")):
        if not isinstance(item, dict):
            continue
        try:
            checklist.append(ChecklistItem(
                title=str(item.get("title", "")).strip(),
                status=str(item.get("status", "pending")).strip(),
                required=_as_bool(item.get("required", True)),
                reason=str(item.get("reason", "")).strip(),
                source=str(item.get("source", "")).strip(),
            ))
        except Exception as e:
            logger.warning(f"PlannerAgent: bỏ qua ChecklistItem không hợp lệ: {e}. Item={item!r}")

    # ── Parse timeline ───────────────────────────────────────────────────────
    timeline: list[TimelineTask] = []
    for task in _as_list(data.get("timeline")):
        if not isinstance(task, dict):
            continue
        try:
            timeline.append(TimelineTask(
                date=task.get("date") or None,
                task=str(task.get("task", "")).strip(),
                priority=str(task.get("priority", "medium")).strip(),
                reason=str(task.get("reason", "")).strip(),
            ))
        except Exception as e:
            logger.warning(f"PlannerAgent: bỏ qua TimelineTask không hợp lệ: {e}. Task={task!r}")

    return PlanResult(
        missing_questions=_as_list(data.get("missing_questions")),
        checklist=checklist,
        timeline=timeline,
        risks=_as_list(data.get("risks")),
        needs_human_confirmation=_as_bool(data.get("needs_human_confirmation", False)),
    )


def _get_credential():
    """Lấy Azure credential theo thứ tự ưu tiên (giống ResearcherAgent)."""
    try:
        return DefaultAzureCredential()
    except Exception:
        return AzureCliCredential()


# =============================================================================
# PlannerAgent class
# =============================================================================

class PlannerAgent(BaseAgent):
    """
    Planner Agent using Microsoft Agent Framework (FoundryChatClient).

    How it works:
    1. Initialize FoundryChatClient with FOUNDRY_PROJECT_ENDPOINT + credential
    2. Create Agent with .as_agent(tools=self._tools, instructions=...)
       - self._tools = ALL_RAG_TOOLS (query_knowledge_base) by default
       - extra_tools can be injected from outside for future extensions
    3. Call agent.run(message) — framework handles tool calls automatically
    4. Parse response.text → PlanResult
    5. Update AgentState and return

    Extensibility:
        PlannerAgent(verbose=True, extra_tools=[my_regulation_tool])
        → inject additional tools when new modules are ready, without changing internal logic
    """

    name = "planner"
    description = "Phân tích thông tin tuyển sinh và sinh kế hoạch chuẩn bị hồ sơ"

    def __init__(self, verbose: bool = False, extra_tools: list | None = None):
        super().__init__(verbose=verbose)
        # Gộp tools mặc định (PlannerToolset) + tools inject từ bên ngoài
        self._tools = ALL_PLANNER_TOOLS + (extra_tools or [])
        self._foundry_agent: Optional[Agent] = None

    def _get_agent(self) -> Agent:
        """Lazy-init Foundry Agent."""
        if self._foundry_agent is None:
            project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
            model = os.environ.get("FOUNDRY_MODEL", "gpt-4o-admission")

            if not project_endpoint:
                raise EnvironmentError(
                    "Thiếu biến môi trường FOUNDRY_PROJECT_ENDPOINT trong .env.\n"
                    "Xem file .env.example để biết cách cấu hình."
                )

            client = FoundryChatClient(
                project_endpoint=project_endpoint,
                model=model,
                credential=_get_credential(),
            )

            self._foundry_agent = client.as_agent(
                name="PlannerAgent",
                instructions=_load_prompt(),
                tools=self._tools,
            )

            tool_names = (
                [getattr(t, "__name__", getattr(t, "name", repr(t))) for t in self._tools]
                if self._tools else ["(none)"]
            )
            self.log(
                f"FoundryChatClient initialized. Endpoint: {project_endpoint} | "
                f"Model: {model} | Tools: {tool_names}",
                "info",
            )

        return self._foundry_agent

    async def run(self, state: AgentState) -> AgentState:
        """
        Run the Planner Agent and update AgentState with the result.

        Guard: if research_result is None → log error and return early.
        HITL gate: if missing_questions is not empty → set requires_confirmation = True.

        Args:
            state: AgentState with research_result and student_profile.

        Returns:
            AgentState updated with plan_result.
        """
        state.current_agent = self.name
        self.log("Starting planning task...", "info")

        # ── Guard: Researcher must run first ─────────────────────────────
        if state.research_result is None:
            msg = (
                "PlannerAgent: no research_result in state. "
                "ResearcherAgent has not run or has failed."
            )
            logger.error(msg)
            state.add_error(msg)
            return state

        try:
            agent = self._get_agent()
            user_message = _build_user_message(state)

            self.log(f"Sending message to agent ({len(user_message)} chars)...", "info")

            response = await agent.run(user_message)
            raw_text = response.text

            self.log(f"Raw response: {len(raw_text)} chars", "info")

            # ── Parse kết quả ──────────────────────────────────────────────
            plan_result = _parse_plan_result(raw_text)
            state.plan_result = plan_result
            state.mark_agent_done(self.name)

            # ── HITL gate ──────────────────────────────────────────────────
            if plan_result.missing_questions:
                state.requires_confirmation = True
                state.application_status = "pending_info"
                self.log(
                    f"{len(plan_result.missing_questions)} missing questions found. "
                    f"Agent 3 blocked until student provides more information.",
                    "info",
                )
            elif plan_result.needs_human_confirmation:
                state.requires_confirmation = True
                state.application_status = "pending_confirmation"

            self.log(
                f"Planning complete. "
                f"Checklist: {len(plan_result.checklist)} items | "
                f"Timeline: {len(plan_result.timeline)} tasks | "
                f"Missing questions: {len(plan_result.missing_questions)}",
                "info",
            )

        except Exception as e:
            logger.exception(f"PlannerAgent failed: {e}")
            state.add_error(f"PlannerAgent error: {str(e)}")
            state.plan_result = PlanResult()

        return state

    async def run_stream(self, state: AgentState):
        """
        Streaming version — yield individual text chunks from LLM.
        Used for WebSocket endpoint or real-time UI.

        Usage:
            async for chunk in planner.run_stream(state):
                print(chunk, end="", flush=True)
        """
        if state.research_result is None:
            yield "[ERROR] PlannerAgent: research_result not found in state."
            return

        try:
            agent = self._get_agent()
            user_message = _build_user_message(state)

            async for chunk in agent.run(user_message, stream=True):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.exception(f"PlannerAgent stream failed: {e}")
            yield f"[ERROR] {str(e)}"
