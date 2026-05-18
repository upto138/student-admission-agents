"""
Advisor Agent — Explains admission plan and gives personalized guidance.

Position in pipeline:
    ResearcherAgent → PlannerAgent → [AdvisorAgent] → ApplicationAgent

Input (from AgentState):
    state.research_result   — ResearcherAgent output (structured admission info)
    state.plan_result       — PlannerAgent output (checklist + timeline + risks)
    state.student_profile   — student profile

Output (written to AgentState):
    state.advisor_result    — AdvisorResult: narrative + advice + risks + actions
    state.advisor_response  — backward compat (= advisor_result.narrative)

How it works (Pattern A — same as Researcher/Planner):
    1. _get_agent()            → Lazy-init FoundryChatClient + client.as_agent()
    2. _build_user_message()   → Summarize state context (Lesson 12: no raw dump)
    3. agent.run(message)      → Framework handles LLM call
    4. response.text           → raw text from LLM
    5. _parse_advisor_result() → Extract JSON → AdvisorResult
    6. Write to state and return

Tools (MVP):
    None — pure LLM reasoning. Advisor receives all context via _build_user_message().
    Phase 2: optional RAG tool for additional knowledge base queries.
"""
from __future__ import annotations

import json
import logging
import os
import re
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential, AzureCliCredential

from app.agents.base import BaseAgent
from app.schemas.agent_state import AgentState, AdvisorResult, RiskItem, ActionStep

logger = logging.getLogger(__name__)


# =============================================================================
# Helper functions
# =============================================================================

def _get_credential():
    """
    Get Azure credential in order of priority — same as Researcher/Planner.
    1. DefaultAzureCredential (support Managed Identity, CLI, env)
    2. AzureCliCredential (fallback for local dev)
    """
    try:
        return DefaultAzureCredential()
    except Exception:
        return AzureCliCredential()


def _load_prompt() -> str:
    """Read system prompt from file prompts/advisor.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "advisor.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    logger.warning(f"Prompt file not found: {prompt_path}. Using fallback.")
    return "You are a university admissions advisor for Vietnamese students."


def _build_user_message(state: AgentState) -> str:
    """
    Build summarized context message from AgentState.

    Strategy (Lesson 12 — Context Engineering):
      - Summarize input selectively, do NOT dump raw JSON.
      - Include: student profile, research summary, plan summary, original question.
      - Avoid context distraction by keeping sections focused.
    """
    sections = []
    profile = state.student_profile

    # 1. Student Profile (always present)
    sections.append(
        f"## STUDENT PROFILE\n"
        f"- Full name: {profile.name or '[NOT PROVIDED]'}\n"
        f"- GPA: {profile.gpa if profile.gpa is not None else '[NOT PROVIDED]'}\n"
        f"- Target scores: {json.dumps(profile.target_scores, ensure_ascii=False) if profile.target_scores else '[NOT PROVIDED]'}\n"
        f"- Preferred majors: {', '.join(profile.preferred_majors) if profile.preferred_majors else '[NOT PROVIDED]'}\n"
        f"- Preferred universities: {', '.join(profile.preferred_universities) if profile.preferred_universities else '[NOT PROVIDED]'}\n"
        f"- Extracurricular activities: {', '.join(profile.extra_activities) if profile.extra_activities else '[NOT PROVIDED]'}\n"
        f"- Notes: {profile.notes or 'None'}"
    )

    # 2. Research Summary (summarized, not raw dump)
    if state.research_result:
        r = state.research_result
        uni_lines = []
        for u in r.universities:
            parts = [f"{u.university_name} / {u.major}"]
            if u.benchmark_score is not None:
                parts.append(f"benchmark score: {u.benchmark_score}")
            if u.admission_method:
                parts.append(f"admission method: {u.admission_method}")
            if u.deadline:
                parts.append(f"deadline: {u.deadline}")
            uni_lines.append(f"  - {' | '.join(parts)}")

        uni_summary = "\n".join(uni_lines) if uni_lines else "  (None found)"

        deadlines_str = ""
        if r.important_deadlines:
            deadlines_str = "\n".join(
                f"  - {k}: {v}" for k, v in r.important_deadlines.items()
            )
        else:
            deadlines_str = "  (No information available)"

        sections.append(
            f"## RESEARCH RESULTS\n"
            f"Summary: {r.raw_summary or '(No summary available)'}\n"
            f"Universities/Majors found:\n{uni_summary}\n"
            f"Admission methods: {', '.join(r.admission_methods) if r.admission_methods else '(No information available)'}\n"
            f"Key deadlines:\n{deadlines_str}"
        )

    # 3. Plan Summary
    if state.plan_result:
        p = state.plan_result

        checklist_lines = []
        for item in p.checklist:
            req_label = "required" if item.required else "optional"
            checklist_lines.append(f"  - [{item.status}] {item.title} ({req_label})")
        checklist_summary = "\n".join(checklist_lines) if checklist_lines else "  (None)"

        timeline_lines = []
        for t in p.timeline:
            date_str = t.date or "TBD"
            timeline_lines.append(f"  - [{t.priority}] {date_str}: {t.task}")
        timeline_summary = "\n".join(timeline_lines) if timeline_lines else "  (None)"

        risks_str = "; ".join(p.risks) if p.risks else "None"
        missing_str = "; ".join(p.missing_questions) if p.missing_questions else "None"

        sections.append(
            f"## ADMISSION PLAN\n"
            f"Document checklist:\n{checklist_summary}\n"
            f"Timeline:\n{timeline_summary}\n"
            f"Risks: {risks_str}\n"
            f"Unanswered questions: {missing_str}"
        )

    # 4. Original user question
    if state.user_message:
        sections.append(
            f"## STUDENT'S ORIGINAL QUESTION\n"
            f"{state.user_message}"
        )

    return "\n\n".join(sections)


def _parse_advisor_result(raw_text: str) -> AdvisorResult:
    """
    Parse JSON from agent response into AdvisorResult.

    Strategy (same multi-layer approach as Researcher/Planner):
      1. Try pure JSON parse of entire text
      2. Try fenced code block (```json ... ```)
      3. Raw scan for JSON object containing 'narrative' key
      4. Fallback: use raw text as narrative
    """
    text = (raw_text or "").strip()
    if not text:
        return AdvisorResult(narrative="", advised_at=datetime.now(timezone.utc))

    # ── Helpers ────────────────────────────────────────────────────────────

    def _try_parse(candidate: str) -> Optional[dict]:
        """Strict JSON → ast.literal_eval fallback."""
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            obj = ast.literal_eval(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # ── Try 1: whole text is pure JSON ────────────────────────────────────
    data = _try_parse(text)

    # ── Try 2: fenced code block ──────────────────────────────────────────
    if data is None:
        for fence_match in re.finditer(r"```(?:json|JSON)?\s*(.*?)\s*```", text, re.DOTALL):
            data = _try_parse(fence_match.group(1).strip())
            if data:
                break

    # ── Try 3: raw scan for JSON object with 'narrative' key ──────────────
    if data is None:
        decoder = json.JSONDecoder()
        for m in re.finditer(r"\{", text):
            try:
                obj, _ = decoder.raw_decode(text, m.start())
                if isinstance(obj, dict) and "narrative" in obj:
                    data = obj
                    break
            except json.JSONDecodeError:
                continue

    # ── Build AdvisorResult from parsed dict ──────────────────────────────
    if data is not None:
        try:
            risk_list = []
            for r in data.get("risk_analysis", []):
                if isinstance(r, dict):
                    risk_list.append(RiskItem(
                        description=str(r.get("description", "")),
                        severity=str(r.get("severity", "medium")),
                        mitigation=str(r.get("mitigation", "")),
                    ))
                else:
                    risk_list.append(RiskItem(description=str(r)))

            action_list = []
            for a in data.get("action_steps", []):
                if isinstance(a, dict):
                    action_list.append(ActionStep(
                        step=str(a.get("step", "")),
                        priority=str(a.get("priority", "medium")),
                        deadline_hint=a.get("deadline_hint"),
                    ))
                else:
                    action_list.append(ActionStep(step=str(a)))

            return AdvisorResult(
                narrative=str(data.get("narrative", "")),
                personalized_advice=data.get("personalized_advice", []),
                risk_analysis=risk_list,
                action_steps=action_list,
                confidence_level=str(data.get("confidence_level", "medium")),
                caveats=data.get("caveats", []),
                advised_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.warning(f"AdvisorAgent: error building AdvisorResult from dict: {e}")

    # ── Fallback: use raw text as narrative ────────────────────────────────
    logger.warning(
        "AdvisorAgent: could not parse JSON from response. "
        "Falling back to raw narrative. text[:200]=%r",
        text[:200].replace("\n", "\\n"),
    )
    return AdvisorResult(
        narrative=text,
        confidence_level="low",
        caveats=["Output could not be parsed as structured JSON — using raw text as narrative"],
        advised_at=datetime.now(timezone.utc),
    )


# =============================================================================
# AdvisorAgent class
# =============================================================================

class AdvisorAgent(BaseAgent):
    """
    Advisor Agent using Microsoft Agent Framework (FoundryChatClient).

    How it works (Pattern A — same as Researcher/Planner):
    1. Initialize FoundryChatClient with FOUNDRY_PROJECT_ENDPOINT + credential
    2. Create Agent via client.as_agent(tools=[], instructions=...)
       - MVP: No tools — pure LLM reasoning
       - Phase 2: optional RAG tool for additional KB queries
    3. Call agent.run(message) — framework handles LLM call
    4. Parse response.text → AdvisorResult
    5. Update AgentState and return
    """

    name = "advisor"
    description = "Explains admission plan and gives personalized guidance"

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._foundry_agent = None  # Lazy-init

    def _get_agent(self):
        """
        Lazy-init Foundry Agent — Pattern A (client.as_agent()).

        Uses FoundryChatClient (NOT FoundryAgent) to ensure local Python
        tools work correctly. See Lesson 02, 13 for details.
        """
        if self._foundry_agent is None:
            project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
            model = os.environ.get("FOUNDRY_MODEL", "gpt-4o-admission")

            if not project_endpoint:
                raise EnvironmentError(
                    "Missing variable FOUNDRY_PROJECT_ENDPOINT in .env.\n"
                )

            client = FoundryChatClient(
                project_endpoint=project_endpoint,
                model=model,
                credential=_get_credential(),
            )

            # Pattern A: client.as_agent() — verified with Researcher/Planner
            self._foundry_agent = client.as_agent(
                name="AdvisorAgent",
                instructions=_load_prompt(),
                tools=[],   # MVP: No tools — pure LLM reasoning
            )

            self.log(
                f"FoundryChatClient initialized. Endpoint: {project_endpoint} | Model: {model}",
                "info",
            )

        return self._foundry_agent

    async def run(self, state: AgentState) -> AgentState:
        """
        Run the Advisor Agent and update AgentState with result.

        Guard: if plan_result is None → log error and return early.

        Args:
            state: AgentState with research_result, plan_result, student_profile.

        Returns:
            AgentState updated with advisor_result and advisor_response.
        """
        state.current_agent = self.name
        self.log("Starting advisor task...", "info")

        # ── Guard: Planner must run first ─────────────────────────────
        if state.plan_result is None:
            msg = (
                "AdvisorAgent: no plan_result in state. "
                "PlannerAgent has not run or has failed."
            )
            logger.error(msg)
            state.add_error(msg)
            return state

        try:
            agent = self._get_agent()
            user_message = _build_user_message(state)

            self.log(f"Sending message to agent ({len(user_message)} chars)...", "info")

            # agent.run() — framework handles LLM call (no tool calls for MVP)
            response = await agent.run(user_message)
            raw_text = response.text  # .text — verified with Researcher/Planner

            self.log(f"Raw response: {len(raw_text)} chars", "info")

            # ── Parse result ──────────────────────────────────────────
            advisor_result = _parse_advisor_result(raw_text)
            state.advisor_result = advisor_result
            state.advisor_response = advisor_result.narrative  # backward compat
            state.mark_agent_done(self.name)

            self.log(
                f"Advisor complete. Confidence: {advisor_result.confidence_level} | "
                f"Risks: {len(advisor_result.risk_analysis)} | "
                f"Actions: {len(advisor_result.action_steps)} | "
                f"Caveats: {len(advisor_result.caveats)}",
                "info",
            )

        except Exception as e:
            logger.exception(f"AdvisorAgent failed: {e}")
            state.add_error(f"AdvisorAgent error: {str(e)}")
            state.advisor_response = f"Advisor failed: {str(e)}"

        return state

    async def run_stream(self, state: AgentState):
        """
        Streaming version of run() - yield individual chunks of text.
        Used for WebSocket endpoint or real-time UI.

        Usage:
            async for chunk in advisor.run_stream(state):
                print(chunk, end="", flush=True)
        """
        if state.plan_result is None:
            yield "⚠️ Cannot advise: no admission plan available from Planner."
            return

        try:
            agent = self._get_agent()
            user_message = _build_user_message(state)

            async for chunk in agent.run(user_message, stream=True):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.exception(f"AdvisorAgent stream failed: {e}")
            yield f"\n⚠️ Error: {str(e)}"
