"""
Researcher Agent — Collections and Extracts university admissions information.

Using Microsoft Agent Framework (agent-framework >= 1.2.2) with:
- FoundryChatClient  : connects to Azure AI Foundry
- @tool decorator    : registers tools
- agent.run()        : automatically handles the entire tool-call loop
- response.text      : Retrieves the final agent

Flow:
    Input (URL/query + student profile)
    → FoundryChatClient.as_agent(tools=[...])
    → agent.run(message)          ← framework call tools automatically
    → response.text               ← final result
    → _parse_research_result()    ← extract JSON → ResearchResult
    → AgentState.research_result
"""
from __future__ import annotations

import json
import logging
import os
import re
import ast
from datetime import datetime
from pathlib import Path
from typing import Optional
from datetime import timezone

from agent_framework import tool, Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential, AzureCliCredential

from app.agents.base import BaseAgent
from app.ingestion.normalizers.to_knowledge_base import ingest_university_data
from app.schemas.agent_state import AgentState, ResearchResult, UniversityInfo
from app.tools.web_search_tool import ALL_SEARCH_TOOLS
from app.tools.rag_tool import ALL_RAG_TOOLS

logger = logging.getLogger(__name__)

# All the tools Researcher Agent can use
ALL_RESEARCHER_TOOLS = ALL_SEARCH_TOOLS + ALL_RAG_TOOLS


# =============================================================================
# Helper functions
# =============================================================================

def _load_prompt() -> str:
    """Read system prompt from file prompts/researcher.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "researcher.txt" # Todo: enumerate pathname folder
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    
    logger.warning(f"Prompt file not found: {prompt_path}")
    return (
        "You are a researcher agent that collects and extracts "
        "Vietnamese university admission information."
    )


def _build_user_message(state: AgentState) -> str:
    """
    Create user message from AgentState.
    Includes: student's request + profile + URL to be researched
    """
    profile = state.student_profile
    parts = []

    if state.user_message:
        parts.append(f"**Student's request:** {state.user_message}")

    if profile.name:
        parts.append(
            f"**Student information:**\n"
            f"- Name: {profile.name}\n"
            f"- GPA: {profile.gpa or 'Not yet'}\n"
            f"- Target Scores: {json.dumps(profile.target_scores, ensure_ascii=False)}\n"
            f"- Preferred Major: {', '.join(profile.preferred_majors) or 'Not determined'}\n"
            f"- Preferred University: {', '.join(profile.preferred_universities) or 'Not determined'}"
        )

    if state.input_urls:
        parts.append(
            "**URL to Analyze:**\n"
            + "\n".join(f"- {u}" for u in state.input_urls)
        )

    parts.append(
        # Todo: fix hardcode on here
        "\n**Request for output:** Return valid JSON according to schema ResearchResult:\n"
        "{\n"
        '  "universities": [{university_name, major, benchmark_score, admission_method,\n'
        '                    required_documents, deadline, tuition_fee, website, source_url, notes}],\n'
        '  "general_requirements": [...],\n'
        '  "important_deadlines": {"deadline name": "date"},\n'
        '  "admission_methods": [...],\n'
        '  "raw_summary": "brief summary",\n'
        '  "sources": ["url1", "url2"]\n'
        "}"
    )

    return "\n\n".join(parts)


def _parse_research_result(raw_text: str) -> ResearchResult:
    """
    Parse JSON from the agent's response into ResearchResult.
    Priority: pure JSON → fenced block → raw scan → raw_summary fallback.
    """
    text = (raw_text or "").strip()
    if not text:
        return ResearchResult(raw_summary="", sources=[])

    EXPECTED_KEYS = frozenset({
        "universities", "general_requirements", "important_deadlines",
        "admission_methods", "raw_summary", "sources", "query",
    })

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _coerce(obj) -> Optional[dict]:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"universities": obj}
        return None

    def _try_parse(candidate: str) -> Optional[dict]:
        """Strict JSON → ast.literal_eval fallback."""
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
        """
        Scan for all JSON objects/arrays in candidate; return the one whose
        keys overlap most with EXPECTED_KEYS.  Short-circuits when a clearly
        matching object is found.
        """
        if not candidate:
            return None

        decoder = json.JSONDecoder()
        best: Optional[dict] = None
        best_score = -1

        for m in re.finditer(r"[{\[]", candidate):
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
                if score >= 2 and ("universities" in as_dict or "sources" in as_dict):
                    return best  # good enough — stop early

        return best

    # ── Try 1: whole text is pure JSON / python-literal ────────────────────────
    decoded = _try_parse(text)
    if decoded is not None:
        return _dict_to_research_result(decoded)

    # ── Try 2: fenced code block (``` or ```json) — most explicit signal ───────
    # Collect ALL fenced blocks and pick the best-scoring one.
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
        return _dict_to_research_result(best_from_fence)

    # ── Try 3: raw scan of the entire text ────────────────────────────────────
    decoded = _best_json_in_text(text)
    if decoded is not None:
        return _dict_to_research_result(decoded)

    # ── Fallback: keep raw text as summary ────────────────────────────────────
    logger.warning(
        "Could not parse structured JSON from researcher response; "
        "falling back to raw_summary. text[:400]=%r",
        text[:400].replace("\n", "\\n"),
    )
    return ResearchResult(raw_summary=text, sources=[])


def _build_ingestion_items(result: ResearchResult) -> list[dict]:
    items: list[dict] = []
    default_year = str(datetime.now(timezone.utc).year)

    for u in result.universities:
        content_parts = []
        if u.university_name:
            content_parts.append(f"University: {u.university_name}")
        if u.major:
            content_parts.append(f"Major: {u.major}")
        if u.benchmark_score is not None:
            content_parts.append(f"Benchmark score: {u.benchmark_score}")
        if u.admission_method:
            content_parts.append(f"Admission method: {u.admission_method}")
        if u.required_documents:
            content_parts.append(f"Required documents: {', '.join(u.required_documents)}")
        if u.deadline:
            content_parts.append(f"Deadline: {u.deadline}")
        if u.tuition_fee:
            content_parts.append(f"Tuition fee: {u.tuition_fee}")
        if u.website:
            content_parts.append(f"Website: {u.website}")
        if u.notes:
            content_parts.append(f"Notes: {u.notes}")

        content = "\n".join(content_parts).strip()
        if not content:
            continue

        source_url = u.source_url or (result.sources[0] if len(result.sources) == 1 else "")
        items.append(
            {
                "content": content,
                "source_url": source_url,
                "university": u.university_name or "",
                "year": default_year,
            }
        )

    general_parts = []
    if result.general_requirements:
        general_parts.append(
            f"General requirements: {', '.join(result.general_requirements)}"
        )
    if result.admission_methods:
        general_parts.append(
            f"Admission methods: {', '.join(result.admission_methods)}"
        )
    if result.important_deadlines:
        deadlines = "; ".join(
            f"{k}: {v}" for k, v in result.important_deadlines.items()
        )
        general_parts.append(f"Important deadlines: {deadlines}")
    if result.raw_summary:
        general_parts.append(f"Summary: {result.raw_summary}")

    general_content = "\n".join(general_parts).strip()
    if general_content:
        source_url = result.sources[0] if len(result.sources) == 1 else ""
        items.append(
            {
                "content": general_content,
                "source_url": source_url,
                "university": "",
                "year": default_year,
            }
        )

    return items


def _dict_to_research_result(data: dict) -> ResearchResult:
    """Convert the dict into a ResearchResult Pydantic model, process the partial data."""
    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        return [value]

    def _as_dict(value) -> dict:
        if isinstance(value, dict):
            return value
        return {}

    def _as_str_str_dict(value) -> dict[str, str]:
        """Coerce to dict[str, str], dropping invalid/None entries."""
        raw = _as_dict(value)
        out: dict[str, str] = {}
        for k, v in raw.items():
            if k is None:
                continue
            key = str(k).strip()
            if not key:
                continue
            if v is None:
                # Don't include null deadlines; schema expects strings.
                continue
            if isinstance(v, (dict, list, tuple)):
                val = json.dumps(v, ensure_ascii=False)
            else:
                val = str(v)
            val = val.strip()
            if not val:
                continue
            out[key] = val
        return out

    def _as_str(value) -> str:
        if value is None:
            return ""
        return str(value)

    def _as_float_or_none(value):
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    universities = []
    for u in data.get("universities", []):
        if not isinstance(u, dict):
            continue

        payload = {
            # Todo: enumerate field name of class
            "university_name": _as_str(u.get("university_name")),
            "major": _as_str(u.get("major")),
            "benchmark_score": _as_float_or_none(u.get("benchmark_score")),
            "admission_method": _as_str(u.get("admission_method")),
            "required_documents": _as_list(u.get("required_documents")),
            "deadline": u.get("deadline"),
            "tuition_fee": u.get("tuition_fee"),
            "website": u.get("website"),
            "source_url": u.get("source_url"),
            "notes": _as_str(u.get("notes")),
        }

        try:
            universities.append(UniversityInfo(**payload))
        except Exception as e:
            logger.warning(f"Skipping invalid university entry: {e}. Entry={payload!r}")
            continue

    return ResearchResult(
        query=_as_str(data.get("query", "")),
        universities=universities,
        general_requirements=_as_list(data.get("general_requirements", [])),
        important_deadlines=_as_str_str_dict(data.get("important_deadlines", {})),
        admission_methods=_as_list(data.get("admission_methods", [])),
        raw_summary=_as_str(data.get("raw_summary", "")),
        sources=_as_list(data.get("sources", [])),
    )


def _get_credential():
    """
    Get Azure credential in order of priority:
    1. DefaultAzureCredential (support Managed Identity, CLI, env)
    2. AzureCliCredential (fallback for local dev)
    """
    try:
        return DefaultAzureCredential()
    except Exception:
        return AzureCliCredential()


# =============================================================================
# ResearcherAgent class
# =============================================================================

class ResearcherAgent(BaseAgent):
    """
    The Researcher Agent uses Microsoft Agent Framework (FoundryChatClient).

    How it works:
    1. Initialize FoundryChatClient with FOUNDRY_PROJECT_ENDPOINT + credential
    2. Create an Agent with .as_agent(tools=[...], instructions=...)
    3. Call agent.run(message) - the framework handles it automatically:
       - Send a message to LLM (GPT-4o, GPT4.1,...)
       - Receives a tool call request
       - Executes tool (search_admission_articles / query_knowledge_base / scrape_url)
       - Sends the tool's results back to LLM
       - Repeats until a final answer is obtained
    4. Parse response.text → ResearchResult
    """

    name = "researcher"
    description = "Collections and Extracts university admissions information"

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._foundry_agent: Optional[Agent] = None

    def _get_agent(self) -> Agent:
        """
        Lazy-init Foundry Agent.

        FoundryChatClient read FOUNDRY_PROJECT_ENDPOINT from the environment if
        it's not passed directly
        """
        if self._foundry_agent is None:
            project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
            model = os.environ.get("FOUNDRY_MODEL", "gpt-4o-admission")

            if not project_endpoint:
                raise EnvironmentError("Missing variable FOUNDRY_PROJECT_ENDPOINT in .env.\n")

            client = FoundryChatClient(
                project_endpoint=project_endpoint,
                model=model,
                credential=_get_credential(),
            )

            self._foundry_agent = client.as_agent(
                name="ResearcherAgent",
                instructions=_load_prompt(),
                tools=ALL_RESEARCHER_TOOLS,
            )

            self.log(
                f"FoundryChatClient initialized. Endpoint: {project_endpoint} | Model: {model}",
                "info",
            )

        return self._foundry_agent

    async def run(self, state: AgentState) -> AgentState:
        """
        Run the Researcher Agent and update AgentState with result.

        Args:
            state: AgentState contains student_profile, user_message, input_urls.

        Returns:
            AgentState has been update with research_result.
        """
        state.current_agent = self.name
        self.log("Starting research task...", "info")

        try:
            agent = self._get_agent()
            user_message = _build_user_message(state)

            self.log(f"Sending message to agent ({len(user_message)} chars)...", "info")

            # agent.run() automatically:
            # - Send message
            # - Handle tool calls (search / rag / scrape)
            # - Return AgentResponse when finished
            response = await agent.run(user_message)

            raw_text = response.text
            self.log(f"Raw response: {len(raw_text)} chars", "info")

            # Parse kết quả
            research_result = _parse_research_result(raw_text)
            research_result.query = state.user_message
            research_result.researched_at = datetime.now(timezone.utc)

            state.research_result = research_result
            state.mark_agent_done(self.name)

            ingestion_items = _build_ingestion_items(research_result)
            if ingestion_items:
                try:
                    ingestion_results = ingest_university_data(ingestion_items)
                    self.log(
                        f"Ingested {len(ingestion_results)} items into knowledge base.",
                        "info",
                    )
                except Exception as e:
                    logger.exception(f"Ingestion failed: {e}")
                    state.add_error(f"Ingestion error: {str(e)}")
            else:
                self.log("No ingestion items to add.", "info")

            self.log(
                f"Research complete. Found {len(research_result.universities)} universities, "
                f"{len(research_result.sources)} sources.",
                "info",
            )

        except Exception as e:
            logger.exception(f"ResearcherAgent failed: {e}")
            state.add_error(f"ResearcherAgent error: {str(e)}")
            state.research_result = ResearchResult(
                raw_summary=f"Research failed: {str(e)}"
            )

        return state

    async def run_stream(self, state: AgentState):
        """
        Streaming version of run() - yield individual chunk of text.
        Used for WebSocket endpoint or realtime UI.

        Usage:
            async for chunk in researcher.run_stream(state):
                print(chunk, end="", flush=True)
        """
        try:
            agent = self._get_agent()
            user_message = _build_user_message(state)

            async for chunk in agent.run(user_message, stream=True):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.exception(f"ResearcherAgent stream failed: {e}")
            yield f"[ERROR] {str(e)}"