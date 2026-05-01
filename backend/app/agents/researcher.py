"""
Researcher Agent - Collects and Extracts university admissions information

Uses Azure AI Agent Service (azure-ai-agents SDK) with:
- GPT-4o for reasoning and extraction
- FunctionTool: scrape_url, search_admission_info, query_knowledge_base
- create_and_process_run to automatically process tool calls

Flow:
    Input (URL/query + student profile)
    → AgentsClient.create_and_process_run()
    → Tool calls (web scrape / RAG)
    → GPT-4o analysis + extract
    → ResearchResult (structured JSON)
    → AgentState.research_result
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FunctionTool,
    MessageRole,
    RunStatus,
    ToolSet,
)
from azure.core.credentials import AzureKeyCredential

from app.agents.base import BaseAgent
from app.schemas.agent_state import AgentState, ResearchResult, UniversityInfo
from app.tools.web_search_tool import (
    SCRAPE_URL_DEFINITION,
    SEARCH_ADMISSION_DEFINITION,
    TOOL_FUNCTIONS as WEB_TOOL_FUNCTIONS,
)
from app.tools.rag_tool import (
    QUERY_KB_DEFINITION,
    TOOL_FUNCTIONS as RAG_TOOL_FUNCTIONS,
)

logger = logging.getLogger(__name__)

# ── Merge all tool functions ──
ALL_TOOL_FUNCTIONS = {**WEB_TOOL_FUNCTIONS, **RAG_TOOL_FUNCTIONS}


def _load_prompt() -> str:
    """Read system prompt from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "researcher.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are a researcher agent that collects university admission information."


def _build_user_message(state: AgentState) -> str:
    """
    Tạo user message từ AgentState.
    Bao gồm profile học sinh + câu hỏi / URL cần nghiên cứu.
    """
    profile = state.student_profile
    parts = []

    if state.user_message:
        parts.append(f"**Yêu cầu của học sinh:** {state.user_message}")

    if profile.name:
        parts.append(
            f"**Thông tin học sinh:**\n"
            f"- Tên: {profile.name}\n"
            f"- GPA: {profile.gpa or 'Chưa có'}\n"
            f"- Điểm mục tiêu: {json.dumps(profile.target_scores, ensure_ascii=False)}\n"
            f"- Ngành mong muốn: {', '.join(profile.preferred_majors) or 'Chưa xác định'}\n"
            f"- Trường mong muốn: {', '.join(profile.preferred_universities) or 'Chưa xác định'}"
        )

    if state.input_urls:
        parts.append(
            f"**URL cần phân tích:**\n" + "\n".join(f"- {u}" for u in state.input_urls)
        )

    parts.append(
        "\n**Yêu cầu output:** Trả về JSON hợp lệ theo schema ResearchResult với các trường: "
        "universities, general_requirements, important_deadlines, admission_methods, "
        "raw_summary, sources."
    )

    return "\n\n".join(parts)


def _parse_research_result(raw_text: str) -> ResearchResult:
    """
    Parse JSON từ response của agent thành ResearchResult.
    Xử lý các trường hợp agent trả về text kèm JSON.
    """
    # Tìm JSON block trong response
    text = raw_text.strip()

    # Thử parse trực tiếp
    try:
        data = json.loads(text)
        return _dict_to_research_result(data)
    except json.JSONDecodeError:
        pass

    # Tìm JSON trong markdown code block
    import re
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return _dict_to_research_result(data)
        except json.JSONDecodeError:
            pass

    # Tìm JSON object trong text thô
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            return _dict_to_research_result(data)
        except json.JSONDecodeError:
            pass

    # Fallback: tạo result từ raw text
    logger.warning("Could not parse structured JSON from researcher. Using raw_summary fallback.")
    return ResearchResult(raw_summary=text, sources=[])


def _dict_to_research_result(data: dict) -> ResearchResult:
    """Chuyển dict thành ResearchResult Pydantic model."""
    universities = []
    for u in data.get("universities", []):
        universities.append(
            UniversityInfo(
                university_name=u.get("university_name", ""),
                major=u.get("major", ""),
                benchmark_score=u.get("benchmark_score"),
                admission_method=u.get("admission_method", ""),
                required_documents=u.get("required_documents", []),
                deadline=u.get("deadline"),
                tuition_fee=u.get("tuition_fee"),
                website=u.get("website"),
                source_url=u.get("source_url"),
                notes=u.get("notes", ""),
            )
        )

    return ResearchResult(
        query=data.get("query", ""),
        universities=universities,
        general_requirements=data.get("general_requirements", []),
        important_deadlines=data.get("important_deadlines", {}),
        admission_methods=data.get("admission_methods", []),
        raw_summary=data.get("raw_summary", ""),
        sources=data.get("sources", []),
    )


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent sử dụng Azure AI Agent Service.

    Công nghệ:
    - azure-ai-agents: AgentsClient, FunctionTool, ToolSet
    - create_and_process_run: tự động handle tool_calls loop
    - GPT-4o deployment: gpt-4o-admission
    """

    name = "researcher"
    description = "Thu thập và trích xuất thông tin tuyển sinh từ nguồn bên ngoài"

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._client: Optional[AgentsClient] = None
        self._agent_id: Optional[str] = None

    def _get_client(self) -> AgentsClient:
        """Lazy-init AgentsClient."""
        if self._client is None:
            connection_string = os.environ.get("AZURE_AI_PROJECT_CONNECTION_STRING")

            if connection_string:
                # Sử dụng connection string (ưu tiên)
                self._client = AgentsClient.from_connection_string(
                    conn_str=connection_string,
                    credential=AzureKeyCredential(os.environ["AZURE_OPENAI_API_KEY"]),
                )
            else:
                # Fallback: dùng endpoint + key
                self._client = AgentsClient(
                    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    credential=AzureKeyCredential(os.environ["AZURE_OPENAI_API_KEY"]),
                )

            self.log("AgentsClient initialized.", "info")
        return self._client

    def _build_toolset(self) -> ToolSet:
        """
        Xây dựng ToolSet với các FunctionTool.
        Azure AI Agent sẽ tự quyết định khi nào gọi tool nào.
        """
        tools = [
            FunctionTool(definitions=[SCRAPE_URL_DEFINITION]),
            FunctionTool(definitions=[SEARCH_ADMISSION_DEFINITION]),
            FunctionTool(definitions=[QUERY_KB_DEFINITION]),
        ]
        return ToolSet(tools=tools)

    def _get_or_create_agent(self, client: AgentsClient) -> str:
        """
        Tạo hoặc lấy agent ID đã tồn tại.
        Trong production nên cache agent_id vào DB để tái sử dụng.
        """
        if self._agent_id:
            return self._agent_id

        toolset = self._build_toolset()
        system_prompt = _load_prompt()

        agent = client.create_agent(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-admission"),
            name="ResearcherAgent",
            instructions=system_prompt,
            tools=toolset.definitions,
            tool_resources=toolset.resources,
        )

        self._agent_id = agent.id
        self.log(f"Created Azure AI Agent: {self._agent_id}", "info")
        return self._agent_id

    def _dispatch_tool_call(self, tool_name: str, tool_args_str: str) -> str:
        """
        Thực thi tool call được yêu cầu bởi agent.

        Args:
            tool_name: Tên function cần gọi.
            tool_args_str: Arguments dạng JSON string.

        Returns:
            Kết quả tool dưới dạng string.
        """
        func = ALL_TOOL_FUNCTIONS.get(tool_name)
        if func is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            args = json.loads(tool_args_str) if tool_args_str else {}
            self.log(f"Calling tool '{tool_name}' with args: {args}", "info")
            result = func(**args)
            return result
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return json.dumps({"error": str(e)})

    async def run(self, state: AgentState) -> AgentState:
        """
        Chạy Researcher Agent:
        1. Tạo thread + message
        2. create_and_process_run (auto tool-call loop)
        3. Parse kết quả → ResearchResult
        4. Cập nhật AgentState

        Args:
            state: AgentState chứa student_profile, user_message, input_urls.

        Returns:
            AgentState với research_result đã được điền.
        """
        state.current_agent = self.name
        self.log("Starting research task...", "info")

        client = self._get_client()

        try:
            agent_id = self._get_or_create_agent(client)

            # Tạo thread mới cho session này
            thread = client.threads.create()
            self.log(f"Created thread: {thread.id}", "info")

            # Thêm user message
            user_message = _build_user_message(state)
            client.messages.create(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=user_message,
            )

            # Chạy agent với auto tool-call handling
            # create_and_process_run tự động gọi tool → submit result → tiếp tục
            run = client.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent_id,
                # Override tool execution với custom dispatcher
                # Note: azure-ai-agents >= 1.1.0 hỗ trợ toolset parameter
                # để auto-dispatch, hoặc dùng manual loop dưới đây
            )

            if run.status == RunStatus.REQUIRES_ACTION:
                # Manual tool execution loop (fallback nếu auto không hoạt động)
                run = self._handle_tool_calls_loop(client, thread.id, run)

            if run.status != RunStatus.COMPLETED:
                error_msg = f"Run ended with status: {run.status}"
                logger.error(error_msg)
                state.add_error(error_msg)
                state.research_result = ResearchResult(raw_summary="Research failed.")
                return state

            # Lấy message cuối cùng từ assistant
            messages = client.messages.list(thread_id=thread.id, order="desc")
            assistant_text = ""
            for msg in messages:
                if msg.role == MessageRole.ASSISTANT:
                    for content_block in msg.content:
                        if hasattr(content_block, "text"):
                            assistant_text = content_block.text.value
                            break
                    if assistant_text:
                        break

            self.log(f"Raw response length: {len(assistant_text)} chars", "info")

            # Parse kết quả
            research_result = _parse_research_result(assistant_text)
            research_result.query = state.user_message

            from datetime import datetime
            research_result.researched_at = datetime.utcnow()

            state.research_result = research_result
            state.mark_agent_done(self.name)

            self.log(
                f"Research complete. Found {len(research_result.universities)} universities.",
                "info",
            )

        except Exception as e:
            logger.exception(f"ResearcherAgent failed: {e}")
            state.add_error(f"ResearcherAgent error: {str(e)}")
            state.research_result = ResearchResult(
                raw_summary=f"Research failed due to error: {str(e)}"
            )

        return state

    def _handle_tool_calls_loop(self, client: AgentsClient, thread_id: str, run) -> object:
        """
        Manual tool execution loop cho trường hợp cần xử lý thủ công.
        Được dùng khi create_and_process_run không tự dispatch tool.
        """
        import time

        max_iterations = 10
        iteration = 0

        while run.status == RunStatus.REQUIRES_ACTION and iteration < max_iterations:
            iteration += 1
            self.log(f"Tool call loop iteration {iteration}", "info")

            tool_outputs = []

            required_action = run.required_action
            if required_action and hasattr(required_action, "submit_tool_outputs"):
                for tool_call in required_action.submit_tool_outputs.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    output = self._dispatch_tool_call(tool_name, tool_args)
                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "output": output}
                    )

            # Submit tool results
            run = client.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )

            # Poll until status changes
            poll_count = 0
            while run.status in (RunStatus.IN_PROGRESS, RunStatus.QUEUED) and poll_count < 30:
                time.sleep(1)
                run = client.runs.get(thread_id=thread_id, run_id=run.id)
                poll_count += 1

        return run

    def cleanup(self) -> None:
        """
        Dọn dẹp agent khi không còn dùng nữa.
        Trong production: nên giữ agent_id và tái sử dụng.
        """
        if self._agent_id and self._client:
            try:
                self._client.delete_agent(self._agent_id)
                self.log(f"Deleted agent: {self._agent_id}", "info")
                self._agent_id = None
            except Exception as e:
                logger.warning(f"Failed to delete agent: {e}")