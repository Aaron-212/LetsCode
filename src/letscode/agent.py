from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, cast

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI


@dataclass
class LetsCodeAgentState:
    messages: list[BaseMessage] = field(default_factory=list)
    current_problem_id: int | None = None
    problem_markdown: str = ""
    fetched_hints: list[str] = field(default_factory=list)
    generated_hints: list[str] = field(default_factory=list)
    current_solution_code: str = ""
    last_judge_output: str = ""
    difficulty: str | None = None
    attempts_count: int = 0


class LetsCodeAgent:
    SYSTEM_PROMPT = """\
    You are a helpful LeetCode assistant.

    You have access to tools for the currently loaded problem:
    - get_current_problem_context: read the active problem statement and metadata.
    - get_existing_hints: read hints that were already fetched or generated.
    - get_current_solution_code: read the active Python solution file.
    - get_last_judge_output: read the latest judge/test output.
    - record_solution_attempt: increment the user's solution-attempt counter.

    For problem-specific requests, use the tools instead of asking the user to
    paste the problem statement. For hint requests, call
    get_current_problem_context before answering. Give progressive hints that
    help the user make the next step without revealing a full solution unless
    the user explicitly asks for one. For fix requests, use the problem,
    solution, and judge-output tools, then return only the corrected Python code
    for the full solution file.
    """

    def __init__(self, model_slug: str) -> None:
        model = ChatOpenAI(model=model_slug)
        self.state = LetsCodeAgentState()

        self.agent = create_deep_agent(
            model=model,
            tools=tool_factory(self.state),
            system_prompt=self.SYSTEM_PROMPT,
        )

    @property
    def messages(self) -> list[BaseMessage]:
        return self.state.messages

    def set_problem_context(
        self,
        *,
        question_id: int,
        problem_markdown: str,
        fetched_hints: Sequence[str] = (),
        generated_hints: Sequence[str] = (),
        difficulty: str | None = None,
    ) -> None:
        self.state.current_problem_id = question_id
        self.state.problem_markdown = problem_markdown
        self.state.fetched_hints = list(fetched_hints)
        self.state.generated_hints = list(generated_hints)
        self.state.difficulty = difficulty
        self.state.attempts_count = 0
        self.state.messages.clear()

    def set_solution_context(
        self,
        *,
        solution_code: str,
        judge_output: str,
    ) -> None:
        self.state.current_solution_code = solution_code
        self.state.last_judge_output = judge_output

    def chat(self, user_text: str) -> str:
        return self._invoke(user_text)

    def generate_hint(self, user_prompt: str | None = None) -> str:
        prompt_suffix = f"\nUser focus: {user_prompt}" if user_prompt else ""
        hint = self._invoke(
            "Generate one additional concise, progressive hint for the current "
            "LeetCode problem. Use the current-problem tools for context. Do not "
            "include full code or solve the problem outright."
            f"{prompt_suffix}"
        )
        self.state.generated_hints.append(hint)
        return hint

    def fix_solution(self, suite: str) -> str:
        self.state.attempts_count += 1
        response = self._invoke(
            "Fix the current Python solution for the active LeetCode problem. "
            f"The failing suite is {suite}. Use the current-problem, current-solution, "
            "and judge-output tools before answering. Return only corrected Python "
            "code for the complete solution file, with no Markdown fences or explanation."
        )
        fixed_code = extract_python_code(response)
        self.state.current_solution_code = fixed_code
        return fixed_code

    def _invoke(self, user_text: str) -> str:
        self.state.messages.append(HumanMessage(content=user_text))
        result = self.agent.invoke(cast(Any, {"messages": self.state.messages}))
        self.state.messages = list(result["messages"])

        return latest_ai_message_text(self.state.messages)


def latest_ai_message_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return format_message_content(message.content)

    return "The agent completed without returning a response."


def format_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(format_content_part(item) for item in content)

    return str(content)


def format_content_part(item: Any) -> str:
    if isinstance(item, str):
        return item

    if isinstance(item, dict) and "text" in item:
        return str(item["text"])

    return str(item)


def extract_python_code(content: str) -> str:
    stripped = content.strip()
    fence_start = stripped.find("```")
    if fence_start == -1:
        return stripped + "\n"

    code_start = stripped.find("\n", fence_start)
    if code_start == -1:
        return stripped + "\n"

    code_end = stripped.find("```", code_start + 1)
    if code_end == -1:
        return stripped[code_start + 1 :].strip() + "\n"

    return stripped[code_start + 1 : code_end].strip() + "\n"


def tool_factory(state: LetsCodeAgentState) -> list[BaseTool]:
    @tool
    def get_current_problem_context() -> str:
        """Return the active LeetCode problem statement and metadata."""
        if not state.problem_markdown:
            return "No LeetCode problem is currently loaded."

        metadata = []
        if state.current_problem_id is not None:
            metadata.append(f"Problem ID: {state.current_problem_id}")
        if state.difficulty:
            metadata.append(f"Difficulty: {state.difficulty}")
        metadata.append(f"Recorded solution attempts: {state.attempts_count}")

        return "\n".join([*metadata, "", state.problem_markdown]).strip()

    @tool
    def get_existing_hints() -> str:
        """Return hints that already exist for the active problem."""
        hints = [
            *(
                f"Fetched hint {index}: {hint}"
                for index, hint in enumerate(state.fetched_hints, start=1)
            ),
            *(
                f"Generated hint {index}: {hint}"
                for index, hint in enumerate(state.generated_hints, start=1)
            ),
        ]

        if not hints:
            return "No hints are available for the active problem."

        return "\n\n".join(hints)

    @tool
    def get_current_solution_code() -> str:
        """Return the active Python solution code."""
        if not state.current_solution_code:
            return "No solution code is currently loaded."

        return state.current_solution_code

    @tool
    def get_last_judge_output() -> str:
        """Return the latest judge output for the active solution."""
        if not state.last_judge_output:
            return "No judge output is currently available."

        return state.last_judge_output

    @tool
    def record_solution_attempt() -> str:
        """Record that the user made another solution attempt."""
        state.attempts_count += 1
        return f"Recorded attempt {state.attempts_count}."

    return [
        get_current_problem_context,
        get_existing_hints,
        get_current_solution_code,
        get_last_judge_output,
        record_solution_attempt,
    ]
