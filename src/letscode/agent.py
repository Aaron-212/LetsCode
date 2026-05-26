from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI


@dataclass
class LetsCodeAgentState:
    messages: list[BaseMessage] = field(default_factory=list)
    current_problem_id: int | None = None
    difficulty: str | None = None
    attempts_count: int = 0


class LetsCodeAgent:
    SYSTEM_PROMPT = """\
    You are a helpful assistant that can write code.
    Assist the user on solving LeetCode problems.

    You have access to the following tools:
        - TODO
    """

    def __init__(self, model_slug: str) -> None:
        model = ChatOpenAI(model=model_slug)

        self.agent = create_deep_agent(
            model=model, tools=tool_factory(), system_prompt=self.SYSTEM_PROMPT
        )
        self.state = LetsCodeAgentState()

    @property
    def messages(self) -> list[BaseMessage]:
        return self.state.messages

    def chat(self, user_text: str) -> str:
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


def tool_factory() -> list:
    return []
