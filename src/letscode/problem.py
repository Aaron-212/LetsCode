from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .ai_http import OpenAIHttpClients, create_chat_model
from .leetcode import LeetCodeFetcher
from .markdown import html_to_markdown
from .model.leetcode import LeetCodeQuestion
from .storage import GeneratedHint, ProblemStore, StoredProblem


@dataclass(frozen=True)
class ProblemMarkdown:
    problem: str
    fetched_hints: list[str]
    generated_hints: list[GeneratedHint]


CUSTOM_PROBLEM_SYSTEM_PROMPT = """\
You create original LeetCode-style programming interview questions.
Return only Markdown, with no surrounding commentary or code fences.

The problem must include:
- A title as an H1 heading.
- Difficulty.
- A clear problem statement.
- The expected Python Solution method signature.
- At least two examples with inputs and outputs.
- Constraints.

Make the question original and consistent with the user's requested topic,
constraints, or difficulty. Do not copy an existing LeetCode problem. Do not
include a solution.
"""


async def load_problem_markdown(
    question_id: int = 1,
    *,
    fetcher: LeetCodeFetcher | None = None,
    store: ProblemStore | None = None,
) -> ProblemMarkdown:
    problem_store = store or ProblemStore()

    if question_id < 0:
        return _stored_or_fallback_problem(question_id, problem_store)

    problem_fetcher = fetcher or LeetCodeFetcher()
    question = await problem_fetcher.get_problem(question_id)

    if question is None:
        return _stored_or_fallback_problem(question_id, problem_store)

    problem, hints = format_problem_markdown(question)
    try:
        stored_problem = problem_store.save_fetched_problem(
            question.questionId,
            problem,
            hints,
        )
    except OSError:
        stored_problem = problem_store.load_problem(question.questionId)
        if stored_problem is None:
            return ProblemMarkdown(
                problem=problem,
                fetched_hints=hints,
                generated_hints=[],
            )

    return _problem_from_store(stored_problem)


def format_problem_markdown(question: LeetCodeQuestion) -> tuple[str, list[str]]:
    content = html_to_markdown(question.content)
    hints = [html_to_markdown(hint) for hint in question.hints]
    problem = (
        f"# {question.questionId}. {question.title}\n\n"
        f"Difficulty: {question.difficulty}\n\n"
        f"{content}"
    )
    return problem, hints


def generate_custom_problem_markdown(
    prompt: str,
    model_slug: str = "openai:gpt-5.4-mini",
    *,
    model: Any | None = None,
    http_clients: OpenAIHttpClients | None = None,
) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("custom problem prompt cannot be empty")

    chat_model = create_chat_model(
        model_slug,
        http_clients=http_clients,
        model=model,
    )
    response = chat_model.invoke(
        [
            SystemMessage(content=CUSTOM_PROBLEM_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    problem_markdown = _format_message_content(response.content).strip()
    problem_markdown = _strip_markdown_fence(problem_markdown)

    if not problem_markdown.startswith("#"):
        problem_markdown = f"# Custom Problem\n\n{problem_markdown}"

    return problem_markdown.strip() + "\n"


def _stored_or_fallback_problem(
    question_id: int,
    store: ProblemStore,
) -> ProblemMarkdown:
    stored_problem = store.load_problem(question_id)
    if stored_problem is not None:
        return _problem_from_store(stored_problem)

    fallback_markdown = f"# Problem {question_id}\n\nUnable to load this problem."
    return ProblemMarkdown(
        problem=fallback_markdown,
        fetched_hints=[fallback_markdown],
        generated_hints=[],
    )


def _problem_from_store(stored_problem: StoredProblem) -> ProblemMarkdown:
    return ProblemMarkdown(
        problem=stored_problem.problem,
        fetched_hints=stored_problem.fetched_hints,
        generated_hints=stored_problem.generated_hints,
    )


def _format_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(_format_content_part(item) for item in content)

    return str(content)


def _format_content_part(item: Any) -> str:
    if isinstance(item, str):
        return item

    if isinstance(item, dict) and "text" in item:
        return str(item["text"])

    return str(item)


def _strip_markdown_fence(markdown: str) -> str:
    cleaned = markdown.strip()
    if not cleaned.startswith("```"):
        return cleaned

    first_line_end = cleaned.find("\n")
    if first_line_end == -1:
        return cleaned

    cleaned = cleaned[first_line_end + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()
