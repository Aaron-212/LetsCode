from __future__ import annotations

from dataclasses import dataclass

from .leetcode import LeetCodeFetcher
from .markdown import html_to_markdown
from .model.leetcode import LeetCodeQuestion
from .storage import GeneratedHint, ProblemStore, StoredProblem


@dataclass(frozen=True)
class ProblemMarkdown:
    problem: str
    fetched_hints: list[str]
    generated_hints: list[GeneratedHint]


async def load_problem_markdown(
    question_id: int = 1,
    *,
    fetcher: LeetCodeFetcher | None = None,
    store: ProblemStore | None = None,
) -> ProblemMarkdown:
    problem_store = store or ProblemStore()
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
