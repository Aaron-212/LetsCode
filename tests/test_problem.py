import asyncio
import unittest

from letscode.model.leetcode import LeetCodeQuestion
from letscode.problem import format_problem_markdown, load_problem_markdown
from letscode.storage import GeneratedHint, StoredProblem


class FakeFetcher:
    def __init__(self, question: LeetCodeQuestion | None) -> None:
        self.question = question

    async def get_problem(self, question_id: int) -> LeetCodeQuestion | None:
        return self.question


class FakeStore:
    def __init__(self, stored_problem: StoredProblem | None = None) -> None:
        self.stored_problem = stored_problem

    def load_problem(self, question_id: int) -> StoredProblem | None:
        return self.stored_problem

    def save_fetched_problem(
        self,
        question_id: int,
        problem: str,
        fetched_hints: list[str],
    ) -> StoredProblem:
        self.stored_problem = StoredProblem(
            question_id=question_id,
            problem=problem,
            fetched_hints=fetched_hints,
            generated_hints=[],
        )
        return self.stored_problem


class ProblemMarkdownTest(unittest.TestCase):
    def test_format_problem_markdown_converts_html(self) -> None:
        question = LeetCodeQuestion(
            questionId=1,
            title="Two Sum",
            difficulty="Easy",
            content="<p>Return <strong>indices</strong>.</p>",
            hints=["<p>Use a hash map.</p>"],
        )

        problem, hints = format_problem_markdown(question)

        self.assertIn("# 1. Two Sum", problem)
        self.assertIn("Difficulty: Easy", problem)
        self.assertIn("indices", problem)
        self.assertNotIn("<p>", problem)
        self.assertEqual(hints, ["Use a hash map."])

    def test_load_problem_markdown_uses_stored_problem_when_fetch_fails(self) -> None:
        stored_problem = StoredProblem(
            question_id=42,
            problem="# Stored",
            fetched_hints=["cached hint"],
            generated_hints=[GeneratedHint(hint="generated hint")],
        )

        result = asyncio.run(
            load_problem_markdown(
                42,
                fetcher=FakeFetcher(None),
                store=FakeStore(stored_problem),
            )
        )

        self.assertEqual(result.problem, "# Stored")
        self.assertEqual(result.fetched_hints, ["cached hint"])
        self.assertEqual(result.generated_hints[0].hint, "generated hint")

    def test_load_problem_markdown_returns_fallback_without_fetch_or_cache(self) -> None:
        result = asyncio.run(
            load_problem_markdown(
                99,
                fetcher=FakeFetcher(None),
                store=FakeStore(),
            )
        )

        self.assertIn("# Problem 99", result.problem)
        self.assertEqual(result.fetched_hints, [result.problem])
        self.assertEqual(result.generated_hints, [])


if __name__ == "__main__":
    unittest.main()
