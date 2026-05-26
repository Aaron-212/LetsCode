import asyncio
import unittest
from types import SimpleNamespace

from letscode.model.leetcode import LeetCodeQuestion
from letscode.problem import (
    format_problem_markdown,
    generate_custom_problem_markdown,
    load_problem_markdown,
)
from letscode.storage import GeneratedHint, StoredProblem


class FakeFetcher:
    def __init__(self, question: LeetCodeQuestion | None) -> None:
        self.question = question
        self.called = False

    async def get_problem(self, question_id: int) -> LeetCodeQuestion | None:
        self.called = True
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


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None

    def invoke(self, messages: object) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content=self.content)


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

    def test_load_custom_problem_uses_store_without_fetching(self) -> None:
        stored_problem = StoredProblem(
            question_id=-1,
            problem="# Custom",
            fetched_hints=[],
            generated_hints=[],
        )
        fetcher = FakeFetcher(None)

        result = asyncio.run(
            load_problem_markdown(
                -1,
                fetcher=fetcher,
                store=FakeStore(stored_problem),
            )
        )

        self.assertEqual(result.problem, "# Custom")
        self.assertFalse(fetcher.called)

    def test_generate_custom_problem_markdown_normalizes_model_output(self) -> None:
        model = FakeModel("```markdown\n# Prefix Windows\n\nDifficulty: Medium\n```")

        result = generate_custom_problem_markdown(
            "strings with prefix constraints",
            model=model,
        )

        self.assertEqual(result, "# Prefix Windows\n\nDifficulty: Medium\n")
        self.assertIsNotNone(model.messages)

    def test_generate_custom_problem_markdown_rejects_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            generate_custom_problem_markdown(" ", model=FakeModel("# Unused"))


if __name__ == "__main__":
    unittest.main()
