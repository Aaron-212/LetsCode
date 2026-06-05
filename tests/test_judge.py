import unittest

from letscode.judge import (
    GeneratedJudge,
    JudgeRun,
    TestResult,
    format_judge_error,
    format_judge_run,
    is_judge_suite,
    run_solution_suite,
)


class JudgeHelpersTest(unittest.TestCase):
    def test_run_solution_suite_dispatches_testfire(self) -> None:
        judge = GeneratedJudge(
            problem_hash="abc123",
            initial_code_snippet="class Solution:\n    pass\n",
            testfire_code=(
                "def testfire(Solution):\n"
                "    assert Solution().add(1, 2) == 3\n"
            ),
            full_test_code=(
                "def full_test_suite(Solution):\n"
                "    assert Solution().add(-1, 1) == 0\n"
            ),
            model_slug="test-model",
        )
        solution_code = (
            "class Solution:\n"
            "    def add(self, left, right):\n"
            "        return left + right\n"
        )

        run = run_solution_suite("testfire", solution_code, judge)

        self.assertTrue(run.passed)
        self.assertEqual([result.suite for result in run.results], ["testfire"])

    def test_format_judge_run_includes_failures_and_errors(self) -> None:
        run = JudgeRun(
            passed=False,
            results=[
                TestResult(suite="testfire", passed=False, message="wrong answer"),
                TestResult(suite="full", passed=False, error="ValueError: boom"),
            ],
        )

        rendered = format_judge_run("full", run)

        self.assertIn("Full tests: failed", rendered)
        self.assertIn("## testfire: failed", rendered)
        self.assertIn("wrong answer", rendered)
        self.assertIn("```text\nValueError: boom\n```", rendered)

    def test_format_judge_error_includes_exception_type(self) -> None:
        rendered = format_judge_error("testfire", RuntimeError("bad test"))

        self.assertIn("Testfire tests: error", rendered)
        self.assertIn("RuntimeError: bad test", rendered)

    def test_run_solution_suite_reports_assertion_context(self) -> None:
        judge = GeneratedJudge(
            problem_hash="abc123",
            initial_code_snippet="class Solution:\n    pass\n",
            testfire_code="def testfire(Solution):\n    pass\n",
            full_test_code=(
                "def full_test_suite(Solution):\n"
                "    actual = Solution().add(1, 2)\n"
                "    expected = 4\n"
                "    assert actual == expected\n"
            ),
            model_slug="test-model",
        )
        solution_code = (
            "class Solution:\n"
            "    def add(self, left, right):\n"
            "        return left + right\n"
        )

        run = run_solution_suite("full", solution_code, judge)

        self.assertFalse(run.passed)
        [result] = run.results
        self.assertIn(
            "Assertion failed in full_test_suite at line 4: "
            "assert actual == expected",
            result.message,
        )
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("Location: <generated_full_tests>:4", result.error)
        self.assertIn("Failing assertion:\n  assert actual == expected", result.error)
        self.assertIn("actual = 3", result.error)
        self.assertIn("expected = 4", result.error)

    def test_is_judge_suite_validates_known_suites(self) -> None:
        self.assertTrue(is_judge_suite("testfire"))
        self.assertTrue(is_judge_suite("full"))
        self.assertFalse(is_judge_suite("smoke"))


if __name__ == "__main__":
    unittest.main()
