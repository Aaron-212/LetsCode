import inspect
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from rich.syntax import Syntax
from textual.app import App, ComposeResult
from textual.widgets import Markdown, Static

from letscode.main import (
    ChatBubble,
    ChatEntry,
    JudgeOutput,
    LetsCodeApp,
    SolutionPreview,
    main,
    parse_cli_args,
    parse_question_id,
)


class JudgeOutputSmokeApp(App[None]):
    def compose(self) -> ComposeResult:
        yield JudgeOutput()


class SolutionPreviewSmokeApp(App[None]):
    def compose(self) -> ComposeResult:
        yield SolutionPreview()


class ChatBubbleTest(unittest.TestCase):
    def test_chat_bubble_uses_role_classes(self) -> None:
        bubble = ChatBubble(ChatEntry("user", "How should I start?"))
        child = list(bubble.compose())[0]

        self.assertTrue(bubble.has_class("chat-row"))
        self.assertTrue(bubble.has_class("user"))
        self.assertTrue(child.has_class("chat-bubble"))
        self.assertTrue(child.has_class("user"))

    def test_chat_bubble_preserves_agent_role(self) -> None:
        bubble = ChatBubble(ChatEntry("agent", "Look for a hash map invariant."))
        child = list(bubble.compose())[0]

        self.assertTrue(bubble.has_class("agent"))
        self.assertTrue(child.has_class("agent"))


class JudgeTabWidgetsTest(unittest.TestCase):
    def test_solution_preview_is_composed_from_static_widgets(self) -> None:
        children = list(SolutionPreview().compose())

        self.assertTrue(all(isinstance(child, Static) for child in children))
        self.assertFalse(any(isinstance(child, Markdown) for child in children))

    def test_solution_preview_uses_syntax_renderable_for_code(self) -> None:
        code_widget = SolutionPreview._code_widget("class Solution:\n    pass")

        self.assertIsInstance(getattr(code_widget, "_Static__content"), Syntax)

    def test_judge_output_is_composed_from_static_widgets(self) -> None:
        children = list(JudgeOutput().compose())

        self.assertTrue(all(isinstance(child, Static) for child in children))
        self.assertFalse(any(isinstance(child, Markdown) for child in children))

    def test_judge_output_does_not_override_textual_render_with_coroutine(
        self,
    ) -> None:
        self.assertFalse(inspect.iscoroutinefunction(JudgeOutput._render))


class JudgeOutputSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_judge_output_renders_when_mounted(self) -> None:
        app = JudgeOutputSmokeApp()

        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            judge_output = app.query_one(JudgeOutput)
            await judge_output.update_status("Ready")
            await pilot.pause()


class SolutionPreviewSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_solution_preview_renders_when_mounted(self) -> None:
        app = SolutionPreviewSmokeApp()

        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            solution_preview = app.query_one(SolutionPreview)
            await solution_preview.update_solution("class Solution:\n    pass")
            await pilot.pause()


class ParseQuestionIdTest(unittest.TestCase):
    def test_parse_question_id_accepts_custom_negative_ids(self) -> None:
        self.assertEqual(parse_question_id("-1"), -1)

    def test_parse_question_id_rejects_non_integer_values(self) -> None:
        self.assertIsNone(parse_question_id("two-sum"))


class CliArgsTest(unittest.TestCase):
    def test_parse_cli_args_accepts_problem_id(self) -> None:
        args = parse_cli_args(["--problem", "42"])

        self.assertEqual(args.problem, 42)
        self.assertFalse(args.reset)

    def test_parse_cli_args_accepts_custom_negative_problem_id(self) -> None:
        args = parse_cli_args(["--problem", "-1"])

        self.assertEqual(args.problem, -1)

    def test_parse_cli_args_rejects_non_integer_problem_id(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_cli_args(["--problem", "two-sum"])


class MainCliTest(unittest.TestCase):
    def test_main_reset_resets_store_and_does_not_start_app(self) -> None:
        with (
            patch("letscode.main.ProblemStore") as problem_store_class,
            patch("letscode.main.LetsCodeApp") as app_class,
        ):
            main(["--reset"])

        problem_store_class.return_value.reset_all.assert_called_once_with()
        app_class.assert_not_called()

    def test_main_problem_starts_app_with_initial_problem_id(self) -> None:
        with patch("letscode.main.LetsCodeApp") as app_class:
            main(["--problem", "42"])

        app_class.assert_called_once_with(initial_problem_id=42)
        app_class.return_value.run.assert_called_once_with()


class SlashCommandTest(unittest.TestCase):
    def test_random_slash_command_starts_random_problem_worker(self) -> None:
        app = LetsCodeApp()
        workers: list[object] = []

        def run_worker(work: object, *args: object, **kwargs: object) -> object:
            workers.append(work)
            if inspect.iscoroutine(work):
                work.close()
            return work

        app.run_worker = run_worker  # type: ignore[method-assign]

        self.assertTrue(app.handle_slash_command("/random"))
        self.assertEqual(len(workers), 1)

    def test_unknown_slash_command_is_consumed(self) -> None:
        app = LetsCodeApp()
        notifications: list[tuple[str, str | None]] = []

        def notify(message: str, *args: object, **kwargs: object) -> None:
            notifications.append((message, kwargs.get("severity")))

        app.notify = notify  # type: ignore[method-assign]

        self.assertTrue(app.handle_slash_command("/typo ask the agent"))
        self.assertEqual(
            notifications,
            [
                (
                    "Unknown command: /typo. Available commands: "
                    "/problem, /new, /hint, /fix, /reset, /random",
                    "warning",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
