import asyncio
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.message import Message
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Markdown,
    TabbedContent,
    TabPane,
    TextArea,
)

from .agent import LetsCodeAgent
from .judge import (
    DEFAULT_JUDGE_SUITE,
    GeneratedJudge,
    JudgeSuite,
    format_judge_error,
    format_judge_run,
    generate_judge,
    is_judge_suite,
    run_solution_suite,
)
from .problem import load_problem_markdown
from .solution import (
    editor_command,
    ensure_solution_file,
    read_solution_file,
    solution_path,
    write_solution_file,
)
from .storage import GeneratedHint, ProblemStore


class PromptInput(TextArea):
    """Multiline prompt input that submits on Enter."""

    MIN_CONTENT_LINES = 1
    MAX_CONTENT_LINES = 5
    BORDER_HEIGHT = 2

    BINDINGS = [
        Binding("alt+enter", "new_line", show=False, priority=True),
        Binding("enter", "submit", show=False, priority=True),
        Binding("ctrl+c", "clear_input", show=False, priority=True),
    ]

    class Submitted(Message):
        """Posted when the prompt should be submitted."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_new_line(self) -> None:
        self.insert("\n")

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self.text))

    def action_clear_input(self) -> None:
        self.text = ""

    def on_mount(self) -> None:
        self.resize_to_content()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is self:
            self.resize_to_content()

    def resize_to_content(self) -> None:
        content_lines = max(self.MIN_CONTENT_LINES, self.text.count("\n") + 1)
        visible_lines = min(content_lines, self.MAX_CONTENT_LINES)
        self.styles.height = visible_lines + self.BORDER_HEIGHT


class LetsCodeApp(App):
    DEFAULT_PROBLEM_ID = 1
    DEFAULT_AGENT_MODEL = "openai:gpt-5.4-mini"

    INPUT_SELECTOR = "#input"
    PROBLEM_SELECTOR = "#problem"
    HINTS_SELECTOR = "#hints"
    JUDGE_SELECTOR = "#judge-output"
    SOLUTION_PREVIEW_SELECTOR = "#solution-preview"

    TESTFIRE_BUTTON_ID = "testfire-button"
    FULL_TEST_BUTTON_ID = "full-test-button"
    EDIT_BUTTON_ID = "edit-button"

    LOAD_PROBLEM_COMMAND = "/load_problem"
    HINT_COMMAND = "/hint"
    FIX_COMMAND = "/fix"

    CSS = """\
    Screen {
        layout: vertical;
        background: transparent;
    }

    TabbedContent {
        height: 1fr;
        margin: 1;
        background: transparent;
    }

    TabPane {
        padding: 1;
    }

    #problem, #hints, #judge-output, #solution-preview {
        height: auto;
        background: transparent;
    }

    #judge-buttons {
        height: auto;
        margin-bottom: 1;
    }

    #judge-buttons Button {
        margin-right: 1;
    }

    #input {
        min-height: 3;
        max-height: 7;
        margin: 1;
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Problem", id="problem-tab"):
                yield ScrollableContainer(
                    Markdown("# Loading problem...", id="problem")
                )
            with TabPane("Judge", id="judge-tab"):
                yield ScrollableContainer(
                    Horizontal(
                        Button("Edit", id=self.EDIT_BUTTON_ID),
                        Button("Testfire", id=self.TESTFIRE_BUTTON_ID),
                        Button("Full", id=self.FULL_TEST_BUTTON_ID),
                        id="judge-buttons",
                    ),
                    Markdown(
                        "```python\n# Loading solution...\n```", id="solution-preview"
                    ),
                    Markdown("# Loading problem...", id="judge-output"),
                    id="judge",
                )
            with TabPane("Hints", id="hints-tab"):
                yield ScrollableContainer(
                    Markdown("# Loading problem..."),
                    id="hints",
                )
        yield PromptInput(
            language="markdown",
            placeholder="Type your commands. Enter submits, Alt+Enter adds a new line.",
            id="input",
            compact=True,
            highlight_cursor_line=False,
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(self.INPUT_SELECTOR, PromptInput).focus()
        self.current_problem_id = self.DEFAULT_PROBLEM_ID
        self.current_problem_markdown = ""
        self.current_fetched_hints_markdown: list[str] = []
        self.current_generated_hints_markdown: list[str] = []
        self.generated_hints_count = 0
        self.agent: LetsCodeAgent | None = None
        self.current_judge: GeneratedJudge | None = None
        self.action_load_problem()

    def action_load_problem(self) -> None:
        self.run_worker(self.load_problem(self.current_problem_id), exclusive=True)

    def handle_slash_command(self, message: str) -> bool:
        command, *args = message.split()

        if command == self.LOAD_PROBLEM_COMMAND:
            return self._handle_load_problem_command(args)

        if command == self.HINT_COMMAND:
            return self._handle_hint_command(args)

        if command == self.FIX_COMMAND:
            return self._handle_fix_command(args)

        return False

    def _handle_load_problem_command(self, args: list[str]) -> bool:
        if not args:
            self.action_load_problem()
            return True

        if len(args) == 1 and args[0].isdigit():
            self.run_worker(self.load_problem(int(args[0])), exclusive=True)
            return True

        self.notify("Usage: /load_problem [question_id]", severity="warning")
        return True

    def _handle_hint_command(self, args: list[str]) -> bool:
        if len(args) > 1:
            self.notify("Usage: /hint [HINT_PROMPT]", severity="warning")
            return True

        self.run_worker(self.generate_hint(args[0] if args else None), exclusive=True)
        return True

    def _handle_fix_command(self, args: list[str]) -> bool:
        suite = args[0] if args else DEFAULT_JUDGE_SUITE
        if len(args) > 1 or not is_judge_suite(suite):
            self.notify("Usage: /fix [testfire|full]", severity="warning")
            return True

        self.run_worker(self.fix_solution(suite), exclusive=True)
        return True

    async def load_problem(self, question_id: int) -> None:
        self.current_problem_id = question_id
        self.agent = None
        self.current_judge = None

        problem_markdown = await load_problem_markdown(question_id)
        self.current_problem_markdown = problem_markdown.problem
        self.current_fetched_hints_markdown = problem_markdown.fetched_hints
        self.current_generated_hints_markdown = [
            hint.hint for hint in problem_markdown.generated_hints
        ]
        self.generated_hints_count = len(problem_markdown.generated_hints)

        await self.query_one(self.PROBLEM_SELECTOR, Markdown).update(
            problem_markdown.problem
        )
        await self._update_solution_preview("")
        await self.query_one(self.JUDGE_SELECTOR, Markdown).update(
            "# Judge\n\nGenerating judge from the current problem..."
        )
        await self._render_hints(
            problem_markdown.fetched_hints,
            problem_markdown.generated_hints,
        )
        await self.prepare_judge()

    async def _render_hints(
        self,
        fetched_hints_markdown: list[str],
        generated_hints: list[GeneratedHint],
    ) -> None:
        hints_container = self.query_one(self.HINTS_SELECTOR, ScrollableContainer)
        await hints_container.remove_children()

        if not fetched_hints_markdown and not generated_hints:
            await hints_container.mount(Markdown("No hints available."))
            return

        for index, hint_markdown in enumerate(fetched_hints_markdown, start=1):
            await self._mount_hint(
                hints_container,
                hint_markdown,
                title=f"Hint {index}",
                collapsed=True,
            )

        for index, generated_hint in enumerate(generated_hints, start=1):
            await self._mount_hint(
                hints_container,
                generated_hint.hint,
                title=f"AI Hint {index}",
                collapsed=True,
            )

    async def generate_hint(self, prompt: str | None) -> None:
        if not self.current_problem_markdown:
            self.notify("Problem is still loading.", severity="warning")
            return

        self.notify("Generating hint...")

        try:
            hint_markdown = await asyncio.to_thread(
                self.get_agent().generate_hint,
                prompt,
            )
        except Exception as error:
            self.notify(f"Unable to generate hint: {error}", severity="error")
            return

        total_hints_count = (
            len(self.current_fetched_hints_markdown) + self.generated_hints_count
        )
        self.generated_hints_count += 1
        self.current_generated_hints_markdown.append(hint_markdown)

        try:
            ProblemStore().add_generated_hint(
                self.current_problem_id,
                self.current_problem_markdown,
                self.current_fetched_hints_markdown,
                hint_markdown,
                prompt,
            )
        except OSError as error:
            self.notify(f"Unable to persist hint: {error}", severity="warning")

        hints_container = self.query_one(self.HINTS_SELECTOR, ScrollableContainer)

        if total_hints_count == 0:
            await hints_container.remove_children()

        await self._mount_hint(
            hints_container,
            hint_markdown,
            title=f"AI Hint {self.generated_hints_count}",
            collapsed=False,
        )
        self.notify("Hint generated.")

    async def _mount_hint(
        self,
        container: ScrollableContainer,
        hint_markdown: str,
        *,
        title: str,
        collapsed: bool,
    ) -> None:
        await container.mount(
            Collapsible(
                Markdown(hint_markdown),
                title=title,
                collapsed=collapsed,
            )
        )

    def get_agent(self) -> LetsCodeAgent:
        if self.agent is None:
            model_slug = os.getenv("LETSCODE_AGENT_MODEL", self.DEFAULT_AGENT_MODEL)
            self.agent = LetsCodeAgent(model_slug)
            self.agent.set_problem_context(
                question_id=self.current_problem_id,
                problem_markdown=self.current_problem_markdown,
                fetched_hints=self.current_fetched_hints_markdown,
                generated_hints=self.current_generated_hints_markdown,
            )

        return self.agent

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == self.EDIT_BUTTON_ID:
            event.stop()
            self.run_worker(self.open_solution_editor(), exclusive=True)
            return

        if event.button.id == self.TESTFIRE_BUTTON_ID:
            event.stop()
            self.run_worker(self.run_judge("testfire"), exclusive=True)
            return

        if event.button.id == self.FULL_TEST_BUTTON_ID:
            event.stop()
            self.run_worker(self.run_judge("full"), exclusive=True)
            return

    async def run_judge(self, suite: JudgeSuite) -> None:
        if not self.current_problem_markdown:
            self.notify("Problem is still loading.", severity="warning")
            return

        judge_output = self.query_one(self.JUDGE_SELECTOR, Markdown)
        await judge_output.update(f"# Judge\n\nGenerating {suite} judge...")

        try:
            judge = await self._ensure_judge()
        except Exception as error:
            await judge_output.update(f"# Judge\n\nUnable to generate judge: `{error}`")
            self.notify(f"Unable to generate judge: {error}", severity="error")
            return

        path = solution_path(judge)
        ensure_solution_file(path, judge.initial_code_snippet)

        await judge_output.update(f"# Judge\n\nRunning {suite} tests...")

        try:
            solution_code = read_solution_file(path)
            run = await asyncio.to_thread(
                run_solution_suite,
                suite,
                solution_code,
                judge,
            )
        except Exception as error:
            await judge_output.update(f"# Judge\n\nUnable to run tests: `{error}`")
            self.notify(f"Unable to run tests: {error}", severity="error")
            return

        await judge_output.update(format_judge_run(suite, run))
        await self._update_solution_preview(solution_code, path)

    async def fix_solution(self, suite: JudgeSuite) -> None:
        if not self.current_problem_markdown:
            self.notify("Problem is still loading.", severity="warning")
            return

        judge_output = self.query_one(self.JUDGE_SELECTOR, Markdown)
        await judge_output.update(f"# Judge\n\nPreparing to fix with {suite} tests...")

        try:
            judge = await self._ensure_judge()
            path = solution_path(judge)
            ensure_solution_file(path, judge.initial_code_snippet)
            solution_code = read_solution_file(path)
        except Exception as error:
            await judge_output.update(f"# Judge\n\nUnable to prepare fix: `{error}`")
            self.notify(f"Unable to prepare fix: {error}", severity="error")
            return

        await judge_output.update(f"# Judge\n\nRunning {suite} tests before fixing...")
        try:
            run = await asyncio.to_thread(
                run_solution_suite,
                suite,
                solution_code,
                judge,
            )
            judge_report = format_judge_run(suite, run)
        except Exception as error:
            judge_report = format_judge_error(suite, error)

        await judge_output.update(f"{judge_report}\n\nGenerating fix...")

        try:
            agent = self.get_agent()
            agent.set_solution_context(
                solution_code=solution_code,
                judge_output=judge_report,
            )
            fixed_code = await asyncio.to_thread(agent.fix_solution, suite)
        except Exception as error:
            await judge_output.update(f"# Judge\n\nUnable to generate fix: `{error}`")
            self.notify(f"Unable to generate fix: {error}", severity="error")
            return

        if "class Solution" not in fixed_code:
            await judge_output.update(
                "# Judge\n\nGenerated fix did not include a `Solution` class."
            )
            self.notify(
                "Generated fix did not include a Solution class.", severity="error"
            )
            return

        try:
            write_solution_file(path, fixed_code)
        except OSError as error:
            await judge_output.update(
                f"# Judge\n\nUnable to write fixed solution: `{error}`"
            )
            self.notify(f"Unable to write fixed solution: {error}", severity="error")
            return

        await self._update_solution_preview(fixed_code, path)
        await judge_output.update(
            f"# Judge\n\nFixed solution written to `{path}`. Re-running {suite} tests..."
        )

        try:
            fixed_run = await asyncio.to_thread(
                run_solution_suite,
                suite,
                fixed_code,
                judge,
            )
        except Exception as error:
            await judge_output.update(
                "# Judge\n\nFixed solution was written, but tests errored:\n\n"
                "```text\n"
                f"{type(error).__name__}: {error}\n"
                "```"
            )
            self.notify("Fix written, but tests errored.", severity="warning")
            return

        await judge_output.update(format_judge_run(suite, fixed_run))
        if fixed_run.passed:
            self.notify("Solution fixed.")
        else:
            self.notify("Fix written, but tests still fail.", severity="warning")

    async def open_solution_editor(self) -> None:
        if not self.current_problem_markdown:
            self.notify("Problem is still loading.", severity="warning")
            return

        judge_output = self.query_one(self.JUDGE_SELECTOR, Markdown)
        await judge_output.update("# Judge\n\nPreparing editor...")

        try:
            judge = await self._ensure_judge()
            path = solution_path(judge)
            ensure_solution_file(path, judge.initial_code_snippet)
            editor = editor_command(path)
        except Exception as error:
            await judge_output.update(f"# Judge\n\nUnable to prepare editor: `{error}`")
            self.notify(f"Unable to prepare editor: {error}", severity="error")
            return

        try:
            with self.suspend():
                await asyncio.to_thread(subprocess.run, editor, check=True)
        except FileNotFoundError:
            await judge_output.update(
                f"# Judge\n\nEditor command not found: `{editor[0]}`"
            )
            self.notify(f"Editor command not found: {editor[0]}", severity="error")
            return
        except subprocess.CalledProcessError as error:
            await judge_output.update(
                f"# Judge\n\nEditor exited with status `{error.returncode}`."
            )
            self.notify(
                f"Editor exited with status {error.returncode}.",
                severity="error",
            )
            return

        try:
            solution_code = read_solution_file(path)
        except OSError as error:
            await judge_output.update(
                f"# Judge\n\nUnable to read edited code: `{error}`"
            )
            self.notify(f"Unable to read edited code: {error}", severity="error")
            return

        await self._update_solution_preview(solution_code, path)
        await judge_output.update(
            f"# Judge\n\nLoaded edited solution from `{path}`. Run Testfire or Full."
        )

    async def prepare_judge(self) -> None:
        if not self.current_problem_markdown:
            return

        judge_output = self.query_one(self.JUDGE_SELECTOR, Markdown)

        try:
            judge = await self._ensure_judge()
        except Exception as error:
            await judge_output.update(f"# Judge\n\nUnable to generate judge: `{error}`")
            self.notify(f"Unable to generate judge: {error}", severity="error")
            return

        path = solution_path(judge)
        ensure_solution_file(path, judge.initial_code_snippet)

        try:
            solution_code = read_solution_file(path)
        except OSError as error:
            await judge_output.update(f"# Judge\n\nUnable to read solution: `{error}`")
            self.notify(f"Unable to read solution: {error}", severity="error")
            return

        await self._update_solution_preview(solution_code, path)
        await judge_output.update(
            "# Judge\n\nJudge ready. Use Edit to change the external solution file, "
            "then run Testfire or Full."
        )

    async def _ensure_judge(self) -> GeneratedJudge:
        if self.current_judge is None:
            model_slug = os.getenv("LETSCODE_AGENT_MODEL", self.DEFAULT_AGENT_MODEL)
            self.current_judge = await asyncio.to_thread(
                generate_judge,
                self.current_problem_markdown,
                model_slug,
            )

        return self.current_judge

    async def _update_solution_preview(
        self,
        solution_code: str,
        path: Path | None = None,
    ) -> None:
        label = f"`{path}`" if path is not None else "No solution file loaded."
        preview = solution_code.strip() or "# No solution loaded."
        await self.query_one(self.SOLUTION_PREVIEW_SELECTOR, Markdown).update(
            f"{label}\n\n```python\n{preview}\n```"
        )

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        """Handle user input."""
        input_widget = self.query_one(self.INPUT_SELECTOR, PromptInput)
        message = event.value.strip()

        if not message:
            return

        input_widget.clear()
        input_widget.focus()

        if message.startswith("/") and self.handle_slash_command(message):
            return


def main() -> None:
    load_dotenv()
    app = LetsCodeApp()
    app.run()


if __name__ == "__main__":
    main()
