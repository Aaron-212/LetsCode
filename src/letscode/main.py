import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.message import Message
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Markdown,
    TabbedContent,
    TabPane,
    TextArea,
)

from .agent import LetsCodeAgent
from .leetcode import LeetCodeFetcher
from .markdown import html_to_markdown


@dataclass(frozen=True)
class ProblemMarkdown:
    problem: str
    hints: list[str]


async def load_problem_markdown(question_id: int = 1) -> tuple[str, list[str]]:
    problem_markdown = await fetch_problem_markdown(question_id)
    return problem_markdown.problem, problem_markdown.hints


async def fetch_problem_markdown(question_id: int = 1) -> ProblemMarkdown:
    question = await LeetCodeFetcher().get_problem(question_id)

    if question is None:
        fallback_markdown = f"# Problem {question_id}\n\nUnable to load this problem."
        return ProblemMarkdown(problem=fallback_markdown, hints=[fallback_markdown])

    content = html_to_markdown(question.content)
    hints = [html_to_markdown(hint) for hint in question.hints]
    problem = (
        f"# {question.questionId}. {question.title}\n\n"
        f"Difficulty: {question.difficulty}\n\n"
        f"{content}"
    )
    return ProblemMarkdown(problem=problem, hints=hints)


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
    DEFAULT_PROBLEM_ID = 3
    DEFAULT_AGENT_MODEL = "openai:gpt-4.1-mini"

    INPUT_SELECTOR = "#input"
    PROBLEM_SELECTOR = "#problem"
    HINTS_SELECTOR = "#hints"

    LOAD_PROBLEM_COMMAND = "/load_problem"
    HINT_COMMAND = "/hint"

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

    #problem, #hints {
        height: auto;
        background: transparent;
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
            with TabPane("Status", id="status-tab"):
                yield ScrollableContainer(
                    Markdown("# Loading problem..."),
                    id="status",
                )
            with TabPane("Hints", id="hints-tab"):
                yield ScrollableContainer(
                    Markdown("# Loading problem..."),
                    id="hints",
                )
        yield PromptInput(
            language="markdown",
            placeholder="Type your message. Enter submits, Alt+Enter adds a new line.",
            id="input",
            compact=True,
            highlight_cursor_line=False,
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(self.INPUT_SELECTOR, PromptInput).focus()
        self.current_problem_id = self.DEFAULT_PROBLEM_ID
        self.current_problem_markdown = ""
        self.hints_count = 0
        self.agent: LetsCodeAgent | None = None
        self.action_load_problem()

    def action_load_problem(self) -> None:
        self.run_worker(self.load_problem(self.current_problem_id), exclusive=True)

    def handle_slash_command(self, message: str) -> bool:
        command, *args = message.split()

        if command == self.LOAD_PROBLEM_COMMAND:
            return self._handle_load_problem_command(args)

        if command == self.HINT_COMMAND:
            return self._handle_hint_command(args)

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

    async def load_problem(self, question_id: int) -> None:
        self.current_problem_id = question_id
        self.agent = None

        problem_markdown, hints_markdown = await load_problem_markdown(question_id)
        self.current_problem_markdown = problem_markdown
        self.hints_count = len(hints_markdown)

        await self.query_one(self.PROBLEM_SELECTOR, Markdown).update(problem_markdown)
        await self._render_hints(hints_markdown)

    async def _render_hints(self, hints_markdown: list[str]) -> None:
        hints_container = self.query_one(self.HINTS_SELECTOR, ScrollableContainer)
        await hints_container.remove_children()

        if not hints_markdown:
            await hints_container.mount(Markdown("No hints available."))
            return

        for index, hint_markdown in enumerate(hints_markdown, start=1):
            await self._mount_hint(
                hints_container,
                hint_markdown,
                title=f"Hint {index}",
                collapsed=True,
            )

    async def generate_hint(self, prompt: str | None) -> None:
        if not self.current_problem_markdown:
            self.notify("Problem is still loading.", severity="warning")
            return

        self.notify("Generating hint...")

        try:
            hint_markdown = await asyncio.to_thread(
                self.get_agent().chat,
                self._build_hint_prompt(prompt),
            )
        except Exception as error:
            self.notify(f"Unable to generate hint: {error}", severity="error")
            return

        self.hints_count += 1
        hints_container = self.query_one(self.HINTS_SELECTOR, ScrollableContainer)

        if self.hints_count == 1:
            await hints_container.remove_children()

        await self._mount_hint(
            hints_container,
            hint_markdown,
            title=f"AI Hint {self.hints_count}",
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

    def _build_hint_prompt(self, prompt: str | None) -> str:
        return (
            "Generate one additional hint for this LeetCode problem. "
            "Do not solve the problem outright or include full code. "
            "Keep the hint concise, progressive, and focused on the next useful insight.\n\n"
            f"{self.current_problem_markdown}"
            f"User prompt: {prompt}\n"
        )

    def get_agent(self) -> LetsCodeAgent:
        if self.agent is None:
            model_slug = os.getenv("LETSCODE_AGENT_MODEL", self.DEFAULT_AGENT_MODEL)
            self.agent = LetsCodeAgent(model_slug)

        return self.agent

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
