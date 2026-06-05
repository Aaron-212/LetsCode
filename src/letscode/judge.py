from __future__ import annotations

import hashlib
import json
import linecache
import reprlib
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType, ModuleType, TracebackType
from typing import Any, Literal, TypeGuard

from langchain_core.messages import HumanMessage, SystemMessage
from platformdirs import PlatformDirs

from .ai_http import OpenAIHttpClients, create_chat_model
from .agent import format_message_content

JudgeSuite = Literal["testfire", "full"]
DEFAULT_JUDGE_SUITE: JudgeSuite = "testfire"
VALID_JUDGE_SUITES: tuple[JudgeSuite, ...] = ("testfire", "full")


@dataclass(frozen=True)
class GeneratedJudge:
    problem_hash: str
    initial_code_snippet: str
    testfire_code: str
    full_test_code: str
    model_slug: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class TestResult:
    suite: str
    passed: bool
    message: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class JudgeRun:
    passed: bool
    results: list[TestResult]


JUDGE_SYSTEM_PROMPT = """\
You generate Python starter code and tests for LeetCode-style problems.
Return only strict JSON with these string fields:
- initial_code_snippet
- testfire_code
- full_test_code

initial_code_snippet must define class Solution with the method signature that
matches the problem, with proper type annotations.

testfire_code must define exactly this function:
def testfire(Solution): ...

full_test_code must define exactly this function:
def full_test_suite(Solution): ...

Both test functions receive the candidate Solution class, instantiate it as
needed, and use plain assert statements. testfire must cover only examples from
the problem statement. full_test_suite must include those examples plus broader
edge cases derived from the constraints. Do not use pytest, unittest, file I/O,
network access, sleeps, subprocesses, or shell commands.
Using random library is allowed.

READ AND COMPREHEND THE PROBLEM THOROUGHLY TO AVOID WRITING WRONG TESTS.
"""


def default_data_path() -> Path:
    return PlatformDirs("MyApp", "MyCompany").user_data_path / "judge"


def generate_judge(
    problem: str,
    model_slug: str = "openai:gpt-5.4-mini",
    data_path: Path | None = None,
    overwrite: bool = False,
    http_clients: OpenAIHttpClients | None = None,
) -> GeneratedJudge:
    problem_hash = hash_problem(problem)
    existing = load_generated_judge(problem_hash, data_path)
    if existing is not None and not overwrite:
        return existing

    model = create_chat_model(model_slug, http_clients=http_clients)
    response = model.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=problem),
        ]
    )
    payload = _parse_generated_payload(format_message_content(response.content))
    judge = GeneratedJudge(
        problem_hash=problem_hash,
        initial_code_snippet=str(payload["initial_code_snippet"]).strip() + "\n",
        testfire_code=str(payload["testfire_code"]).strip() + "\n",
        full_test_code=str(payload["full_test_code"]).strip() + "\n",
        model_slug=model_slug,
    )
    persist_generated_judge(judge, data_path)
    return judge


def load_generated_judge(
    problem_hash: str,
    data_path: Path | None = None,
) -> GeneratedJudge | None:
    path = _judge_path(problem_hash, data_path)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    try:
        return GeneratedJudge(
            problem_hash=str(payload["problem_hash"]),
            initial_code_snippet=str(payload["initial_code_snippet"]),
            testfire_code=str(payload["testfire_code"]),
            full_test_code=str(payload["full_test_code"]),
            model_slug=str(payload["model_slug"]),
            created_at=str(payload["created_at"]),
        )
    except KeyError:
        return None


def persist_generated_judge(
    judge: GeneratedJudge,
    data_path: Path | None = None,
) -> Path:
    path = _judge_path(judge.problem_hash, data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(asdict(judge), indent=2), encoding="utf-8")
    temp_path.replace(path)
    return path


def run_testfire(solution_code: str, judge: GeneratedJudge) -> JudgeRun:
    return run_generated_tests(solution_code, judge, suites=("testfire",))


def run_full_suite(solution_code: str, judge: GeneratedJudge) -> JudgeRun:
    return run_generated_tests(solution_code, judge, suites=("full",))


def is_judge_suite(value: str) -> TypeGuard[JudgeSuite]:
    return value in VALID_JUDGE_SUITES


def run_solution_suite(
    suite: JudgeSuite,
    solution_code: str,
    judge: GeneratedJudge,
) -> JudgeRun:
    if suite == "testfire":
        return run_testfire(solution_code, judge)
    if suite == "full":
        return run_full_suite(solution_code, judge)

    raise ValueError(f"Unknown judge suite: {suite}")


def format_judge_run(suite: JudgeSuite, run: JudgeRun) -> str:
    lines = [
        "# Judge",
        "",
        f"{suite.title()} tests: {'passed' if run.passed else 'failed'}",
    ]

    for result in run.results:
        state = "passed" if result.passed else "failed"
        lines.extend(["", f"## {result.suite}: {state}"])
        if result.message:
            lines.extend(["", result.message])
        if result.error:
            lines.extend(["", "```text", result.error.strip(), "```"])

    return "\n".join(lines)


def format_judge_error(suite: JudgeSuite, error: Exception) -> str:
    return (
        "# Judge\n\n"
        f"{suite.title()} tests: error\n\n"
        "```text\n"
        f"{type(error).__name__}: {error}\n"
        "```"
    )


def run_generated_tests(
    solution_code: str,
    judge: GeneratedJudge,
    suites: tuple[str, ...] = ("testfire", "full"),
) -> JudgeRun:
    solution_namespace = _load_code(solution_code, "generated_solution")
    if not hasattr(solution_namespace, "Solution"):
        raise ValueError("solution_code must define a Solution class")

    results: list[TestResult] = []
    if "testfire" in suites:
        results.append(
            _run_generated_test(
                "testfire",
                judge.testfire_code,
                "testfire",
                solution_namespace.Solution,
            )
        )
    if "full" in suites:
        results.append(
            _run_generated_test(
                "full",
                judge.full_test_code,
                "full_test_suite",
                solution_namespace.Solution,
            )
        )

    return JudgeRun(
        passed=all(result.passed for result in results),
        results=results,
    )


def hash_problem(problem: str) -> str:
    return hashlib.sha256(problem.encode("utf-8")).hexdigest()[:16]


def _judge_path(problem_hash: str, data_path: Path | None) -> Path:
    return (data_path or default_data_path()) / f"{problem_hash}.json"


def _parse_generated_payload(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("judge generator must return a JSON object")

    required = {"initial_code_snippet", "testfire_code", "full_test_code"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"judge generator missing fields: {sorted(missing)}")

    return payload


def _load_code(code: str, module_name: str) -> ModuleType:
    namespace = ModuleType(module_name)
    filename = f"<{module_name}>"
    lines = [line if line.endswith("\n") else f"{line}\n" for line in code.splitlines()]
    linecache.cache[filename] = (len(code), None, lines, filename)
    exec(compile(code, filename, "exec"), namespace.__dict__)
    return namespace


def _run_generated_test(
    suite: str,
    test_code: str,
    function_name: str,
    solution_class: type,
) -> TestResult:
    try:
        test_module_name = f"generated_{suite}_tests"
        test_namespace = _load_code(test_code, test_module_name)
        test_function = getattr(test_namespace, function_name)
        test_function(solution_class)
    except AssertionError as error:
        message, details = _format_assertion_failure(
            error,
            module_name=test_module_name,
            function_name=function_name,
        )
        return TestResult(
            suite=suite,
            passed=False,
            message=message,
            error=details,
        )
    except Exception:
        return TestResult(
            suite=suite,
            passed=False,
            error=_format_current_error(),
        )

    return TestResult(suite=suite, passed=True)


def _format_current_error() -> str:
    error_type, error, _traceback = sys.exc_info()
    if error_type is None or error is None:
        return "Unknown test failure."

    return f"{error_type.__name__}: {error}"


def _format_assertion_failure(
    error: AssertionError,
    module_name: str,
    function_name: str,
) -> tuple[str, str]:
    frame = _find_traceback_frame(error.__traceback__, f"<{module_name}>")
    assertion_message = str(error).strip()
    if frame is None:
        message = assertion_message or "Assertion failed."
        details = "".join(traceback.format_exception_only(type(error), error)).strip()
        return message, details

    filename = frame.f_code.co_filename
    line_number = frame.f_lineno
    source_line = linecache.getline(filename, line_number).strip()
    if source_line:
        message = (
            f"Assertion failed in {function_name} at line {line_number}: {source_line}"
        )
    else:
        message = f"Assertion failed in {function_name} at line {line_number}."

    details = [
        f"Location: {filename}:{line_number}",
    ]
    if assertion_message:
        details.append(f"Assertion message: {assertion_message}")
    if source_line:
        details.extend(["Failing assertion:", f"  {source_line}"])

    local_values = _format_frame_locals(frame)
    if local_values:
        details.extend(["Locals:", *local_values])

    return message, "\n".join(details)


def _find_traceback_frame(
    current_traceback: TracebackType | None,
    filename: str,
) -> FrameType | None:
    frame: FrameType | None = None
    while current_traceback is not None:
        if current_traceback.tb_frame.f_code.co_filename == filename:
            frame = current_traceback.tb_frame
        current_traceback = current_traceback.tb_next

    return frame


def _format_frame_locals(frame: FrameType) -> list[str]:
    formatter = reprlib.Repr()
    formatter.maxdict = 8
    formatter.maxlist = 12
    formatter.maxtuple = 12
    formatter.maxset = 12
    formatter.maxstring = 200
    formatter.maxother = 200

    lines: list[str] = []
    for name, value in frame.f_locals.items():
        if len(lines) >= 12:
            lines.append("  ...")
            break
        if name.startswith("__") or name == "Solution":
            continue
        if isinstance(value, ModuleType | type) or callable(value):
            continue

        rendered = formatter.repr(value)
        if rendered.startswith("<") and rendered.endswith(">"):
            continue
        lines.append(f"  {name} = {rendered}")

    return lines
