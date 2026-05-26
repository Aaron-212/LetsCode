from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, TypeGuard

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from platformdirs import PlatformDirs

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
matches the problem.

testfire_code must define exactly this function:
def testfire(Solution): ...

full_test_code must define exactly this function:
def full_test_suite(Solution): ...

Both test functions receive the candidate Solution class, instantiate it as
needed, and use plain assert statements. testfire must cover only examples from
the problem statement. full_test_suite must include those examples plus broader
edge cases derived from the constraints. Do not use pytest, unittest, file I/O,
network access, randomness, sleeps, subprocesses, or shell commands.
"""


def default_data_path() -> Path:
    return PlatformDirs("MyApp", "MyCompany").user_data_path / "judge"


def generate_judge(
    problem: str,
    model_slug: str = "openai:gpt-5.4-mini",
    data_path: Path | None = None,
    overwrite: bool = False,
) -> GeneratedJudge:
    problem_hash = hash_problem(problem)
    existing = load_generated_judge(problem_hash, data_path)
    if existing is not None and not overwrite:
        return existing

    model = ChatOpenAI(model=model_slug)
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
    exec(code, namespace.__dict__)
    return namespace


def _run_generated_test(
    suite: str,
    test_code: str,
    function_name: str,
    solution_class: type,
) -> TestResult:
    try:
        test_namespace = _load_code(test_code, f"generated_{suite}_tests")
        test_function = getattr(test_namespace, function_name)
        test_function(solution_class)
    except AssertionError as error:
        return TestResult(
            suite=suite,
            passed=False,
            message=str(error) or "Assertion failed.",
        )
    except Exception:
        return TestResult(
            suite=suite,
            passed=False,
            error=_format_current_error(),
        )

    return TestResult(suite=suite, passed=True)


def _format_current_error() -> str:
    error_type, error, _traceback = __import__("sys").exc_info()
    if error_type is None or error is None:
        return "Unknown test failure."

    return f"{error_type.__name__}: {error}"
