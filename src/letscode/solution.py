from __future__ import annotations

import os
import shlex
from pathlib import Path

from .judge import GeneratedJudge, default_data_path


def solution_path(judge: GeneratedJudge, data_path: Path | None = None) -> Path:
    return (data_path or default_data_path()) / "solutions" / f"{judge.problem_hash}.py"


def ensure_solution_file(path: Path, initial_code: str) -> Path:
    if not path.exists():
        write_solution_file(path, initial_code)

    return path


def reset_solution_file(path: Path, initial_code: str) -> Path:
    write_solution_file(path, initial_code)
    return path


def write_solution_file(path: Path, solution_code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(solution_code, encoding="utf-8")


def read_solution_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def editor_command(path: Path, editor: str | None = None) -> list[str]:
    configured_editor = os.getenv("EDITOR") if editor is None else editor
    if not configured_editor:
        raise RuntimeError("Set $EDITOR before opening the code editor")

    command = shlex.split(configured_editor)
    if not command:
        raise RuntimeError("Set $EDITOR before opening the code editor")

    return [*command, str(path)]
