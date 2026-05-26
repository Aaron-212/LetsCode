from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from platformdirs import PlatformDirs


@dataclass(frozen=True)
class GeneratedHint:
    hint: str
    prompt: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass(frozen=True)
class StoredProblem:
    question_id: int
    problem: str
    fetched_hints: list[str]
    generated_hints: list[GeneratedHint]
    updated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class ProblemStore:
    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or PlatformDirs(
            "MyApp", "MyCompany"
        ).user_data_path
        self.problems_path = self.data_path / "problems"
        self.state_path = self.data_path / "state.json"

    def load_last_question_id(self) -> int | None:
        if not self.state_path.exists():
            return None

        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        try:
            question_id = int(payload["last_question_id"])
        except (KeyError, TypeError, ValueError):
            return None

        return question_id

    def save_last_question_id(self, question_id: int) -> None:
        self.data_path.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(
                {
                    "last_question_id": question_id,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_path.replace(self.state_path)

    def load_problem(self, question_id: int) -> StoredProblem | None:
        path = self._problem_path(question_id)

        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        return self._decode_problem(payload)

    def next_custom_question_id(self) -> int:
        if not self.problems_path.exists():
            return -1

        custom_question_ids = []
        for path in self.problems_path.glob("*.json"):
            try:
                question_id = int(path.stem)
            except ValueError:
                continue

            if question_id < 0:
                custom_question_ids.append(question_id)

        return min(custom_question_ids, default=0) - 1

    def save_fetched_problem(
        self,
        question_id: int,
        problem: str,
        fetched_hints: list[str],
    ) -> StoredProblem:
        existing = self.load_problem(question_id)
        generated_hints = existing.generated_hints if existing else []
        stored_problem = StoredProblem(
            question_id=question_id,
            problem=problem,
            fetched_hints=fetched_hints,
            generated_hints=generated_hints,
        )
        self._save_problem(stored_problem)
        return stored_problem

    def save_generated_problem(
        self,
        question_id: int,
        problem: str,
        fetched_hints: list[str] | None = None,
    ) -> StoredProblem:
        existing = self.load_problem(question_id)
        generated_hints = existing.generated_hints if existing else []
        stored_problem = StoredProblem(
            question_id=question_id,
            problem=problem,
            fetched_hints=fetched_hints or [],
            generated_hints=generated_hints,
        )
        self._save_problem(stored_problem)
        return stored_problem

    def add_generated_hint(
        self,
        question_id: int,
        problem: str,
        fetched_hints: list[str],
        hint: str,
        prompt: str | None,
    ) -> StoredProblem:
        existing = self.load_problem(question_id)
        generated_hints = list(existing.generated_hints) if existing else []
        generated_hints.append(GeneratedHint(hint=hint, prompt=prompt))

        stored_problem = StoredProblem(
            question_id=question_id,
            problem=existing.problem if existing else problem,
            fetched_hints=existing.fetched_hints if existing else fetched_hints,
            generated_hints=generated_hints,
        )
        self._save_problem(stored_problem)
        return stored_problem

    def _problem_path(self, question_id: int) -> Path:
        return self.problems_path / f"{question_id}.json"

    def _save_problem(self, stored_problem: StoredProblem) -> None:
        self.problems_path.mkdir(parents=True, exist_ok=True)
        path = self._problem_path(stored_problem.question_id)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(self._encode_problem(stored_problem), indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _encode_problem(self, stored_problem: StoredProblem) -> dict[str, Any]:
        return {
            "question_id": stored_problem.question_id,
            "problem": stored_problem.problem,
            "fetched_hints": stored_problem.fetched_hints,
            "generated_hints": [
                {
                    "hint": generated_hint.hint,
                    "prompt": generated_hint.prompt,
                    "created_at": generated_hint.created_at,
                }
                for generated_hint in stored_problem.generated_hints
            ],
            "updated_at": stored_problem.updated_at,
        }

    def _decode_problem(self, payload: Any) -> StoredProblem | None:
        if not isinstance(payload, dict):
            return None

        try:
            question_id = int(payload["question_id"])
            problem = str(payload["problem"])
            fetched_hints = [
                str(hint) for hint in payload.get("fetched_hints", [])
            ]
            generated_hints = []
            for hint_payload in payload.get("generated_hints", []):
                if not isinstance(hint_payload, dict) or "hint" not in hint_payload:
                    continue

                prompt = hint_payload.get("prompt")
                generated_hints.append(
                    GeneratedHint(
                        hint=str(hint_payload["hint"]),
                        prompt=str(prompt) if prompt is not None else None,
                        created_at=str(
                            hint_payload.get(
                                "created_at", datetime.now(UTC).isoformat()
                            )
                        ),
                    )
                )
            updated_at = str(payload.get("updated_at", ""))
        except (KeyError, TypeError, ValueError):
            return None

        return StoredProblem(
            question_id=question_id,
            problem=problem,
            fetched_hints=fetched_hints,
            generated_hints=generated_hints,
            updated_at=updated_at,
        )
