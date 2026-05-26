# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 package using a `src/` layout. Application code lives in `src/letscode/`. The CLI/Textual entry point is `src/letscode/main.py`, with supporting modules for agent orchestration (`agent.py`), LeetCode fetching (`leetcode.py`), judging (`judge.py`), Markdown conversion (`markdown.py`), storage (`storage.py`), and shared constants (`constants.py`). Pydantic models are under `src/letscode/model/`.

There is no dedicated `tests/` directory yet. Add tests under `tests/` mirroring package modules, for example `tests/test_markdown.py` for `src/letscode/markdown.py`.

## Build, Test, and Development Commands

- `uv sync` installs runtime and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run letscode` runs the installed console script (`letscode.main:main`).
- `uv run python -m letscode.main` runs the app module directly during development.
- `uv build` builds the distributable package using `uv_build`.
- `uv run textual console` starts the Textual dev console when debugging the TUI.

No test runner is configured yet. If adding tests, prefer `pytest` and document the resulting command, typically `uv run pytest`.

## Coding Style & Naming Conventions

Follow standard Python style with 4-space indentation, type hints for public functions, and clear dataclass or Pydantic models for structured data. Use `snake_case` for functions, variables, modules, and command helpers; use `PascalCase` for classes such as `LetsCodeApp` and `LeetCodeQuestion`.

Keep imports grouped as standard library, third-party, then local imports. Preserve the existing explicit constants style for selector IDs, command names, and defaults in `LetsCodeApp`.

## Testing Guidelines

Place tests in `tests/` and name files `test_*.py`. Focus unit tests on pure behavior first: Markdown conversion, model validation, storage paths, and judge helpers. For networked LeetCode and LLM interactions, mock HTTP and model clients instead of depending on live services. Add regression tests for parsing, fallback, and error-handling changes.

## Commit & Pull Request Guidelines

The current history uses very short subjects such as `Init` and `update`. Keep commits concise but more descriptive going forward, for example `Add cached problem storage` or `Fix judge output rendering`.

Pull requests should include a short summary, the reason for the change, commands run for verification, and screenshots or terminal captures for TUI changes when useful. Link related issues if available, and call out any required environment variables or network access.

## Security & Configuration Tips

The project uses `python-dotenv`; keep secrets in local `.env` files and do not commit them. Treat LeetCode credentials, API keys, generated solutions, and cached problem data as user-local unless explicitly intended for version control.
