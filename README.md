# LetsCode

LetsCode is a Textual CLI for loading LeetCode problems, generating hints, and
running generated Python judge suites against local solution files.

## Commands

- `/load_problem [question_id]` loads a LeetCode problem.
- `/new [PROMPT]` asks the agent to create a new local coding problem.
- `/hint [HINT_PROMPT]` generates a progressive hint.
- `/fix [testfire|full]` fixes the current solution against a judge suite.
- `/reset [question_id]` resets the solution file.

## Development

Install dependencies:

```bash
uv sync
```

Run the app:

```bash
uv run letscode
```

Run the unit tests:

```bash
uv run python -m unittest discover
```
