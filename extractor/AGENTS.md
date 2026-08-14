# Extractor Instructions

## Routing — read only what the task needs, when it needs it

### This context

- How this module is written, and the idioms it rejects → CODING_STANDARDS.md
- Extractor vocabulary → docs/agents/domain.md
- Extractor decisions → docs/adr/

## Verification

Run from `extractor/`. There is no task runner and there must not be one.

- MUST run `uv run ruff format .` before verification.
- MUST run `uv run ruff check .`.
- MUST run `uv run mypy`.
- MUST run `uv run pytest`.
- MUST keep the default test run offline: it passes with no API key present.
- MUST mark any test that calls a provider `live`; those run only under `uv run pytest -m live`, which needs a real key and costs money.
- MUST keep development tools in the `dev` dependency group, never in `project.dependencies`.
- MUST commit `uv.lock`.
- MUST NOT add a task runner, a `setup.cfg`, or a second tool-configuration file — `pyproject.toml` holds all of it.
