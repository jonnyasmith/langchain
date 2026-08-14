# extractor

CLI tool that turns messy raw text into a strictly typed JSON object.

Feed it a scraped Terms of Service page or a cluttered email thread. A `ChatPromptTemplate` frames the task, the model is bound to a Pydantic schema with `.with_structured_output()`, and the result comes back as a validated object — no string parsing, no regex salvage of malformed JSON.

## Features (planned)

- LCEL chain: prompt | model | structured output
- Pydantic schemas as the extraction contract
- Validation errors surfaced, not swallowed
- Reads from a file or stdin

## Status

Not implemented.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management
- An OpenAI API key in `.env` — copy `.env.example` and fill it in

## Install

```bash
uv sync
```

## Verify

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
```

The default test run is offline and passes with no API key. Tests marked `live` make a real
provider call and are deselected unless you ask for them:

```bash
uv run pytest -m live
```

## Run

```bash
uv run python -m extractor --schema tos FILE
```
