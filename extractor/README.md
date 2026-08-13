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

- Python (version TBD when a `pyproject.toml` is added)
- [uv](https://docs.astral.sh/uv/) for dependency management
- A model provider API key in `.env`

## Install

```bash
uv sync
```

## Test

```bash
uv run pytest
```

## Run

```bash
uv run python -m extractor
```
