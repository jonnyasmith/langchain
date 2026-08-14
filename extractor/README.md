# extractor

CLI tool that turns messy raw text into a strictly typed JSON object.

Feed it a scraped Terms of Service page or a cluttered email thread. A `ChatPromptTemplate` frames the task, the model is bound to a Pydantic schema with `.with_structured_output()`, and the result comes back as a validated object — no string parsing, no regex salvage of malformed JSON.

## Features

- Strict, provider-enforced Pydantic structured output
- Named extraction schemas; `tos` ships with the tool
- Validated JSON on stdout and diagnostics on stderr
- Distinct validation, empty-extraction, and refusal outcomes
- File and stdin input with a 100,000-character safety limit
- App-local OpenAI configuration with a model override

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

Those tests need a funded key with access to the model under test, and they cost money. With
no key they skip rather than fail, so read the count: `2 passed` means the provider really
enforced the schema, while `2 skipped` means nothing was checked.

## Run

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, then extract from a file:

```bash
uv run python -m extractor --schema tos terms.html
```

Read the source from stdin:

```bash
uv run python -m extractor --schema tos - < terms.html
```

List the available named schemas:

```bash
uv run python -m extractor --list-schemas
```

The default model is `gpt-5-nano`. Use `--model MODEL_ID` to override it and `--debug`
to dump the raw model message to stderr. Successful extraction exits 0. Validation
failure exits 2, empty extraction exits 3, provider refusal exits 4, and input,
configuration, oversize, or unexpected failures exit 1.
