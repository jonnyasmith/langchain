# extractor

CLI tool that turns messy raw text into a strictly typed JSON object.

Feed it a scraped Terms of Service page or a cluttered email thread. A `ChatPromptTemplate` frames the task, the model is bound to a Pydantic schema with `.with_structured_output()`, and the result comes back as a validated object — no string parsing, no regex salvage of malformed JSON.

## Features

- Strict, provider-enforced Pydantic structured output
- Named extraction schemas; `tos` ships with the tool
- Validated JSON on stdout and diagnostics on stderr
- Distinct validation, empty-extraction, refusal, provider-failure, and rejected-request outcomes
- File and stdin input with a 100,000-character safety limit
- Provider selection, model override, and provider-neutral reasoning control

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

The default test run is offline and passes with no API key. The test marked `live` is
deselected unless explicitly selected:

```bash
uv run pytest -m live
```

This makes one real OpenAI provider call and therefore costs money. Set an
`OPENAI_API_KEY` with access to the configured model in the environment or in
`extractor/.env` before running it. Without a key, or when the provider cannot serve
the request, the test skips with a warning naming the unchecked strict-schema contract.
A provider-rejected request still fails the test because it means the request itself is invalid.

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

The default provider is `openai`, the default model is `gpt-5-nano`, and reasoning defaults to
`medium`. Use `--provider PROVIDER`, `--model MODEL_ID`, and
`--reasoning off|low|medium|high` to choose them. Use `--debug` to dump the raw model message to
stderr. Provider calls use a 60-second request timeout and two SDK retries.

Exit statuses are:

| Status | Meaning |
| ---: | --- |
| 0 | Successful extraction |
| 1 | Bad invocation, unknown provider/reasoning/schema, input, configuration, oversize-document, or unexpected failure |
| 2 | Schema validation failure |
| 3 | Empty extraction |
| 4 | Provider refusal |
| 5 | Provider failure, such as credentials, quota, rate limit, server, network, or timeout |
| 6 | Provider-rejected request: HTTP 400, 404, or 422 |

An unknown model id is a provider-rejected request and exits 6; before this outcome was
introduced, it fell through to the generic exit 1.
