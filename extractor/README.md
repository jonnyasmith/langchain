# extractor

CLI tool that turns messy raw text into a strictly typed JSON object.

Feed it a scraped Terms of Service page or a cluttered email thread. A `ChatPromptTemplate` frames the task, the model is bound to a Pydantic schema with `.with_structured_output()`, and the result comes back as a validated object — no string parsing, no regex salvage of malformed JSON.

## Features

- Strict, provider-enforced Pydantic structured output on OpenAI, Anthropic, and OpenRouter
- Named extraction schemas; `tos` ships with the tool
- Validated JSON on stdout and diagnostics on stderr
- Distinct validation, empty-extraction, refusal, provider-failure, and rejected-request outcomes
- File and stdin input with a 100,000-character safety limit
- Provider selection, per-provider default models, and provider-neutral reasoning control

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management
- One API key in `.env` for the provider you intend to use — copy `.env.example` and fill in
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY`. Each provider reads only its
  own key, so a run against one never requires another's.

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

The default test run is offline and passes with no API key. The tests marked `live` are
deselected unless explicitly selected:

```bash
uv run pytest -m live
```

That is one real call per provider and therefore costs money. Each live test needs its own key
in the environment or in `extractor/.env`, with access to that provider's default model. A test
whose key is absent skips with a warning naming the provider whose enforced-schema contract went
unchecked; so does a provider that cannot serve the request. A provider-rejected request still
fails the test, because it means the request itself is invalid.

## Run

Copy `.env.example` to `.env`, set the key for the provider you want, then extract from a file:

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

The default provider is `openai` and reasoning defaults to `medium`. Use `--provider PROVIDER`,
`--model MODEL_ID`, and `--reasoning off|low|medium|high` to choose them. Use `--debug` to dump
the raw provider message to stderr. Every provider call uses a 60-second request timeout and two
SDK retries.

| `--provider` | Default model | API key | How the schema is enforced |
| --- | --- | --- | --- |
| `openai` | `gpt-5-nano` | `OPENAI_API_KEY` | Native strict JSON schema |
| `anthropic` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` | The JSON-schema structured-output method |
| `openrouter` | `openai/gpt-5-nano` | `OPENROUTER_API_KEY` | Strict JSON schema plus a routing guard, so only an endpoint that can enforce it is selected |

`--model` overrides the default on any provider. The OpenRouter default is deliberately a model
you can also reach directly, so comparing the two paths isolates routing rather than changing two
variables at once. The Anthropic default is deliberately not a Haiku tier: Haiku 4.5 and
Sonnet 4.5 reject the effort parameter server-side, so the default reasoning level would fail on
every run.

`--reasoning` takes the same four words everywhere, but the level is nominal: `medium` on OpenAI
and `medium` on a Claude-backed aggregator endpoint are not the same quantity, because the
aggregator turns an effort into a proportional token budget. On `openrouter` the reasoning setting
is also covered by the routing guard, which requires every parameter sent to be honoured — so a
`--model` that does not advertise reasoning is reported as a rejected request rather than having
the level quietly ignored. The guard cannot be scoped to one parameter, and losing schema
enforcement is the worse trade.

Only `openai` pins temperature to 0. Anthropic rejects a modified temperature while thinking is
on, and under the routing guard temperature is another parameter the default aggregator endpoint
does not advertise, so neither of the other two sends one.

Refusal reporting is asymmetric, so read silence as unknown rather than as consent. OpenAI
reports a refusal, either raised or on the raw message. Anthropic reports it as a stop reason on
the raw message. Through OpenRouter it is reachable but not guaranteed: it rides a response field
the upstream provider may not pass through.

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
