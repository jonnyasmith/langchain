# Extractor architecture

> **Status:** Current implementation
>
> **Verification basis:** `2858bcd`

## 1. Executive summary

The extractor is a one-shot command line tool. You give it one messy document and the name of
a schema. It gives you back one validated JSON object on stdout, or it names exactly why it
could not.

It has four modules and no state. `__main__.py` parses the command line, resolves the selected
provider through the registry, and decides the exit code. `intake.py` turns a path or `-` into a
document string. `schemas.py` holds the Pydantic models that define what can be extracted.
`extraction.py` owns the provider registry and its three adapters — OpenAI, Anthropic, and
OpenRouter: it builds the prompt, binds the schema through each provider's enforced path, makes
the call, and translates everything that can happen into one of six named outcomes.

There is no database and no cache. The only data that outlives a run is what the caller
redirects from stdout. The rule a contributor must not break: an extraction attempt returns a
value from the closed `Extraction` union, never an exception, and `__main__.py` matches that
union exhaustively. `mypy --strict` is what enforces it, so `uv run mypy` is not optional.

## 2. System context

```
   operator
      |
      | argv, stdin, extractor/.env
      v
+-----------------+       enforced structured output       +------------+
|   extractor     | -------------------------------------> | selected   |
|   (one process) | <------------------------------------- | provider   |
+-----------------+       object, refusal, or error         +------------+
      |        |
      |        +--> stderr: diagnostics, optional raw message dump
      +-----------> stdout: one JSON object, or the schema name list
```

Inside the boundary: argument parsing, document intake, the character ceiling, prompt
assembly, the schema binding, outcome classification, and exit codes.

Outside the boundary: provider services, the filesystem the document is read from, and the
`.env` file that supplies provider credentials.

## 3. Architectural invariants

1. **An extraction attempt returns a value, never raises.** Each adapter catches its provider's
   refusal, rejected-request, and family-base exceptions, and returns an outcome for each.
   Specific rejected-request classes are caught before the family base so a new SDK subclass
   degrades to `ProviderFailure` rather than escaping. There are two funnels for three
   providers — OpenAI and OpenRouter share one integration, Anthropic has its own unrelated
   exception tree — and both are exercised offline. See ADR-0004.

2. **Outcomes are a closed union, matched exhaustively.** `Extraction` is a `type` alias over
   six frozen dataclasses. `_report` ends its `match` with `case unreachable:
   assert_never(unreachable)`. Enforced by `mypy --strict`: adding a seventh member fails type
   checking at the match instead of falling through at runtime. See ADR-0002.

3. **A document returned by intake has already cleared the ceiling.** `load_source_document`
   returns `Intake`, and the `str` branch is only reachable after the
   `MAX_DOCUMENT_CHARACTERS` check. Enforced by the return type, not by call ordering in
   `main`.

4. **No provider or framework type crosses into the CLI.** `__main__.py` imports only from
   `extractor.*`. The `PROVIDERS` registry in `extraction.py` is the sole place provider names
   appear as strings, and each entry is a `Provider` carrying that provider's default model and
   port factory; the CLI only looks up the parsed value. `ExtractionPort` remains
   `(str, type[BaseModel]) -> Extraction`.

5. **Every schema field is required and nullable with no default.** Enforced structured output
   demands it. Enforced by a test that reads the generated JSON schema
   (`tests/test_schemas.py:19`), not by the type system. See ADR-0001.

6. **The extracted object is the only thing on stdout.** Every diagnostic goes to stderr,
   including the `--debug` raw message dump. Enforced by `_report` being the sole writer of
   outcome output, and by CLI tests that assert stdout is empty on failure.

7. **Exit code numbers are published.** `README.md` documents them and
   `tests/test_cli.py:361` pins each member's integer. Renumbering is a breaking change.

8. **Every registered adapter enforces the schema provider-side.** An adapter must bind through
   its provider's enforced structured-output path, never through function calling, and fail
   rather than degrade when enforcement cannot be guaranteed. On an aggregator that means
   sending the routing guard, so an endpoint that cannot enforce the schema is never selected.
   Enforced by ADR-0001, adapter binding tests, the emitted-request test, and review.

## 4. Components and dependencies

Dependencies point one way: `__main__` depends on `intake`, `schemas`, and `extraction`.
Nothing depends on `__main__`. `intake` and `schemas` depend on nothing in the module.

**`__main__.py`** owns argument parsing, manual provider and reasoning validation, the
`ExitCode` enum, the outcome-to-exit-code mapping, and the top-level error net. It does not own
how a document is read, what the schemas are, or how a provider is called. It never imports
LangChain or a provider SDK.

**`intake.py`** owns reading stdin or a UTF-8 file, the 100,000-character ceiling, and the
classification of a refusal into `UnreadableSource` or `OversizeDocument`. It does not own the
wording of either report: failures carry their facts and `__main__` renders them, the same
split `extraction.py` and `_report` use. It is deliberately not a seam.

**`schemas.py`** owns the named schema registry (`SCHEMAS`, currently one entry, `tos`) and the
field descriptions, which are prompt surface as well as validation. It does not own schema
selection or error reporting.

**`extraction.py`** owns everything provider-shaped: the `PROVIDERS` registry and its `Provider`
entries, frozen `PortSettings`, `_load_env_file` and the per-provider key check, the three
adapters and their model construction, the shared `ChatPromptTemplate`, enforced
structured-output binding, debug dumps, both exception-to-outcome funnels, and the six outcome
dataclasses. It also declares `ExtractionPort` because the union lives here. It does not own exit
codes or any output formatting other than adapter debug dumps.

The three adapters share the prompt, the envelope classification, and — for OpenAI and
OpenRouter — one exception funnel. What differs per provider is the model construction, the
reasoning spelling, the binding arguments, and where a refusal is read from.

The seam between the CLI and providers is `PortFactory = Callable[[PortSettings],
ExtractionPort]`, reached through `Provider.build_port`. `PortSettings` carries the model id,
provider-neutral reasoning level, and optional debug stream. `main` receives a mapping of
provider names to `Provider` records, defaulting to `PROVIDERS`; CLI tests substitute the
mapping without importing a provider or framework type.

## 5. Critical flows

### Successful extraction

1. `main` catches argument-parser exits and returns `FAILURE`. It manually validates
   `--provider` and `--reasoning`, listing valid values and returning `FAILURE` before document
   intake or provider construction on an unknown value. `--list-schemas` then short-circuits:
   it writes sorted names to stdout and returns `OK`.
2. Missing `--schema` or missing input path writes an input error and returns `FAILURE`.
3. An unrecognised schema name writes the valid names and returns `FAILURE`. No provider call
   happens, and no document is read.
4. `load_source_document` reads stdin or the file. Anything that is not a `str` goes to
   `_report_intake`, which writes the diagnostic and returns `FAILURE`.
5. The selected `Provider` supplies its own default model when `--model` is absent, and its
   factory receives one frozen `PortSettings`. Every adapter loads `extractor/.env`, raises
   `ConfigurationError` if that file is unreadable or if *its own* key — `OPENAI_API_KEY`,
   `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` — is absent, then constructs its chat model.
   Construction happens after intake, so an oversize document costs nothing.
6. Reasoning is translated per provider. OpenAI maps `off` to a `none` effort and keeps the
   other three spellings. Anthropic maps `off` to a disabled thinking configuration with no
   effort at all, and each named level to an effort with thinking left adaptive — an effort and
   an explicit thinking configuration are mutually exclusive. OpenRouter sends an effort inside
   a `reasoning` object, `none` included.
7. `extract(document, schema)` binds the schema through the provider's enforced path — strict
   `json_schema` on OpenAI and OpenRouter, `json_schema` on Anthropic, which has no `strict`
   argument because the method *is* the enforcement — invokes `prompt | structured_model`, and
   inspects the `{"raw", "parsed", "parsing_error"}` dict.
8. If `--debug` was passed, the raw message is written to stderr before classification.
9. Classification order, shared by all three: a refusal becomes `Refusal`; any `parsing_error`
   becomes `ValidationFailure`; `parsed is None` becomes `EmptyExtraction`; otherwise
   `Extracted`. Where the refusal is read from is the per-provider part — a refusal error in
   the parsing-error slot on OpenAI and OpenRouter, a `stop_reason` of `refusal` on the raw
   message on Anthropic.
10. `_report` writes `value.model_dump_json()` plus a newline to stdout and returns `OK`.

### Failure before an answer

The `chain.invoke` call is wrapped in one funnel per exception family, not a check per call.

OpenAI and OpenRouter share a funnel: `OpenAIRefusalError` becomes `Refusal`;
`BadRequestError`, `NotFoundError`, and `UnprocessableEntityError` become
`ProviderRejectedRequest`; any remaining `APIError` becomes `ProviderFailure`. Exhausted
aggregator credit has no named subclass and lands there, which is correct.

Anthropic has its own funnel over the identically named but unrelated `anthropic` classes, with
no raised-refusal case because that provider does not raise one.

In both, the specific classes are caught before the family base. Anything else propagates to
`main`'s `except Exception`, which prints "Unexpected error" and returns `FAILURE`.

### Recovery

Two SDK retries and a 60-second timeout are configured on the client. A rate limit that
surfaces as `ProviderFailure` has therefore already exhausted them. There is no retry logic in
this module. A `ProviderFailure` may succeed on re-run; a `ProviderRejectedRequest` will not,
because the request itself is malformed.

## 6. Interfaces and data

**Command line:** `python -m extractor [input] --schema NAME [--provider NAME] [--model ID]
[--reasoning off|low|medium|high] [--list-schemas] [--debug]`. The provider defaults to the
registry's first entry, `openai`; the model defaults to the selected provider's own — `gpt-5-nano`
for OpenAI, `claude-sonnet-5` for Anthropic, `openai/gpt-5-nano` for OpenRouter — and reasoning
defaults to `medium`.

**Exit codes** (published in `README.md`, pinned by test):

| Code | Meaning |
| ---: | --- |
| 0 | Successful extraction, or schema listing |
| 1 | Bad invocation, unknown schema, input failure, oversize document, missing configuration, unexpected error |
| 2 | Validation failure |
| 3 | Empty extraction |
| 4 | Refusal |
| 5 | Provider failure |
| 6 | Provider-rejected request |

**Stdout:** exactly one line of JSON from `model_dump_json()` on success, or the newline-joined
schema names for `--list-schemas`. Nothing else, ever.

**Stderr:** one diagnostic line per failure, prefixed by the outcome name. The `--debug` raw
message dump when enabled.

**Stored data:** none. The only persistent input is `extractor/.env`, which is gitignored;
`.env.example` is the committed template.

**Schema registry:** `SCHEMAS: dict[str, type[BaseModel]]`. Adding a schema means adding a
Pydantic model with every field required, nullable, and described. The registry is closed by
design; the tool does not load user-supplied schemas.

**Compatibility:** exit numbers and the stdout/stderr split are the contract. The outcome
dataclass names are internal but appear in stderr text.

## 7. Security and trust boundaries

There is no identity or authorization inside the tool. It runs with the invoking user's
permissions and reads whatever path they pass.

API keys are the only secrets. There are three — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`OPENROUTER_API_KEY` — and each adapter reads only its own, so running one provider never
requires another's key. They are read from the process environment or `extractor/.env` by
`_load_env_file` inside the selected adapter, never at import time. A `.env` that exists but
cannot be read raises `ConfigurationError` rather than leaking an `OSError` into the top-level
net, and an absent one is not an error at all. An exported variable wins
over the file, so a stale `.env` cannot shadow the shell. A missing key raises
`ConfigurationError` before the model is constructed, so no call is attempted without one. No
key is ever written to stdout or stderr.

Untrusted input is the source document. It is sent to the selected provider verbatim as the human
message; with `--provider openrouter` that provider is an aggregator, which forwards it to
whichever endpoint it selects. There is no sanitisation and no prompt-injection defence: a
document that instructs the model to ignore the system message can influence the extracted
values. Schema enforcement bounds the *shape* of what comes back, not its truthfulness. This is
an accepted property of a single-user CLI, not a mitigated risk.

The character ceiling is a cost and blast-radius guard, not a security control. It fails closed:
an oversize document is refused before the provider is constructed.

## 8. Failure, capacity, and operations

One document, one call, one process. No concurrency primitives and no context threading.
Splitting a document is `rag/`'s concern.

Hard limits: 100,000 characters per document, a 60-second request timeout, two SDK retries, on
every provider. Temperature is 0 on OpenAI only: Anthropic rejects a modified temperature while
thinking is on, and on OpenRouter the routing guard would turn temperature into a routing
constraint the default model's endpoint does not advertise. Reasoning defaults to `medium`, so a
run is no longer as deterministic as the provider allows; `off` is the cheapest and most
repeatable setting available, spelled differently on each of the three.

Partial failure does not exist here: an extraction either yields an object or yields one named
failure. There is no partial result and no resumption.

Deployment is `uv sync` in `extractor/` and `uv run python -m extractor`. There is no server,
no scheduler, and no monitoring. The operator's tools are the exit code, the stderr line, and
`--debug` for the raw model message.

## 9. Verification

Run from `extractor/`: `uv run ruff format .`, `uv run ruff check .`, `uv run mypy`, `uv run
pytest`. There is no task runner and the module forbids adding one.

- `tests/test_schemas.py` proves every field is described and is required-and-nullable, by
  reading the generated JSON schema. This is the only check on invariant 5.
- `tests/test_intake.py` covers stdin, UTF-8 files, missing and unreadable paths, undecodable
  bytes, and both sides of the ceiling boundary.
- `tests/test_extraction.py` covers all three adapters through the substituted chat-model seam:
  classification, each provider's refusal reporting, the debug dump on all three, every
  reasoning translation, each model configuration, the per-provider key check, the binding
  arguments, the aggregator's emitted request against a loopback stub, `.env` handling, and both
  exception funnels.
- `tests/test_cli.py` covers each outcome's exit code and stderr line through a staged port,
  per-provider default models and the `--model` override, the input and configuration paths, and
  pins the exit numbers directly.
- `tests/test_live.py` holds the only tests that let a real provider enforce the schema: one per
  provider, because one test cannot prove three different enforcement mechanisms. They are
  marked `live` and deselected by default (`addopts = "-m 'not live'"`), each skips loudly when
  its own key is absent, and each asserts field *presence* — a field the fixture does not answer
  must come back null — plus values with exactly one faithful rendering. They do not assert
  model wording. See ADR-0003.

The default run passes offline with no API key. `mypy --strict` is load-bearing for invariant
2, not hygiene.

Both adapter exception-to-outcome funnels are verified offline through the substituted
chat-model seam. Nothing in a default run notices when a live test's assertions are weakened,
because those tests are deselected; ADR-0003 states plainly that reading those assertions is the
only mechanism.

## 10. Known limitations

- One schema ships (`tos`). The registry is closed to user-supplied schemas.
- Three providers are registered and the set is closed. The registry and frozen settings seam
  make a fourth enforcing adapter a bounded addition without changing the CLI or extraction port.
- Reasoning does not fail closed the way enforcement does: an aggregated model with no reasoning
  support ignores the setting silently.
- A refusal is only reported where the provider reports one, and through OpenRouter that is not
  guaranteed. See ADR-0004.
- No prompt-injection defence, as described in section 7.
- `ProviderFailure` covers a wide range: credentials, quota, rate limits, server errors,
  network, and timeout all collapse to exit 5 with the provider's rendered text as the only
  discriminator.
- The document ceiling counts characters, not tokens, so it does not bound provider cost
  precisely.

## 11. Source map

| File | What it defines |
| --- | --- |
| `src/extractor/__main__.py` | Entry point, `ExitCode`, argument parsing, `_report`, `_report_intake`, the exhaustive matches |
| `src/extractor/extraction.py` | The `Extraction` union, `ExtractionPort`, `PortFactory`, `Provider`, `PROVIDERS`, the three `build_*_port` adapters, all provider vocabulary |
| `src/extractor/intake.py` | `load_source_document`, the `Intake` and `IntakeFailure` unions, `MAX_DOCUMENT_CHARACTERS` |
| `src/extractor/schemas.py` | `TermsOfService`, the `SCHEMAS` registry |
| `pyproject.toml` | Dependencies, dev group, ruff, mypy strict, pytest markers and default deselection |
| `AGENTS.md` | Verification commands and module-level prohibitions |
| `CODING_STANDARDS.md` | The rules every change must hold, and the rejected alternatives |
| `docs/agents/domain.md` | Vocabulary: outcome, port, absent field, intake |
| `docs/adr/0001-provider-adapters-must-enforce-the-schema.md` | Why every provider must enforce the schema |
| `docs/adr/0002-extraction-outcomes-are-a-closed-union.md` | Why outcomes are a union behind a consumer-declared port |
| `docs/adr/0003-the-live-test-asserts-absent-fields-not-model-wording.md` | What the paid test must assert |
| `docs/adr/0004-provider-failures-are-extraction-outcomes.md` | Provider failure versus rejected request, and the skip-versus-fail rule |
