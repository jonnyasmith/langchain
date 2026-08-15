# Extractor architecture

> **Status:** Current implementation
>
> **Verification basis:** `d0a9856`

## 1. Executive summary

The extractor is a one-shot command line tool. You give it one messy document and the name of
a schema. It gives you back one validated JSON object on stdout, or it names exactly why it
could not.

It has six modules and no state. `invocation.py` turns one command line into a resolved
invocation or a named invocation failure. `__main__.py` runs what was resolved and decides the
exit code. `intake.py` turns a path or `-` into a document string. `schemas.py` holds the
Pydantic models that define what can be extracted. `credentials.py` resolves one provider key
from the environment or `extractor/.env`. `extraction.py` owns the provider registry and its
three adapters — OpenAI, Anthropic, and OpenRouter: it builds the prompt, binds the schema
through each provider's enforced path, makes the call, and translates everything that can happen
into one of six named outcomes.

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

2. **Every expected failure is a closed union, matched exhaustively.** `Extraction` is a `type`
   alias over six frozen dataclasses; `IntakeFailure` and `InvocationFailure` are the same shape
   for their own domains. Each `match` ends with `case unreachable: assert_never(unreachable)`.
   Enforced by `mypy --strict`: adding a member fails type checking at the match instead of
   falling through at runtime. See ADR-0002.

3. **A document returned by intake has already cleared the ceiling.** `load_source_document`
   returns `Intake`, and the `str` branch is only reachable after the
   `MAX_DOCUMENT_CHARACTERS` check. Enforced by the return type, not by call ordering in
   `main`.

4. **No provider or framework type crosses into the CLI.** `__main__.py` imports only from
   `extractor.*`. The `PROVIDERS` registry in `extraction.py` is the sole place provider names
   appear as strings, and each entry is a `ProviderAdapter` carrying that provider's default
   model, its model builder, and its integration; `invocation.py` only looks up the parsed
   value. `main` depends on the `Provider` protocol, not on the concrete record, so a test
   satisfies it without naming a provider SDK. `ExtractionPort` remains
   `(str, type[BaseModel]) -> Extraction`.

5. **Every schema field is required and nullable with no default.** Enforced structured output
   demands it. Enforced by tests that read the generated JSON schema for every entry in
   `SCHEMAS`, not by the type system, so a newly registered schema cannot escape the contract.
   See ADR-0001.

9. **Resolution decides; `main` acts.** `resolve` returns a value and never constructs a port,
   reads a document, or writes a diagnostic. Every argument check therefore happens before
   anything costs money or touches the filesystem, and that ordering is structural rather than
   a sequence `main` has to preserve.

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

Dependencies point one way: `__main__` depends on `invocation`, `intake`, `credentials`, and
`extraction`; `invocation` depends on `extraction` and `schemas`; `extraction` depends on
`credentials`. Nothing depends on `__main__`. `intake`, `schemas`, and `credentials` depend on
nothing in the module.

**`__main__.py`** owns the `ExitCode` enum, the three rendering functions and their exhaustive
matches, and the top-level error net. It does not own how a command line is validated, how a
document is read, what the schemas are, or how a provider is called. It never imports LangChain
or a provider SDK.

**`invocation.py`** owns the argument parser, the order the checks run in, and the closed
`Resolution` union: a resolved `Invocation`, a `SchemaListing`, `HelpRequested`, or one of five
`InvocationFailure` members. It does not own their wording — failures carry their facts and
`__main__` renders them, the same split `intake` and `extraction` use. An `Invocation` holds the
selected `Provider` and the built `PortSettings`, not the raw strings, so no caller can reach a
value that was never checked. Like `intake`, it is deliberately not a seam.

**`credentials.py`** owns `ENV_FILE`, the `.env` reader, and `ConfigurationError`. Its whole
interface is `required_key(name)`: a returned key is usable, and everything else raises. It does
not own which provider needs which key. Not a seam — one implementation, and tests point
`ENV_FILE` at a temporary file rather than substituting it.

**`intake.py`** owns reading stdin or a UTF-8 file, the 100,000-character ceiling, and the
classification of a refusal into `UnreadableSource` or `OversizeDocument`. It does not own the
wording of either report: failures carry their facts and `__main__` renders them, the same
split `extraction.py` and `_report` use. It is deliberately not a seam.

**`schemas.py`** owns the named schema registry (`SCHEMAS`, currently one entry, `tos`) and the
field descriptions, which are prompt surface as well as validation. It does not own schema
selection or error reporting.

**`extraction.py`** owns everything provider-shaped: the `PROVIDERS` registry and its
`ProviderAdapter` entries, the two `Integration` values, frozen `PortSettings`, the three model
builders and their reasoning translation, the shared `ChatPromptTemplate`, enforced
structured-output binding, debug dumps, both exception-to-outcome funnels, and the six outcome
dataclasses. It declares `ExtractionPort` and the `Provider` protocol because the union lives
here. It does not own exit codes, credentials, or any output formatting other than debug dumps.

Two axes vary, and they are held separately. An `Integration` is one vendor chat integration:
how it binds a schema through its enforced path, and how its SDK reports failure. Two exist —
`OPENAI_FAMILY` and `ANTHROPIC` — and ADR-0004 is why they never merge. A `ProviderAdapter` is
one registered provider: its default model, how its chat model is built, and which integration
serves it. Three exist, and the registry states plainly that OpenRouter shares OpenAI's
integration. Everything invariant — the prompt, the debug dump, and outcome classification —
lives once in `ProviderAdapter.build_port`.

The seam between the CLI and providers is the `Provider` protocol: a `default_model` and
`build_port(settings)`. `PortSettings` carries the model id, provider-neutral reasoning level,
and optional debug stream. `main` receives a mapping of provider names to `Provider`, defaulting
to `PROVIDERS`; tests satisfy the protocol with a staged port and never import a provider or
framework type. Adapter tests substitute one step deeper, supplying a `build_model` that returns
a canned chat model.

## 5. Critical flows

### Successful extraction

1. `resolve` parses the command line. A parser exit becomes `HelpRequested` (exit `OK`) or
   `BadInvocation` (exit `FAILURE`, and nothing further is written because argparse already
   named the offending argument). It then checks `--provider` and `--reasoning`, returning
   `UnknownProvider` or `UnknownReasoningLevel` carrying the valid values. `--list-schemas`
   short-circuits *after* those checks, so a bad `--provider` beside it is still reported.
2. Missing `--schema` or missing input path returns `MissingArguments`.
3. An unrecognised schema name returns `UnknownSchema` with the valid names. Nothing has been
   constructed and no document has been read.
4. `main` matches the resolution. Every `InvocationFailure` goes to `_report_invocation`, which
   writes the diagnostic and returns `FAILURE`.
5. `load_source_document` reads stdin or the file. Anything that is not a `str` goes to
   `_report_intake`, which writes the diagnostic and returns `FAILURE`.
6. The resolved `Invocation` already carries the selected provider and one frozen
   `PortSettings`, whose model id is `--model` or the provider's own default. Its model builder
   calls `required_key`, which loads `extractor/.env` and raises `ConfigurationError` if that
   file is unreadable or if *its own* key — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
   `OPENROUTER_API_KEY` — is absent, then constructs its chat model. Construction happens after
   intake, so an oversize document costs nothing.
7. Reasoning is translated per provider by a function per integration, each matching over
   `ReasoningLevel` exhaustively. OpenAI maps `off` to a `none` effort and keeps the
   other three spellings. Anthropic maps `off` to a disabled thinking configuration with no
   effort at all, and each named level to an effort with thinking left adaptive — an effort and
   an explicit thinking configuration are mutually exclusive. OpenRouter sends an effort inside
   a `reasoning` object, `none` included.
8. `extract(document, schema)` binds the schema through the provider's enforced path — strict
   `json_schema` on OpenAI and OpenRouter, `json_schema` on Anthropic, which has no `strict`
   argument because the method *is* the enforcement — invokes `prompt | structured_model`, and
   inspects the `{"raw", "parsed", "parsing_error"}` dict.
9. If `--debug` was passed, the raw message is written to stderr before classification.
10. Classification order, shared by all three: a refusal becomes `Refusal`; any `parsing_error`
   becomes `ValidationFailure`; `parsed is None` becomes `EmptyExtraction`; otherwise
   `Extracted`. Where the refusal is read from is the per-provider part — a refusal error in
   the parsing-error slot on OpenAI and OpenRouter, a `stop_reason` of `refusal` on the raw
   message on Anthropic.
11. `_report` writes `value.model_dump_json()` plus a newline to stdout and returns `OK`.

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
`credentials.required_key`, called by the selected model builder, never at import time. A `.env` that exists but
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

`pyproject.toml` enables the `pydantic.mypy` plugin. It is not optional hygiene: `ChatAnthropic`
declares its model id, reasoning effort, and timeout behind field aliases, and without the plugin
`mypy --strict` rejects the constructor call the Anthropic adapter has to make.

Runtime dependencies for providers are `langchain-openai`, `langchain-anthropic`, `openai`, and
`anthropic`. OpenRouter adds none — it reuses the OpenAI integration. Each SDK is declared
directly rather than relied on transitively, because `extraction.py` imports the exception
classes of both by name.

- `tests/test_schemas.py` proves every field is described and is required-and-nullable, by
  reading the generated JSON schema. It runs over every entry in `SCHEMAS`, plus a guard that
  the registry is non-empty so the parametrised checks cannot pass vacuously. This is the only
  check on invariant 5.
- `tests/test_intake.py` covers stdin, UTF-8 files, missing and unreadable paths, undecodable
  bytes, and both sides of the ceiling boundary.
- `tests/test_credentials.py` covers credential resolution through `required_key` alone: the
  file defining a key the shell lacks, an exported key winning, comments and blank lines, an
  absent file, an absent key, an empty value, and unreadable or non-UTF-8 files.
- `tests/test_invocation.py` covers what `resolve` decides, as values: the resolved invocation's
  schema, source, model default and override, every reasoning level, the debug stream, and each
  of the named failures with the valid values it carries. A tripwire provider proves resolution
  never constructs a port.
- `tests/test_extraction.py` covers all three adapters at two seams. Outcome tests build a
  `ProviderAdapter` whose `build_model` returns a canned chat model — classification, each
  provider's refusal reporting, the debug dump, the binding arguments, and both exception
  funnels. Configuration tests substitute the SDK class in the module namespace, because the
  arguments handed to it are visible nowhere else: every reasoning translation, each model
  configuration, the per-provider key check, and the aggregator's emitted request against a
  loopback stub.
- `tests/test_cli.py` covers each outcome's exit code and stderr line through a staged port,
  per-provider default models and the `--model` override, the input and configuration paths, and
  pins the exit numbers directly.
- `tests/staging.py` holds `StagedProvider`, which satisfies the `Provider` protocol from a
  default model and a port factory. Shared by the CLI, live, and invocation tests.
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
- Reasoning is a cost lever, not a correctness contract, so a provider that ignores it is
  accepted — except on OpenRouter, where the routing guard makes the reasoning field its own
  routing constraint, so a `--model` that does not advertise reasoning is reported as a rejected
  request. The guard cannot be scoped to one parameter.
- Only `openai` pins temperature. Anthropic rejects a modified temperature while thinking is on,
  and under the routing guard temperature is another unadvertised parameter.
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
| `src/extractor/__main__.py` | Entry point, `ExitCode`, `_report`, `_report_intake`, `_report_invocation`, the exhaustive matches |
| `src/extractor/invocation.py` | `resolve`, the argument parser, `Invocation`, `SchemaListing`, the `InvocationFailure` and `Resolution` unions |
| `src/extractor/credentials.py` | `required_key`, `ENV_FILE`, `ConfigurationError` |
| `src/extractor/extraction.py` | The `Extraction` union, `ExtractionPort`, `PortFactory`, the `Provider` protocol, `Integration`, `ProviderAdapter`, `PROVIDERS`, all provider vocabulary |
| `src/extractor/intake.py` | `load_source_document`, the `Intake` and `IntakeFailure` unions, `MAX_DOCUMENT_CHARACTERS` |
| `src/extractor/schemas.py` | `TermsOfService`, the `SCHEMAS` registry |
| `pyproject.toml` | Dependencies, dev group, ruff, mypy strict plus the `pydantic.mypy` plugin, pytest markers and default deselection |
| `AGENTS.md` | Verification commands and module-level prohibitions |
| `CODING_STANDARDS.md` | The rules every change must hold, and the rejected alternatives |
| `docs/agents/domain.md` | Vocabulary: outcome, port, absent field, intake, invocation, integration, provider adapter |
| `docs/adr/0001-provider-adapters-must-enforce-the-schema.md` | Why every provider must enforce the schema |
| `docs/adr/0002-extraction-outcomes-are-a-closed-union.md` | Why outcomes are a union behind a consumer-declared port |
| `docs/adr/0003-the-live-test-asserts-absent-fields-not-model-wording.md` | What the paid test must assert |
| `docs/adr/0004-provider-failures-are-extraction-outcomes.md` | Provider failure versus rejected request, and the skip-versus-fail rule |
