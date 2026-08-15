# Extractor architecture

> **Status:** Current implementation
>
> **Verification basis:** `df3a855`

## 1. Executive summary

The extractor is a one-shot command line tool. You give it one messy document and the name of
a schema. It gives you back one validated JSON object on stdout, or it names exactly why it
could not.

It has four modules and no state. `__main__.py` parses the command line and decides the exit
code. `intake.py` turns a path or `-` into a document string. `schemas.py` holds the Pydantic
models that define what can be extracted. `extraction.py` wraps OpenAI: it builds the prompt,
binds the schema with provider-enforced strict mode, makes the call, and translates everything
that can happen into one of six named outcomes.

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
+-----------------+          strict json_schema call        +----------+
|   extractor     | -------------------------------------> | OpenAI   |
|   (one process) | <------------------------------------- | API      |
+-----------------+          object, refusal, or error      +----------+
      |        |
      |        +--> stderr: diagnostics, optional raw message dump
      +-----------> stdout: one JSON object, or the schema name list
```

Inside the boundary: argument parsing, document intake, the character ceiling, prompt
assembly, the schema binding, outcome classification, and exit codes.

Outside the boundary: the OpenAI service, the filesystem the document is read from, and the
`.env` file that supplies the API key.

## 3. Architectural invariants

1. **An extraction attempt returns a value, never raises.** The adapter's `extract` closure
   catches `OpenAIRefusalError`, the rejected-request exception classes, and the SDK base
   `APIError`, and returns an outcome for each. Enforced by the catch order in
   `src/extractor/extraction.py:109-127`; the base `APIError` catch is last so a new SDK
   subclass degrades to `ProviderFailure` rather than escaping.

2. **Outcomes are a closed union, matched exhaustively.** `Extraction` is a `type` alias over
   six frozen dataclasses. `_report` ends its `match` with `case unreachable:
   assert_never(unreachable)`. Enforced by `mypy --strict`: adding a seventh member fails type
   checking at the match instead of falling through at runtime. See ADR-0002.

3. **A document returned by intake has already cleared the ceiling.** `load_source_document`
   returns `str | InputFailure`, and the `str` branch is only reachable after the
   `MAX_DOCUMENT_CHARACTERS` check. Enforced by the return type, not by call ordering in
   `main`.

4. **No provider or framework type crosses into the CLI.** `__main__.py` imports only from
   `extractor.*`. `ExtractionPort`'s signature is `(str, type[BaseModel]) -> Extraction`.
   Enforced structurally by the Protocol signature and by review of imports.

5. **Every schema field is required and nullable with no default.** OpenAI strict mode demands
   it. Enforced by a test that reads the generated JSON schema
   (`tests/test_schemas.py:19`), not by the type system. See ADR-0001.

6. **The extracted object is the only thing on stdout.** Every diagnostic goes to stderr,
   including the `--debug` raw message dump. Enforced by `_report` being the sole writer of
   outcome output, and by CLI tests that assert stdout is empty on failure.

7. **Exit code numbers are published.** `README.md` documents them and
   `tests/test_cli.py:350` pins each member's integer. Renumbering is a breaking change.

8. **One implementation of `ExtractionPort` in production.** A second adapter would satisfy
   the Protocol and typecheck while silently dropping `strict=True`. Enforced by ADR-0001 and
   review only; nothing mechanical prevents it.

## 4. Components and dependencies

Dependencies point one way: `__main__` depends on `intake`, `schemas`, and `extraction`.
Nothing depends on `__main__`. `intake` and `schemas` depend on nothing in the module.

**`__main__.py`** owns argument parsing, the `ExitCode` enum, the outcome-to-exit-code mapping,
and the top-level error net. It does not own how a document is read, what the schemas are, or
how the provider is called. It never imports LangChain or OpenAI.

**`intake.py`** owns reading stdin or a UTF-8 file, the 100,000-character ceiling, and the
rendered `InputFailure` message. It does not own classifying why the read failed into named
causes; there is one caller and it just prints the message. It is deliberately not a seam.

**`schemas.py`** owns the named schema registry (`SCHEMAS`, currently one entry, `tos`) and the
field descriptions, which are prompt surface as well as validation. It does not own schema
selection or error reporting.

**`extraction.py`** owns everything provider-shaped: `load_dotenv`, the `ChatOpenAI`
construction, the `ChatPromptTemplate`, `with_structured_output(..., method="json_schema",
strict=True, include_raw=True)`, the debug dump, the exception-to-outcome mapping, and the six
outcome dataclasses. It also declares `ExtractionPort` even though `__main__` is the consumer,
because the union lives here. It does not own exit codes or any output formatting other than
the debug dump.

The seam between the CLI and the provider is `PortFactory = Callable[[str, TextIO | None],
ExtractionPort]`. `main` takes it as a keyword-only parameter defaulting to `build_openai_port`,
which is how every CLI test reaches `main` without a provider.

## 5. Critical flows

### Successful extraction

1. `main` parses argv. `--list-schemas` short-circuits: it writes sorted names to stdout and
   returns `OK`.
2. Missing `--schema` or missing input path writes an input error and returns `FAILURE`.
3. An unrecognised schema name writes the valid names and returns `FAILURE`. No provider call
   happens, and no document is read.
4. `load_source_document` reads stdin or the file. An `InputFailure` is printed and returns
   `FAILURE`.
5. `port_factory(model_id, debug_stream)` runs. `build_openai_port` loads
   `extractor/.env`, raises `ConfigurationError` if `OPENAI_API_KEY` is absent, then constructs
   `ChatOpenAI(reasoning_effort="none", temperature=0, timeout=60, max_retries=2)` and the
   prompt template. Construction happens after intake, so an oversize document costs nothing.
6. `extract(document, schema)` binds the schema strictly, invokes `prompt | structured_model`,
   and inspects the `{"raw", "parsed", "parsing_error"}` dict.
7. If `--debug` was passed, the raw message is written to stderr before classification.
8. Classification order: `parsing_error` that is an `OpenAIRefusalError` becomes `Refusal`; any
   other `parsing_error` becomes `ValidationFailure`; `parsed is None` becomes
   `EmptyExtraction`; otherwise `Extracted`.
9. `_report` writes `value.model_dump_json()` plus a newline to stdout and returns `OK`.

### Failure before an answer

The `chain.invoke` call is wrapped in one funnel, not a check per call. `OpenAIRefusalError`
becomes `Refusal`. `BadRequestError`, `NotFoundError`, and `UnprocessableEntityError` become
`ProviderRejectedRequest`, and these are caught before the base class. Any remaining `APIError`
becomes `ProviderFailure`. Anything else propagates to `main`'s `except Exception`, which
prints "Unexpected error" and returns `FAILURE`.

### Recovery

Two SDK retries and a 60-second timeout are configured on the client. A rate limit that
surfaces as `ProviderFailure` has therefore already exhausted them. There is no retry logic in
this module. A `ProviderFailure` may succeed on re-run; a `ProviderRejectedRequest` will not,
because the request itself is malformed.

## 6. Interfaces and data

**Command line:** `python -m extractor [input] --schema NAME [--model ID] [--list-schemas]
[--debug]`. `input` is a path or `-` for stdin. Default model is `gpt-5-nano`.

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

The API key is the one secret. It is read from the process environment or `extractor/.env` by
`load_dotenv` inside `build_openai_port`, never at import time. A missing key raises
`ConfigurationError` before the model is constructed, so no call is attempted without one. The
key is never written to stdout or stderr.

Untrusted input is the source document. It is sent to OpenAI verbatim as the human message.
There is no sanitisation and no prompt-injection defence: a document that instructs the model
to ignore the system message can influence the extracted values. Strict schema enforcement
bounds the *shape* of what comes back, not its truthfulness. This is an accepted property of a
single-user CLI, not a mitigated risk.

The character ceiling is a cost and blast-radius guard, not a security control. It fails closed:
an oversize document is refused before the provider is constructed.

## 8. Failure, capacity, and operations

One document, one call, one process. No concurrency primitives and no context threading.
Splitting a document is `rag/`'s concern.

Hard limits: 100,000 characters per document, a 60-second request timeout, two SDK retries.
Model temperature is 0 and `reasoning_effort` is `"none"`, so runs are as deterministic as the
provider allows.

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
- `tests/test_extraction.py` covers the adapter's classification of parsed objects, empty
  answers, schema rejections, refusals from both the raw message and a raised exception, the
  debug stream branches, the model configuration, the missing-key failure, the strict binding
  arguments, and the dotenv path.
- `tests/test_cli.py` covers each outcome's exit code and stderr line through a staged port,
  the input and configuration paths, and pins the exit numbers directly.
- `tests/test_live.py` is the only test that lets the real provider enforce the schema. It is
  marked `live` and deselected by default (`addopts = "-m 'not live'"`). It asserts field
  *presence*, so a field the fixture does not answer must come back null, and values that have
  exactly one faithful rendering. It does not assert model wording. See ADR-0003.

The default run passes offline with no API key. `mypy --strict` is load-bearing for invariant
2, not hygiene.

Unverified:

- The adapter's exception-to-outcome mapping for `APIError`, `BadRequestError`,
  `NotFoundError`, and `UnprocessableEntityError` has no test. ADR-0004 records this
  deliberately: `mypy --strict` and review are the available proof. Verifying it would need a
  test that installs a stubbed provider raising each exception class, in the shape
  `tests/test_extraction.py` already uses for `OpenAIRefusalError`.
- Nothing in a default run notices when the live test's assertions are weakened, because the
  test is deselected. ADR-0003 states plainly that reading the assertions is the only
  mechanism.

## 10. Known limitations

- One schema ships (`tos`). The registry is closed to user-supplied schemas.
- The module is pinned to OpenAI. `strict=True` is discarded silently by non-OpenAI providers,
  so changing `--model` to another vendor's id degrades enforcement with no error and no
  warning. ADR-0001.
- No prompt-injection defence, as described in section 7.
- `ProviderFailure` covers a wide range: credentials, quota, rate limits, server errors,
  network, and timeout all collapse to exit 5 with the provider's rendered text as the only
  discriminator.
- The document ceiling counts characters, not tokens, so it does not bound provider cost
  precisely.

## 11. Source map

| File | What it defines |
| --- | --- |
| `src/extractor/__main__.py` | Entry point, `ExitCode`, argument parsing, `_report`, the exhaustive match |
| `src/extractor/extraction.py` | The `Extraction` union, `ExtractionPort`, `PortFactory`, `build_openai_port`, all provider vocabulary |
| `src/extractor/intake.py` | `load_source_document`, `InputFailure`, `MAX_DOCUMENT_CHARACTERS` |
| `src/extractor/schemas.py` | `TermsOfService`, the `SCHEMAS` registry |
| `pyproject.toml` | Dependencies, dev group, ruff, mypy strict, pytest markers and default deselection |
| `AGENTS.md` | Verification commands and module-level prohibitions |
| `CODING_STANDARDS.md` | The rules every change must hold, and the rejected alternatives |
| `docs/agents/domain.md` | Vocabulary: outcome, port, absent field, intake |
| `docs/adr/0001-strict-json-schema-pins-openai.md` | Why the module is OpenAI-only |
| `docs/adr/0002-extraction-outcomes-are-a-closed-union.md` | Why outcomes are a union behind a consumer-declared port |
| `docs/adr/0003-the-live-test-asserts-absent-fields-not-model-wording.md` | What the paid test must assert |
| `docs/adr/0004-provider-failures-are-extraction-outcomes.md` | Provider failure versus rejected request, and the skip-versus-fail rule |
