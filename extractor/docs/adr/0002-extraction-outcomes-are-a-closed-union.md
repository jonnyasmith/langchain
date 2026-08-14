# Extraction outcomes are a closed union behind a consumer-declared port

One extraction attempt returns exactly one of four frozen dataclasses — `Extracted`, `EmptyExtraction`, `ValidationFailure`, `Refusal` — aliased as `Extraction`. The CLI obtains them through `ExtractionPort`, a Protocol declared where it is consumed, whose signature is `(document, schema) -> Extraction` and names no provider type.

The rejected alternative is the one LangChain hands you and the one this app originally shipped: `include_raw=True` returns a `{"raw", "parsed", "parsing_error"}` dict, and that shape was passed to the caller as a `TypedDict`. It is a total of three nullable fields describing four states, so the caller has to interrogate them in the right order to recover which state it is in — `parsed is None` means something different depending on whether `parsing_error` is set, and a refusal is a `parsing_error` of one specific class. Nothing enforces the order and nothing notices when a case is missed. Naming the four states directly replaces that reconstruction with a match.

## Consequences

`main` matches the union exhaustively and closes it with `assert_never`, so `mypy --strict` is load-bearing rather than hygiene: adding a fifth outcome fails type-checking at every incomplete match instead of falling through at runtime. `uv run mypy` is therefore not optional in verification — dropping it would remove the only thing enforcing this decision. The same exhaustiveness carries into `ExitCode`, where each outcome maps to exactly one published status.

Provider vocabulary — `include_raw`, `parsing_error`, `OpenAIRefusalError`, the prompt, the strict binding — stays inside the adapter in `extraction.py`. No LangChain type appears in `__main__.py`. Tests reach the CLI by passing a port factory that returns a staged outcome, which is why the suite needs no chat-model subclasses; the provider path is covered separately by the adapter's own tests and by `tests/test_live.py`.

The port is not an invitation to add a second provider. See ADR-0001: a second adapter would satisfy the Protocol and typecheck cleanly while silently dropping `strict`.
