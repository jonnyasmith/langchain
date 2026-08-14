# Extraction outcomes are a closed union behind a consumer-declared port

One extraction attempt returns exactly one frozen dataclass from the closed `Extraction` union. The model-answer outcomes are `Extracted`, `EmptyExtraction`, `ValidationFailure`, and `Refusal`; `ProviderFailure` and `ProviderRejectedRequest` describe attempts that ended before an answer. The CLI obtains them through `ExtractionPort`, a Protocol declared where it is consumed, whose signature is `(document, schema) -> Extraction` and names no provider type.

The rejected alternative is the one LangChain hands you and the one this app originally shipped for model answers: `include_raw=True` returns a `{"raw", "parsed", "parsing_error"}` dict, and that shape was passed to the caller as a `TypedDict`. It is a total of three nullable fields describing four answer states, so the caller has to interrogate them in the right order to recover which state it is in — `parsed is None` means something different depending on whether `parsing_error` is set, and a refusal is a `parsing_error` of one specific class. Nothing enforces the order and nothing notices when a case is missed. Naming every end state directly replaces that reconstruction with a match. ADR-0004 records why failures before an answer also belong to this union.

## Consequences

`main` matches the union exhaustively and closes it with `assert_never`, so `mypy --strict` is load-bearing rather than hygiene: adding another outcome fails type-checking at every incomplete match instead of falling through at runtime. `uv run mypy` is therefore not optional in verification — dropping it would remove the only thing enforcing this decision. The same exhaustiveness carries into `ExitCode`, where each outcome maps to exactly one published status.

Provider vocabulary — `include_raw`, `parsing_error`, `OpenAIRefusalError`, provider API exceptions, the prompt, the strict binding — stays inside the adapter in `extraction.py`. No LangChain or OpenAI type appears in `__main__.py`. Tests reach the CLI by passing a port factory that returns a staged outcome, which is why the suite needs no chat-model subclasses at that seam; `tests/test_live.py` separately exercises the real strict-schema binding.

The port is not an invitation to add a second provider. See ADR-0001: a second adapter would satisfy the Protocol and typecheck cleanly while silently dropping `strict`.
