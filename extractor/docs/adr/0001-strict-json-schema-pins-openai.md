# Strict JSON schema output, which pins this app to OpenAI

The extractor calls `.with_structured_output(Schema, method="json_schema", strict=True)` so the provider enforces the schema rather than the model being asked to honour it. That is an OpenAI-specific guarantee: `langchain-anthropic` has no `strict` parameter, and `BaseChatModel.with_structured_output` pops `method` and `strict` and discards them silently for any provider that has not overridden the method. Swapping the model string to a non-OpenAI provider therefore degrades enforcement to a polite request with no error and no warning.

## Consequences

Schemas must obey OpenAI strict-mode rules: every property listed as required, optionality expressed as a nullable type (`str | None` with no default, never `Optional[str] = None`), and no `allOf` / `not` / `if` / `then` / `else`. Refusals surface as `OpenAIRefusalError` from `langchain_openai.chat_models.base` — a bare `Exception`, not a `ValueError`, and not exported at package top level. Portability is deliberately not a goal here; `chatbot/` and later apps are free to choose differently.

`ExtractionPort` does not soften this. Its signature names no provider type, which makes it look like the seam a second provider would slot into — it is not. The port exists so tests can stage an outcome without a provider; the OpenAI adapter behind it is the only implementation this app intends to have. A second adapter would satisfy the Protocol and typecheck cleanly while silently dropping `strict`, which is exactly the failure described above. Add one only by revisiting this ADR.
