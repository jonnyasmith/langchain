# Provider adapters must enforce the schema

The extractor accepts a provider adapter only when that provider can enforce the bound
Pydantic schema. An adapter must use its provider's native enforced structured-output path. If
the selected provider or model cannot guarantee that enforcement, the adapter fails rather than
silently degrading to a prompt that merely asks the model to honour the schema.

This replaces the original OpenAI-only decision. Its premise expired when Anthropic added native
structured output and LangChain exposed it. Portability is now a goal: the provider registry is
the single place provider names appear as strings, and adding another conforming adapter is a
bounded addition rather than a change to the command-line layer or `ExtractionPort`.

## Consequences

Schemas must satisfy the strictest shared enforcement rules: every property is required,
optionality is expressed as a nullable type (`str | None` with no default, never
`Optional[str] = None`), and schemas avoid constructs an adapter's enforced path cannot support.

Every adapter owns its binding and proves it through the substituted chat-model seam. A second
`ExtractionPort` implementation is allowed, but it must preserve provider-side enforcement and
the same six extraction outcomes. A provider that can only offer loose JSON output cannot be
registered.
