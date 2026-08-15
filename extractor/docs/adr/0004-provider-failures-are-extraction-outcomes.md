# Provider failures are extraction outcomes, but rejected requests are not skippable

An extraction attempt can end before the model answers. That end state is still part of the extraction contract: the adapter returns a frozen `ProviderFailure` or `ProviderRejectedRequest` outcome carrying the provider diagnostic rendered as text.

The distinction is whose request was wrong. A provider failure means the provider could not serve an otherwise well-formed request: authentication and authorization failures, quota and rate limits, exhausted aggregator credit, server failures, connection failures, timeouts, and any provider API error class not otherwise recognised. A provider-rejected request means the provider judged the request malformed: HTTP 400, 404, or 422, including an unknown model id. Re-running a rejected request unchanged cannot succeed.

Each adapter's exception mapping belongs inside `extraction.py`. Specific rejected-request
exceptions are caught before the SDK family's base exception; that final catch makes newly
added SDK subclasses degrade to `ProviderFailure` rather than escape as unexpected failures.
Outcomes store `str(error)`, not exception instances, and no provider type crosses the
extraction port.

## Two exception families, not three

Three providers ship and there are two funnels. OpenAI and OpenRouter share one, because reaching the aggregator through the OpenAI-compatible chat model means both raise the same exception classes. Anthropic has its own. Its SDK happens to use the same class names — `BadRequestError`, `NotFoundError`, `UnprocessableEntityError`, `APIError` — and shares no ancestor with them beyond `Exception`, so a single classifier would have to import both SDKs to name their classes. The mapping therefore stays inside each adapter, which is where this record already places it. The two SDKs do not even share an HTTP library, which is why tests build each family's errors with the flavour its own SDK carries.

Exhausted OpenRouter credit needs no case of its own. It arrives as a status error with no named subclass, falls through to the family base class, and lands on a provider failure — which is correct, since re-running after topping up succeeds.

## Refusal is asymmetric

A refusal is reported wherever the provider reports one, and the providers do not agree on where that is:

- **OpenAI** raises it, and also carries it on the raw response message.
- **OpenRouter** can surface it, but does not guarantee it. The refusal error is raised off a message field any OpenAI-format response may carry, so whether it appears depends on the upstream provider passing it through.
- **Anthropic** reports it as a stop reason on the raw message. Reading it there is the one raw-message inspection sanctioned inside an adapter, because there is nowhere else the provider states it.

The asymmetry is recorded rather than papered over: a provider that silently cannot report a refusal is the same class of quiet degradation the enforcement rule closes elsewhere, so the documentation states plainly which providers can report one. A refusal is classified before a parsing error, because a refused call carries no object to validate and reporting it as a validation failure would name the wrong cause.

## Consequences

Provider failures exit 5 and provider-rejected requests exit 6. Existing exit codes keep their published values. In particular, an unknown model id exits 6 on every provider rather than the generic exit 1.

Every adapter explicitly configures a 60-second request timeout and two SDK retries, spelled with the field name its own integration declares. A surfaced rate limit has therefore already exhausted the configured retries, while one document cannot inherit an SDK's ten-minute default.

A paid live test skips loudly after exit 5 because the enforced-schema contract went unchecked through no fault of the code. It does not skip after exit 6: provider rejection is evidence that the request or the schema binding is invalid, which is what those tests exist to detect. The key-presence check remains an offline construction check, asks only about the provider under test, and makes no separate provider probe.

Both exception funnels are exercised offline through the substituted chat-model seam. The tests
cover every specifically rejected request class on both families, each refusal each provider can
report, an unknown model id on all three providers, exhausted aggregator credit, and each
family's base exception.
