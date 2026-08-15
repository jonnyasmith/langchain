# Provider failures are extraction outcomes, but rejected requests are not skippable

An extraction attempt can end before the model answers. That end state is still part of the extraction contract: the adapter returns a frozen `ProviderFailure` or `ProviderRejectedRequest` outcome carrying the provider diagnostic rendered as text.

The distinction is whose request was wrong. A provider failure means the provider could not serve an otherwise well-formed request: authentication and authorization failures, quota and rate limits, server failures, connection failures, timeouts, and any provider API error class not otherwise recognised. A provider-rejected request means the provider judged the request malformed: HTTP 400, 404, or 422, including an unknown model id. Re-running a rejected request unchanged cannot succeed.

Each adapter's exception mapping belongs inside `extraction.py`. Specific rejected-request
exceptions are caught before the SDK family's base exception; that final catch makes newly
added SDK subclasses degrade to `ProviderFailure` rather than escape as unexpected failures.
Outcomes store `str(error)`, not exception instances, and no provider type crosses the
extraction port.

## Consequences

Provider failures exit 5 and provider-rejected requests exit 6. Existing exit codes keep their published values. In particular, an unknown model id now exits 6 rather than the generic exit 1.

The adapter explicitly configures a 60-second request timeout and two SDK retries. A surfaced rate limit has therefore already exhausted the configured retries, while one document cannot inherit the SDK's ten-minute timeout.

The paid live test skips loudly after exit 5 because the strict-schema contract went unchecked through no fault of the code. It does not skip after exit 6: provider rejection is evidence that the request or strict-schema binding is invalid, which is what that test exists to detect. The key-presence fixture remains an offline construction check and the test makes no separate provider probe.

Each exception-to-outcome funnel is exercised offline through the substituted chat-model seam.
The tests cover every specifically rejected request class, each refusal class an adapter can
report, and the provider family's base exception.
