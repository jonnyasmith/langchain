# The live test asserts absent fields, not model wording

`tests/test_live.py` is the only test that lets the real provider enforce the schema, so it is
the only place two decisions can be checked at all: that `method="json_schema", strict=True`
(ADR-0001) actually holds provider-side, and that the domain's `Absent field` rule survives
contact with a model — a field the source does not answer comes back null rather than filled
with a plausible value.

The line it draws is between a field's *presence* and its *wording*. Presence is a property of
the fixture and is asserted: the fixture states no data retention period, so
`data_retention_period` must be null, and that single assertion is what catches a model that
guesses. Wording is a property of the model and is not asserted:
`termination_notice_period` is "30 days" in the fixture, but "thirty days" is an equally
faithful extraction and no substring covers both. Where a value has exactly one faithful
rendering — a date, a boolean, a jurisdiction name, a numeral in a liability cap — it is
asserted, because a model that gets those wrong is a finding rather than a variation.

## Consequences

"Do not assert on extracted values" is too coarse a rule for this test and must not be applied
to it. Reduced to a shape check — `model_validate(stdout)` and nothing more — this test passes
against a model that invented a retention period, which is precisely the behaviour ADR-0001's
strict binding and the domain's `Absent field` rule exist to prevent. That reduction happened
once, in PR #5, and cost the repository its only check on both. The assertions and the
comments explaining which fields are excluded are therefore load-bearing; thin them only by
revisiting this ADR.

Because the default run deselects this test, nothing in CI notices when its coverage is
weakened, and no coverage or mutation metric computed over the offline suite can see it either.
Its assertions are reviewed by reading them, and that is the only mechanism available.

A missing `OPENAI_API_KEY` skips this test with a warning naming what went unchecked, rather
than failing it. A provider failure after construction skips in the same voice because the
strict-schema contract was not exercised. A provider-rejected request must fail: it means the
provider judged the extractor's request invalid, which is what this test exists to detect.
A red suite for a cost the contributor never opted into is not a signal, and a silent skip is
read as a pass. ADR-0004 records the outcome distinction and skip-versus-fail rule.
