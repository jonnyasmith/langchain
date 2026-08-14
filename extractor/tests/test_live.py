"""Live provider tests: the only tests that let the real provider enforce the schema.

These tests make real, paid OpenAI calls. `pyproject.toml` sets
`addopts = "--strict-markers -m 'not live'"`, so they are deselected by every default run
and `AGENTS.md`'s rule that the default run stays offline holds unchanged. Nothing here is
substituted, stubbed, or recorded — that is the point. `method="json_schema", strict=True`
(ADR-0001) is a provider-side guarantee, and no offline substitute can exercise it.

Run them deliberately, with a funded key in the environment or in `extractor/.env`:

    uv run pytest -m live

Without a key every test here skips, loudly and by name, rather than failing on a
traceback from the provider client.
"""

import json
import warnings
from datetime import date
from pathlib import Path

import pytest

from extractor.__main__ import DEFAULT_MODEL, ExitCode, main
from extractor.extraction import (
    ConfigurationError,
    EmptyExtraction,
    Extracted,
    Extraction,
    Refusal,
    ValidationFailure,
    build_openai_port,
)
from extractor.schemas import TermsOfService

FIXTURE = Path(__file__).parent / "fixtures" / "terms.html"


def _unconfigured_reason() -> str | None:
    """Why the provider is not configured, or `None` when it is.

    `build_openai_port` owns the definition of "configured", so asking it cannot drift from
    the code this guards. Called from the fixture rather than at import: module import
    happens during collection of every run, including offline ones, and `load_dotenv` writes
    the key into `os.environ` process-wide.
    """
    try:
        build_openai_port(DEFAULT_MODEL, None)
    except ConfigurationError as error:
        return str(error)
    return None


def _skip_reason(detail: str) -> str:
    return (
        "LIVE TESTS SKIPPED, NOT PASSED: no OPENAI_API_KEY, so the real provider was never "
        f"called and the strict-schema contract was never checked ({detail}). "
        "Set OPENAI_API_KEY in the environment or in extractor/.env, then rerun "
        "`uv run pytest -m live`."
    )


@pytest.fixture(autouse=True)
def configured_provider() -> None:
    """Skip, loudly, when there is no key — never let a live test look like it ran.

    Skipping beats failing: a contributor with no key must not be handed a red suite for a
    cost they did not opt into, and "skipped" is never read as "passed". pytest prints skip
    reasons only under `-rs`, so the warning carries the reason into the warnings summary,
    which a default run shows. Autouse, so no test lands here without the guard, and a
    fixture rather than a module-level `skipif` so an offline run — where these tests are
    deselected, not skipped — stays silent.
    """
    reason = _unconfigured_reason()
    if reason is not None:
        message = _skip_reason(reason)
        warnings.warn(message, stacklevel=1)
        pytest.skip(message)


def _why(outcome: Extraction) -> str:
    """Render a non-`Extracted` outcome's own diagnostic, so a failure says why."""
    match outcome:
        case Extracted():
            return "extraction succeeded"
        case ValidationFailure(detail=detail):
            return f"the provider returned an object the schema rejected: {detail}"
        case Refusal(detail=detail):
            return f"the provider refused to extract: {detail}"
        case EmptyExtraction():
            return "the provider answered but committed to no object"


@pytest.mark.live
def test_the_provider_enforced_schema_extracts_the_stated_terms() -> None:
    """The real provider, under `strict=True`, returns the fixture's stated terms and no others."""
    extract = build_openai_port(DEFAULT_MODEL, None)

    outcome = extract(FIXTURE.read_text(encoding="utf-8"), TermsOfService)

    assert isinstance(outcome, Extracted), _why(outcome)
    value = outcome.value
    assert isinstance(value, TermsOfService), (
        f"strict schema enforcement should yield a TermsOfService, got {type(value).__name__}"
    )
    # `data_retention_period` first: the fixture states no retention period, and the domain's
    # `Absent field` rule forbids inferring one. This is the assertion that catches a guess.
    assert value.data_retention_period is None, (
        "the fixture states no data retention period, so the field must be null, not "
        f"{value.data_retention_period!r}"
    )
    assert value.governing_law is not None
    assert "new york" in value.governing_law.lower(), value.governing_law
    assert value.effective_date == date(2026, 1, 1)
    assert value.arbitration_required is True
    assert value.arbitration_clause, "the fixture has an arbitration clause"
    assert value.liability_cap is not None
    assert "100" in value.liability_cap, value.liability_cap
    # `termination_notice_period` is deliberately not asserted: the fixture says "30 days",
    # but "thirty days" is an equally faithful extraction, and no substring covers both.


@pytest.mark.live
def test_the_cli_extracts_end_to_end_with_nothing_substituted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole app, real provider included: one document in, one JSON object out, exit 0."""
    exit_code = main(["--schema", "tos", str(FIXTURE)])

    captured = capsys.readouterr()
    assert exit_code is ExitCode.OK, captured.err
    assert captured.err == ""
    payload = json.loads(captured.out)
    # Shape only. Whether the model filled a field right is the other test's job; asserting
    # it here would redden the end-to-end wiring test for a model's guess.
    TermsOfService.model_validate(payload)
