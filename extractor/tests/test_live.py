"""The extractor's paid provider tests and its offline outcome-classification tests.

`pyproject.toml` sets `addopts = "--strict-markers -m 'not live'"`, so the paid tests are
deselected by every default run and the offline rule in `AGENTS.md` holds unchanged.
Run them deliberately, with a funded key in the environment or in `extractor/.env`:

    uv run pytest -m live

Schema enforcement is a provider-side guarantee, so no offline substitute can exercise it, and
the three providers enforce by three different mechanisms — hence one test each rather than one
parameterised test. The offline tests here cover only whether named provider outcomes skip or
fail that contract. Which fields the paid tests assert, and which they deliberately leave
alone, is ADR-0003.
"""

import json
import warnings
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel

from extractor.__main__ import ExitCode, main
from extractor.credentials import ConfigurationError
from extractor.extraction import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    Extraction,
    ExtractionPort,
    PortSettings,
    Provider,
    ProviderFailure,
    ProviderRejectedRequest,
    ReasoningLevel,
)
from extractor.schemas import TermsOfService

FIXTURE = Path(__file__).parent / "fixtures" / "terms.html"


def _staged(outcome: Extraction) -> Provider:
    """A registry entry whose port returns one prepared live-test outcome."""

    def factory(_settings: PortSettings) -> ExtractionPort:
        def extract(document: str, schema: type[BaseModel]) -> Extraction:
            return outcome

        return extract

    return Provider(PROVIDERS[DEFAULT_PROVIDER].default_model, factory)


def _require_live_success(exit_code: ExitCode, stderr: str) -> None:
    """Skip only when the provider could not check the live contract; fail otherwise."""
    if exit_code is ExitCode.PROVIDER_FAILURE:
        message = (
            "LIVE TEST SKIPPED, NOT PASSED: the provider could not serve the request, so "
            f"the enforced-schema contract was never checked ({stderr.strip()}). Rerun "
            "`uv run pytest -m live` after the provider or account recovers."
        )
        warnings.warn(message, stacklevel=1)
        pytest.skip(message)
    assert exit_code is ExitCode.OK, stderr


def _require_configured(name: str) -> None:
    """Skip loudly when this provider has no key — never let a live test look like it ran.

    Skipping beats failing: a contributor with no key must not be handed a red suite for a
    cost they did not opt into, and "skipped" is never read as "passed". pytest prints skip
    reasons only under `-rs`, so the warning carries the reason into the warnings summary.
    The adapter owns the definition of "configured", so asking it cannot drift from the code
    this guards, and each provider is asked only about its own key. Called inside the test
    rather than at collection so a default offline run never asks, and `_load_env_file` does
    not write a key into `os.environ` during a run that never intended to call a provider.
    """
    provider = PROVIDERS[name]
    try:
        provider.build_port(PortSettings(provider.default_model, ReasoningLevel.MEDIUM, None))
    except ConfigurationError as error:
        message = (
            f"LIVE TEST SKIPPED, NOT PASSED: {name} is not configured, so the real provider was "
            f"never called and its enforced-schema contract was never checked ({error}). Set "
            "the key in the environment or in extractor/.env, then rerun `uv run pytest -m live`."
        )
        warnings.warn(message, stacklevel=2)
        pytest.skip(message)


def _assert_fixture_terms(value: TermsOfService) -> None:
    """The field assertions ADR-0003 sanctions: presence and single-rendering values only."""
    # The absent field first: the fixture states no retention period, and the domain's
    # `Absent field` rule forbids inferring one. This is the assertion that catches a guess,
    # and it is the reason these tests assert fields at all rather than only the shape.
    assert value.data_retention_period is None, (
        "the fixture states no data retention period, so the field must be null, not "
        f"{value.data_retention_period!r}"
    )
    assert value.arbitration_required is True
    assert value.arbitration_clause, "the fixture has an arbitration clause"
    assert value.effective_date == date(2026, 1, 1)
    assert value.governing_law is not None
    assert "new york" in value.governing_law.lower(), value.governing_law
    assert value.liability_cap is not None
    assert "100" in value.liability_cap, value.liability_cap
    # `termination_notice_period` is deliberately not asserted: the fixture says "30 days", but
    # "thirty days" is an equally faithful extraction and no substring covers both. ADR-0003.


def _run_live(capsys: pytest.CaptureFixture[str], name: str) -> None:
    """The whole app against one real provider: one document in, one JSON object out, exit 0."""
    _require_configured(name)

    exit_code = main(["--provider", name, "--schema", "tos", str(FIXTURE)])

    captured = capsys.readouterr()
    _require_live_success(exit_code, captured.err)
    assert captured.err == ""
    _assert_fixture_terms(TermsOfService.model_validate(json.loads(captured.out)))


def test_a_provider_failure_skips_the_unchecked_live_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["--schema", "tos", str(FIXTURE)],
        providers={
            DEFAULT_PROVIDER: _staged(ProviderFailure(detail="Error code: 429 - quota exhausted"))
        },
    )
    captured = capsys.readouterr()

    with (
        pytest.warns(UserWarning, match="LIVE TEST SKIPPED, NOT PASSED"),
        pytest.raises(pytest.skip.Exception, match="enforced-schema contract was never checked"),
    ):
        _require_live_success(exit_code, captured.err)


def test_a_provider_rejected_request_fails_the_live_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["--schema", "tos", str(FIXTURE)],
        providers={
            DEFAULT_PROVIDER: _staged(
                ProviderRejectedRequest(detail="Error code: 404 - model does not exist")
            )
        },
    )
    captured = capsys.readouterr()

    with pytest.raises(AssertionError, match="Provider-rejected request"):
        _require_live_success(exit_code, captured.err)


@pytest.mark.live
def test_openai_enforces_the_schema_and_extracts_the_fixture_terms(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Native strict json-schema output, the mechanism only OpenAI uses."""
    _run_live(capsys, "openai")


@pytest.mark.live
def test_anthropic_enforces_the_schema_and_extracts_the_fixture_terms(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The json-schema structured-output method, with thinking on at the default level."""
    _run_live(capsys, "anthropic")


@pytest.mark.live
def test_openrouter_enforces_the_schema_and_extracts_the_fixture_terms(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Strict json schema plus the routing guard, so only an endpoint that can enforce it runs."""
    _run_live(capsys, "openrouter")
