"""The extractor's paid provider test and its offline outcome-classification tests.

`pyproject.toml` sets `addopts = "--strict-markers -m 'not live'"`, so the paid test is
deselected by every default run and the offline rule in `AGENTS.md` holds unchanged.
Run it deliberately, with a funded key in the environment or in `extractor/.env`:

    uv run pytest -m live

`method="json_schema", strict=True` (ADR-0001) is a provider-side guarantee, so no offline
substitute can exercise it. The offline tests cover only whether named provider outcomes
skip or fail that contract. Which fields the paid test asserts, and which it deliberately
leaves alone, is ADR-0003.
"""

import json
import warnings
from datetime import date
from pathlib import Path
from typing import TextIO

import pytest
from pydantic import BaseModel

from extractor.__main__ import DEFAULT_MODEL, ExitCode, main
from extractor.extraction import (
    ConfigurationError,
    Extraction,
    ExtractionPort,
    PortFactory,
    ProviderFailure,
    ProviderRejectedRequest,
    build_openai_port,
)
from extractor.schemas import TermsOfService

FIXTURE = Path(__file__).parent / "fixtures" / "terms.html"


def _staged_port(outcome: Extraction) -> PortFactory:
    """Build a port factory that returns one prepared live-test outcome."""

    def factory(_model_id: str, _debug: TextIO | None) -> ExtractionPort:
        def extract(document: str, schema: type[BaseModel]) -> Extraction:
            return outcome

        return extract

    return factory


def _require_live_success(exit_code: ExitCode, stderr: str) -> None:
    """Skip only when the provider could not check the live contract; fail otherwise."""
    if exit_code is ExitCode.PROVIDER_FAILURE:
        message = (
            "LIVE TEST SKIPPED, NOT PASSED: the provider could not serve the request, so "
            f"the strict-schema contract was never checked ({stderr.strip()}). Rerun "
            "`uv run pytest -m live` after the provider or account recovers."
        )
        warnings.warn(message, stacklevel=1)
        pytest.skip(message)
    assert exit_code is ExitCode.OK, stderr


@pytest.fixture
def configured_provider() -> None:
    """Skip loudly when there is no key — never let a live test look like it ran.

    Skipping beats failing: a contributor with no key must not be handed a red suite for a
    cost they did not opt into, and "skipped" is never read as "passed". pytest prints skip
    reasons only under `-rs`, so the warning carries the reason into the warnings summary.
    `build_openai_port` owns the definition of "configured", so asking it cannot drift from
    the code this guards. A fixture rather than a module-level `skipif` means the default
    offline tests never request it, and `load_dotenv` does not write the key into `os.environ`
    during collection of a run that never intended to call a provider.
    """
    try:
        build_openai_port(DEFAULT_MODEL, None)
    except ConfigurationError as error:
        message = (
            "LIVE TEST SKIPPED, NOT PASSED: no OPENAI_API_KEY, so the real provider was never "
            f"called and the strict-schema contract was never checked ({error}). Set the key "
            "in the environment or in extractor/.env, then rerun `uv run pytest -m live`."
        )
        warnings.warn(message, stacklevel=1)
        pytest.skip(message)


def test_a_provider_failure_skips_the_unchecked_live_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["--schema", "tos", str(FIXTURE)],
        port_factory=_staged_port(ProviderFailure(detail="Error code: 429 - quota exhausted")),
    )
    captured = capsys.readouterr()

    with (
        pytest.warns(UserWarning, match="LIVE TEST SKIPPED, NOT PASSED"),
        pytest.raises(pytest.skip.Exception, match="strict-schema contract was never checked"),
    ):
        _require_live_success(exit_code, captured.err)


def test_a_provider_rejected_request_fails_the_live_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["--schema", "tos", str(FIXTURE)],
        port_factory=_staged_port(
            ProviderRejectedRequest(detail="Error code: 404 - model does not exist")
        ),
    )
    captured = capsys.readouterr()

    with pytest.raises(AssertionError, match="Provider-rejected request"):
        _require_live_success(exit_code, captured.err)


@pytest.mark.live
def test_the_provider_enforced_schema_extracts_the_fixture_terms(
    capsys: pytest.CaptureFixture[str],
    configured_provider: None,
) -> None:
    """The whole app against the real provider: one document in, one JSON object out, exit 0."""
    exit_code = main(["--schema", "tos", str(FIXTURE)])

    captured = capsys.readouterr()
    _require_live_success(exit_code, captured.err)
    assert captured.err == ""

    value = TermsOfService.model_validate(json.loads(captured.out))
    # The absent field first: the fixture states no retention period, and the domain's
    # `Absent field` rule forbids inferring one. This is the assertion that catches a guess,
    # and it is the reason this test asserts fields at all rather than only the shape.
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
