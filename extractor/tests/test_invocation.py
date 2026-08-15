"""Command-line resolution, asserted as values rather than as rendered prose.

`test_cli.py` covers what each failure *reads like* on stderr. These cover what the resolver
*decides*, which is the half `__main__` depends on: which provider, which schema, which
settings, and which named failure. Nothing here captures output or stages a port.
"""

import sys
from collections.abc import Sequence
from io import StringIO

import pytest

from extractor.extraction import PROVIDERS, ExtractionPort, PortSettings, ReasoningLevel
from extractor.invocation import (
    BadInvocation,
    HelpRequested,
    Invocation,
    MissingArguments,
    Resolution,
    SchemaListing,
    UnknownProvider,
    UnknownReasoningLevel,
    UnknownSchema,
    resolve,
)
from extractor.schemas import TermsOfService
from tests.staging import StagedProvider


def unusable_port(_settings: PortSettings) -> ExtractionPort:
    """A tripwire: resolution decides, it does not act, so constructing a port is the bug."""
    raise AssertionError("resolution must not construct a port")


PROVIDER_NAME = "openai"
STAGED = {PROVIDER_NAME: StagedProvider(PROVIDERS[PROVIDER_NAME].default_model, unusable_port)}


def resolved(argv: Sequence[str]) -> Resolution:
    return resolve(argv, STAGED)


def as_invocation(argv: Sequence[str]) -> Invocation:
    outcome = resolved(argv)
    assert isinstance(outcome, Invocation), outcome
    return outcome


def test_a_complete_command_line_resolves_to_an_invocation() -> None:
    invocation = as_invocation(["--schema", "tos", "terms.html"])

    assert invocation.schema is TermsOfService
    assert invocation.source == "terms.html"
    assert invocation.settings == PortSettings(
        PROVIDERS[PROVIDER_NAME].default_model, ReasoningLevel.MEDIUM, None
    )


def test_an_omitted_model_takes_the_selected_providers_default() -> None:
    invocation = as_invocation(["--schema", "tos", "-"])

    assert invocation.settings.model_id == PROVIDERS[PROVIDER_NAME].default_model


def test_the_model_flag_overrides_the_providers_default() -> None:
    invocation = as_invocation(["--model", "pinned-model", "--schema", "tos", "-"])

    assert invocation.settings.model_id == "pinned-model"


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("off", ReasoningLevel.OFF),
        ("low", ReasoningLevel.LOW),
        ("medium", ReasoningLevel.MEDIUM),
        ("high", ReasoningLevel.HIGH),
    ],
)
def test_each_reasoning_level_reaches_the_port_settings(
    argument: str, expected: ReasoningLevel
) -> None:
    invocation = as_invocation(["--reasoning", argument, "--schema", "tos", "-"])

    assert invocation.settings.reasoning is expected


def test_debug_selects_stderr_as_the_dump_stream() -> None:
    invocation = as_invocation(["--debug", "--schema", "tos", "-"])

    assert invocation.settings.debug is sys.stderr


def test_the_stream_is_read_when_resolving_not_captured_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream captured at import would survive any later replacement, so a test that
    redirects stderr would silently keep writing to the real one."""
    replacement = StringIO()
    monkeypatch.setattr(sys, "stderr", replacement)

    invocation = as_invocation(["--debug", "--schema", "tos", "-"])

    assert invocation.settings.debug is replacement


def test_without_debug_no_dump_stream_is_selected() -> None:
    invocation = as_invocation(["--schema", "tos", "-"])

    assert invocation.settings.debug is None


def test_listing_schemas_needs_no_input_or_schema() -> None:
    assert resolved(["--list-schemas"]) == SchemaListing(names=["tos"])


def test_an_unknown_provider_is_named_with_the_valid_ones() -> None:
    assert resolved(["--provider", "unknown", "--schema", "tos", "-"]) == UnknownProvider(
        name="unknown", valid=[PROVIDER_NAME]
    )


def test_an_unknown_reasoning_level_is_named_with_the_valid_ones() -> None:
    assert resolved(["--reasoning", "extreme", "--schema", "tos", "-"]) == UnknownReasoningLevel(
        name="extreme", valid=["off", "low", "medium", "high"]
    )


def test_an_unknown_schema_is_named_with_the_valid_ones() -> None:
    assert resolved(["--schema", "contract", "-"]) == UnknownSchema(name="contract", valid=["tos"])


def test_a_missing_schema_or_input_is_reported_before_a_schema_lookup() -> None:
    assert resolved(["-"]) == MissingArguments()
    assert resolved(["--schema", "tos"]) == MissingArguments()


def test_help_is_not_a_failure() -> None:
    assert resolved(["--help"]) == HelpRequested()


def test_an_argument_the_parser_rejects_is_a_bad_invocation() -> None:
    assert resolved(["--unknown-flag"]) == BadInvocation()


def test_a_bad_provider_beside_list_schemas_is_reported_not_ignored() -> None:
    """The value checks run before the short-circuit, so `--list-schemas` cannot mask a
    command line that would have failed without it."""
    outcome = resolved(["--provider", "unknown", "--list-schemas"])

    assert outcome == UnknownProvider(name="unknown", valid=[PROVIDER_NAME])


@pytest.mark.parametrize(
    "argv",
    [
        ["--provider", "unknown", "--schema", "tos", "-"],
        ["--reasoning", "extreme", "--schema", "tos", "-"],
        ["--schema", "contract", "-"],
        ["--schema", "tos", "-"],
        ["--list-schemas"],
    ],
)
def test_resolution_never_constructs_a_port(argv: list[str]) -> None:
    """The staged provider raises on construction, so any command line that reached it fails
    here. Resolution decides; `main` acts on what it decided."""
    resolved(argv)
