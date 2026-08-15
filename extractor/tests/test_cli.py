import json
import sys
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import NamedTuple

import pytest
from pydantic import BaseModel, ValidationError

from extractor.__main__ import ExitCode, main
from extractor.extraction import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    REASONING_LEVELS,
    EmptyExtraction,
    Extracted,
    Extraction,
    ExtractionPort,
    PortSettings,
    ProviderFailure,
    ProviderRejectedRequest,
    ReasoningLevel,
    Refusal,
    ValidationFailure,
)
from extractor.schemas import TermsOfService
from tests.staging import PortFactory, StagedProvider


class CliResult(NamedTuple):
    exit_code: ExitCode
    stdout: str
    stderr: str


class StagedPort:
    """A port factory yielding one prepared outcome, recording how `main` wired it up."""

    def __init__(self, outcome: Extraction) -> None:
        self.outcome = outcome
        self.documents: list[str] = []
        self.settings: list[PortSettings] = []

    def __call__(self, settings: PortSettings) -> ExtractionPort:
        self.settings.append(settings)

        def extract(document: str, schema: type[BaseModel]) -> Extraction:
            self.documents.append(document)
            return self.outcome

        return extract


class PortCalled(BaseException):
    """The tripwire fired. Not an `Exception`, so `main` cannot absorb it into exit 1."""


def failing_port(error: BaseException) -> PortFactory:
    """A port factory whose port always raises `error`."""

    def factory(_settings: PortSettings) -> ExtractionPort:
        def extract(document: str, schema: type[BaseModel]) -> Extraction:
            raise error

        return extract

    return factory


def tripwire_port(_settings: PortSettings) -> ExtractionPort:
    """A tripwire: reaching the provider at all — even constructing it — is the bug under test."""
    raise PortCalled("the extraction port must not be constructed")


def schema_rejection_detail(candidate: object) -> str:
    """The real schema-rejection text for a candidate object, so assertions stay honest."""
    try:
        TermsOfService.model_validate(candidate)
    except ValidationError as error:
        return str(error)
    raise AssertionError("the candidate object was accepted by the schema")


def run_cli(
    capsys: pytest.CaptureFixture[str],
    argv: Sequence[str],
    *,
    source: str = "",
    port_factory: PortFactory = tripwire_port,
    provider: str = DEFAULT_PROVIDER,
) -> CliResult:
    exit_code = main(
        argv,
        stdin=StringIO(source),
        providers={provider: StagedProvider(PROVIDERS[provider].default_model, port_factory)},
    )
    captured = capsys.readouterr()
    return CliResult(exit_code, captured.out, captured.err)


def test_a_valid_extraction_is_the_only_stdout_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "governing_law": "State of New York",
        "arbitration_required": True,
        "arbitration_clause": "Disputes must be resolved by binding arbitration.",
        "liability_cap": "$100",
        "termination_notice_period": "30 days",
        "data_retention_period": None,
        "effective_date": "2026-01-01",
    }

    staged = StagedPort(Extracted(TermsOfService.model_validate(expected)))
    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=staged,
    )

    assert result == CliResult(ExitCode.OK, json.dumps(expected, separators=(",", ":")) + "\n", "")
    assert staged.documents == ["Terms of Service source"]
    assert staged.settings == [
        PortSettings(PROVIDERS[DEFAULT_PROVIDER].default_model, ReasoningLevel.MEDIUM, None)
    ]


def test_an_invalid_model_value_is_a_validation_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = {
        "governing_law": None,
        "arbitration_required": None,
        "arbitration_clause": None,
        "liability_cap": None,
        "termination_notice_period": None,
        "data_retention_period": None,
        "effective_date": "not-a-date",
    }

    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=StagedPort(ValidationFailure(detail=schema_rejection_detail(invalid))),
    )

    assert result.exit_code == ExitCode.VALIDATION_FAILURE
    assert result.stdout == ""
    assert "validation failure" in result.stderr.lower()
    assert "effective_date" in result.stderr


def test_a_model_answer_without_an_object_is_an_empty_extraction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=StagedPort(EmptyExtraction()),
    )

    assert result.exit_code == ExitCode.EMPTY_EXTRACTION
    assert result.stdout == ""
    assert "empty extraction" in result.stderr.lower()


def test_a_refusal_outcome_is_reported_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=StagedPort(Refusal(detail="The provider declined this extraction.")),
    )

    assert result.exit_code == ExitCode.REFUSAL
    assert result.stdout == ""
    assert "refusal" in result.stderr.lower()
    assert "declined" in result.stderr.lower()


def test_a_provider_failure_is_reported_as_retryable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=StagedPort(ProviderFailure(detail="Error code: 429 - quota exhausted")),
    )

    assert result.exit_code == ExitCode.PROVIDER_FAILURE
    assert result.stdout == ""
    assert result.stderr == "Provider failure: Error code: 429 - quota exhausted\n"


def test_a_provider_rejected_request_is_reported_as_non_retryable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=StagedPort(
            ProviderRejectedRequest(detail="Error code: 404 - model does not exist")
        ),
    )

    assert result.exit_code == ExitCode.PROVIDER_REJECTED_REQUEST
    assert result.stdout == ""
    assert result.stderr == "Provider-rejected request: Error code: 404 - model does not exist\n"


def test_an_unexpected_provider_error_uses_the_generic_failure_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=failing_port(ConnectionError("network unavailable")),
    )

    assert result.exit_code == ExitCode.FAILURE
    assert result.stdout == ""
    assert "unexpected error" in result.stderr.lower()
    assert "network unavailable" in result.stderr.lower()


def test_debug_directs_the_raw_message_dump_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    extracted = {
        "governing_law": None,
        "arbitration_required": None,
        "arbitration_clause": None,
        "liability_cap": None,
        "termination_notice_period": None,
        "data_retention_period": None,
        "effective_date": None,
    }
    staged = StagedPort(Extracted(TermsOfService.model_validate(extracted)))

    result = run_cli(
        capsys,
        ["--debug", "--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=staged,
    )

    assert result.exit_code == ExitCode.OK
    assert json.loads(result.stdout) == extracted
    # `main` owns only the wiring; `test_extraction.py` covers what the adapter writes there.
    assert staged.settings[-1].debug is sys.stderr


def test_a_source_file_is_read_for_extraction(capsys: pytest.CaptureFixture[str]) -> None:
    expected = {
        "governing_law": "State of New York",
        "arbitration_required": True,
        "arbitration_clause": "Binding arbitration administered by the AAA.",
        "liability_cap": "USD 100",
        "termination_notice_period": "30 days",
        "data_retention_period": None,
        "effective_date": "2026-01-01",
    }
    fixture = Path(__file__).parent / "fixtures" / "terms.html"

    staged = StagedPort(Extracted(TermsOfService.model_validate(expected)))
    result = run_cli(capsys, ["--schema", "tos", str(fixture)], port_factory=staged)

    assert result.exit_code == ExitCode.OK
    assert json.loads(result.stdout) == expected
    assert result.stderr == ""
    assert staged.documents == [fixture.read_text(encoding="utf-8")]


def test_listing_schemas_needs_no_input_or_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_cli(capsys, ["--list-schemas"]) == CliResult(ExitCode.OK, "tos\n", "")


def test_an_unknown_schema_name_lists_valid_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys, ["--schema", "contract", "-"], source="source", port_factory=tripwire_port
    )

    assert result.exit_code == ExitCode.FAILURE
    assert result.stdout == ""
    assert "unknown schema" in result.stderr.lower()
    assert "tos" in result.stderr


def test_a_missing_input_file_is_reported_as_an_input_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "missing.html"

    result = run_cli(capsys, ["--schema", "tos", str(missing)], port_factory=tripwire_port)

    assert result.exit_code == ExitCode.FAILURE
    assert result.stdout == ""
    # `main` owns this wording; `intake` returns the facts only.
    assert result.stderr == f"Input file error for {str(missing)!r}: No such file or directory.\n"


def test_an_unreadable_input_path_is_reported_as_an_input_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    result = run_cli(capsys, ["--schema", "tos", str(tmp_path)], port_factory=tripwire_port)

    assert result.exit_code == ExitCode.FAILURE
    assert result.stdout == ""
    assert "input file error" in result.stderr.lower()
    assert str(tmp_path) in result.stderr


def test_a_missing_api_key_is_a_named_configuration_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("extractor.credentials.ENV_FILE", tmp_path / "absent.env")

    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="source",
        port_factory=PROVIDERS[DEFAULT_PROVIDER].build_port,
    )

    assert result.exit_code == ExitCode.FAILURE
    assert result.stdout == ""
    assert "configuration error" in result.stderr.lower()
    assert "openai_api_key" in result.stderr.lower()


def test_an_oversize_document_is_refused_before_calling_the_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = "x" * 100_001

    result = run_cli(capsys, ["--schema", "tos", "-"], source=document, port_factory=tripwire_port)

    assert result.exit_code == ExitCode.FAILURE
    assert result.stdout == ""
    assert result.stderr == (
        "Document too large: maximum is 100,000 characters; received 100,001 characters.\n"
    )


def test_the_documented_exit_numbers_are_pinned() -> None:
    """`README.md` publishes these numbers as the CLI contract; renumbering breaks here."""
    assert {member.name: member.value for member in ExitCode} == {
        "OK": 0,
        "FAILURE": 1,
        "VALIDATION_FAILURE": 2,
        "EMPTY_EXTRACTION": 3,
        "REFUSAL": 4,
        "PROVIDER_FAILURE": 5,
        "PROVIDER_REJECTED_REQUEST": 6,
    }
    # `IntEnum`, not `Enum`: `raise SystemExit(main())` hands a member straight to the
    # process status. Downgrading the base breaks that silently — nothing else fails.
    assert isinstance(ExitCode.REFUSAL, int)


def test_each_provider_registers_its_published_default_model() -> None:
    """`README.md` and the architecture document publish these; the registry is their source."""
    assert {name: provider.default_model for name, provider in PROVIDERS.items()} == {
        "openai": "gpt-5-nano",
        "anthropic": "claude-sonnet-5",
        "openrouter": "openai/gpt-5-nano",
    }


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_an_omitted_model_flag_takes_the_selected_providers_own_default(
    capsys: pytest.CaptureFixture[str], provider: str
) -> None:
    staged = StagedPort(EmptyExtraction())

    run_cli(
        capsys,
        ["--provider", provider, "--schema", "tos", "-"],
        source="source",
        port_factory=staged,
        provider=provider,
    )

    assert staged.settings[-1].model_id == PROVIDERS[provider].default_model


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_the_model_flag_overrides_the_default_on_every_provider(
    capsys: pytest.CaptureFixture[str], provider: str
) -> None:
    """A document the default cannot handle is the reason the flag exists; nothing else in the
    suite notices if the parsed value never reaches the port factory."""
    staged = StagedPort(EmptyExtraction())

    run_cli(
        capsys,
        ["--provider", provider, "--model", "pinned-model", "--schema", "tos", "-"],
        source="source",
        port_factory=staged,
        provider=provider,
    )

    assert staged.settings[-1].model_id == "pinned-model"


def test_an_explicit_openai_provider_matches_the_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    implicit = StagedPort(EmptyExtraction())
    explicit = StagedPort(EmptyExtraction())

    implicit_result = run_cli(
        capsys, ["--schema", "tos", "-"], source="source", port_factory=implicit
    )
    explicit_result = run_cli(
        capsys,
        ["--provider", DEFAULT_PROVIDER, "--schema", "tos", "-"],
        source="source",
        port_factory=explicit,
    )

    assert explicit_result == implicit_result
    assert explicit.settings == implicit.settings


def test_an_unknown_provider_lists_valid_names_without_constructing_a_port(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--provider", "unknown", "--schema", "tos", "-"],
        source="source",
        port_factory=tripwire_port,
    )

    assert result.exit_code is ExitCode.FAILURE
    assert result.stdout == ""
    assert "valid providers" in result.stderr.lower()
    assert DEFAULT_PROVIDER in result.stderr


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
    capsys: pytest.CaptureFixture[str],
    argument: str,
    expected: ReasoningLevel,
) -> None:
    staged = StagedPort(EmptyExtraction())

    run_cli(
        capsys,
        ["--reasoning", argument, "--schema", "tos", "-"],
        source="source",
        port_factory=staged,
    )

    assert staged.settings[-1].reasoning is expected


def test_an_unknown_reasoning_level_lists_valid_values_without_constructing_a_port(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--reasoning", "extreme", "--schema", "tos", "-"],
        source="source",
        port_factory=tripwire_port,
    )

    assert result.exit_code is ExitCode.FAILURE
    assert result.stdout == ""
    assert "valid levels" in result.stderr.lower()
    assert all(level in result.stderr for level in REASONING_LEVELS)


@pytest.mark.parametrize(
    "argv",
    [
        ["--unknown-flag"],
        ["--schema"],
    ],
)
def test_a_bad_invocation_uses_the_generic_failure_status(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    result = run_cli(capsys, argv, port_factory=tripwire_port)

    assert result.exit_code is ExitCode.FAILURE
    assert result.stdout == ""


def test_help_preserves_argparses_success_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(capsys, ["--help"], port_factory=tripwire_port)

    assert result.exit_code is ExitCode.OK
    assert result.stdout.startswith("usage:")
    assert result.stderr == ""
