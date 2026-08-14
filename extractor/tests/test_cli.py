import json
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import NamedTuple, TextIO

import pytest
from pydantic import BaseModel, ValidationError

from extractor.__main__ import main
from extractor.extraction import (
    EmptyExtraction,
    Extracted,
    Extraction,
    ExtractionPort,
    PortFactory,
    Refusal,
    ValidationFailure,
    build_openai_port,
)
from extractor.schemas import TermsOfService


class CliResult(NamedTuple):
    exit_code: int
    stdout: str
    stderr: str


def staged_port(outcome: Extraction, *, raw: str = "") -> PortFactory:
    """A port factory whose port always yields `outcome`, dumping `raw` when debugging."""

    def factory(model_id: str, debug: TextIO | None) -> ExtractionPort:
        def extract(document: str, schema: type[BaseModel]) -> Extraction:
            if debug is not None:
                debug.write(f"Raw model message: {raw!r}\n")
            return outcome

        return extract

    return factory


def failing_port(error: BaseException) -> PortFactory:
    """A port factory whose port always raises `error`."""

    def factory(model_id: str, debug: TextIO | None) -> ExtractionPort:
        def extract(document: str, schema: type[BaseModel]) -> Extraction:
            raise error

        return extract

    return factory


UNUSED_PORT: PortFactory = failing_port(AssertionError("the provider must not be called"))


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
    port_factory: PortFactory = UNUSED_PORT,
) -> CliResult:
    exit_code = main(argv, stdin=StringIO(source), port_factory=port_factory)
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

    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=staged_port(Extracted(TermsOfService.model_validate(expected))),
    )

    assert result == CliResult(0, json.dumps(expected, separators=(",", ":")) + "\n", "")


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
        port_factory=staged_port(ValidationFailure(detail=schema_rejection_detail(invalid))),
    )

    assert result.exit_code == 2
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
        port_factory=staged_port(EmptyExtraction()),
    )

    assert result.exit_code == 3
    assert result.stdout == ""
    assert "empty extraction" in result.stderr.lower()


def test_a_provider_refusal_from_raw_parsing_is_reported_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=staged_port(Refusal(detail="The provider declined this extraction.")),
    )

    assert result.exit_code == 4
    assert result.stdout == ""
    assert "refusal" in result.stderr.lower()
    assert "declined" in result.stderr.lower()


def test_an_unexpected_provider_error_uses_the_generic_failure_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=failing_port(ConnectionError("network unavailable")),
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "unexpected error" in result.stderr.lower()
    assert "network unavailable" in result.stderr.lower()


def test_debug_dumps_the_raw_model_message_to_stderr(
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

    result = run_cli(
        capsys,
        ["--debug", "--schema", "tos", "-"],
        source="Terms of Service source",
        port_factory=staged_port(
            Extracted(TermsOfService.model_validate(extracted)), raw="raw provider message"
        ),
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == extracted
    assert "raw provider message" in result.stderr


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

    result = run_cli(
        capsys,
        ["--schema", "tos", str(fixture)],
        port_factory=staged_port(Extracted(TermsOfService.model_validate(expected))),
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == expected
    assert result.stderr == ""


def test_listing_schemas_needs_no_input_or_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_cli(capsys, ["--list-schemas"]) == CliResult(0, "tos\n", "")


def test_an_unknown_schema_name_lists_valid_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(capsys, ["--schema", "contract", "-"], source="source")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "unknown schema" in result.stderr.lower()
    assert "tos" in result.stderr


def test_a_missing_input_file_is_reported_as_an_input_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "missing.html"

    result = run_cli(capsys, ["--schema", "tos", str(missing)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "input file" in result.stderr.lower()
    assert str(missing) in result.stderr
    assert "validation failure" not in result.stderr.lower()


def test_an_unreadable_input_path_is_reported_as_an_input_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    result = run_cli(capsys, ["--schema", "tos", str(tmp_path)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "input file error" in result.stderr.lower()
    assert str(tmp_path) in result.stderr


def test_a_missing_api_key_is_a_named_configuration_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("extractor.extraction.load_dotenv", lambda _: False)

    result = run_cli(
        capsys,
        ["--schema", "tos", "-"],
        source="source",
        port_factory=build_openai_port,
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "configuration error" in result.stderr.lower()
    assert "openai_api_key" in result.stderr.lower()


def test_an_oversize_document_is_refused_before_calling_the_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = "x" * 100_001

    result = run_cli(capsys, ["--schema", "tos", "-"], source=document)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "100,000" in result.stderr
    assert "100,001" in result.stderr
