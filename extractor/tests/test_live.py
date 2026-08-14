"""Paid opt-in check that the real provider completes strict extraction."""

import json
from pathlib import Path

import pytest

from extractor.__main__ import ExitCode, main
from extractor.schemas import TermsOfService

FIXTURE = Path(__file__).parent / "fixtures" / "terms.html"


@pytest.mark.live
def test_live_extraction_returns_a_validated_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real provider completes extraction through the entry-point seam."""
    exit_code = main(["--schema", "tos", str(FIXTURE)])

    captured = capsys.readouterr()
    assert exit_code is ExitCode.OK, captured.err
    assert captured.err == ""

    payload = json.loads(captured.out)
    TermsOfService.model_validate(payload)
