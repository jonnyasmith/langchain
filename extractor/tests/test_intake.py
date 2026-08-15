from io import StringIO
from pathlib import Path

from extractor.intake import (
    MAX_DOCUMENT_CHARACTERS,
    OversizeDocument,
    UnreadableSource,
    load_source_document,
)


def test_a_dash_spec_reads_the_whole_of_stdin() -> None:
    assert load_source_document("-", StringIO("piped source")) == "piped source"


def test_a_path_spec_reads_the_file_as_utf8(tmp_path: Path) -> None:
    source = tmp_path / "terms.html"
    source.write_text("<p>café — naïve</p>", encoding="utf-8")

    assert load_source_document(str(source), StringIO("stdin must be ignored")) == (
        "<p>café — naïve</p>"
    )


def test_a_missing_path_is_an_unreadable_source(tmp_path: Path) -> None:
    missing = tmp_path / "missing.html"

    failure = load_source_document(str(missing), StringIO(""))

    assert failure == UnreadableSource(spec=str(missing), detail="No such file or directory")


def test_an_unreadable_path_is_an_unreadable_source(tmp_path: Path) -> None:
    failure = load_source_document(str(tmp_path), StringIO(""))

    assert isinstance(failure, UnreadableSource)
    assert failure.spec == str(tmp_path)
    assert failure.detail


def test_a_non_utf8_file_is_an_unreadable_source_not_an_unexpected_error(tmp_path: Path) -> None:
    latin1 = tmp_path / "terms.html"
    latin1.write_bytes(b"<p>caf\xe9</p>")

    failure = load_source_document(str(latin1), StringIO(""))

    assert isinstance(failure, UnreadableSource)
    assert failure.spec == str(latin1)
    assert "utf-8" in failure.detail


def test_a_document_exactly_at_the_ceiling_is_accepted() -> None:
    document = "x" * MAX_DOCUMENT_CHARACTERS

    assert load_source_document("-", StringIO(document)) == document


def test_one_character_over_the_ceiling_is_an_oversize_document() -> None:
    failure = load_source_document("-", StringIO("x" * (MAX_DOCUMENT_CHARACTERS + 1)))

    assert failure == OversizeDocument(
        characters=MAX_DOCUMENT_CHARACTERS + 1, ceiling=MAX_DOCUMENT_CHARACTERS
    )


def test_the_ceiling_counts_characters_after_decoding(tmp_path: Path) -> None:
    """Multi-byte characters count once each, so byte length is not the measure."""
    source = tmp_path / "wide.txt"
    source.write_text("é" * MAX_DOCUMENT_CHARACTERS, encoding="utf-8")

    assert load_source_document(str(source), StringIO("")) == "é" * MAX_DOCUMENT_CHARACTERS
