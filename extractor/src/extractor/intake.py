from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

MAX_DOCUMENT_CHARACTERS = 100_000

STDIN_SPEC = "-"


@dataclass(frozen=True, slots=True)
class InputFailure:
    """Intake refused the spec. `message` is the rendered report, with no trailing newline."""

    message: str


def _detail(error: OSError | UnicodeDecodeError) -> str:
    """The readable half of a read failure; `strerror` is `None` for some `OSError`s."""
    if isinstance(error, OSError):
        return error.strerror or str(error)
    return str(error)


def load_source_document(spec: str, stdin: TextIO) -> str | InputFailure:
    """Read the source document named by `spec`, or name why there is none to extract from.

    `spec` is `-` for stdin, otherwise a filesystem path read as UTF-8. A returned `str` has
    already cleared the character ceiling, so no caller can proceed with an oversize document.
    """
    try:
        document = stdin.read() if spec == STDIN_SPEC else Path(spec).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return InputFailure(message=f"Input file error for {spec!r}: {_detail(error)}.")
    document_size = len(document)
    if document_size > MAX_DOCUMENT_CHARACTERS:
        return InputFailure(
            message=(
                "Document too large: "
                f"maximum is {MAX_DOCUMENT_CHARACTERS:,} characters; "
                f"received {document_size:,} characters."
            )
        )
    return document
