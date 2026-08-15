from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

MAX_DOCUMENT_CHARACTERS = 100_000


@dataclass(frozen=True, slots=True)
class UnreadableSource:
    """The spec named something intake could not read as UTF-8 text."""

    spec: str
    detail: str


@dataclass(frozen=True, slots=True)
class OversizeDocument:
    """The source read cleanly, but it exceeds the character ceiling."""

    characters: int
    ceiling: int


type IntakeFailure = UnreadableSource | OversizeDocument
"""Why there is no document to extract from. Named once here, so a new member reaches
`__main__`'s match as a type error rather than only as a widened call argument."""

type Intake = str | IntakeFailure


def _detail(error: OSError | UnicodeDecodeError) -> str:
    """The readable half of a read failure; `strerror` is `None` for some `OSError`s."""
    if isinstance(error, OSError):
        return error.strerror or str(error)
    return str(error)


def load_source_document(spec: str, stdin: TextIO) -> Intake:
    """Read the source document named by `spec`, or name why there is none to extract from.

    `spec` is `-` for stdin, otherwise a filesystem path read as UTF-8. A returned `str` has
    already cleared the character ceiling, so no caller can proceed with an oversize document.
    Failures carry their facts rather than prose; `__main__` renders them.
    """
    try:
        document = stdin.read() if spec == "-" else Path(spec).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return UnreadableSource(spec=spec, detail=_detail(error))
    if len(document) > MAX_DOCUMENT_CHARACTERS:
        return OversizeDocument(characters=len(document), ceiling=MAX_DOCUMENT_CHARACTERS)
    return document
