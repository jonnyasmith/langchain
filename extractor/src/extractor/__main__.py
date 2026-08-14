import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO, assert_never

from extractor.extraction import (
    ConfigurationError,
    EmptyExtraction,
    Extracted,
    PortFactory,
    Refusal,
    ValidationFailure,
    build_openai_port,
)
from extractor.schemas import SCHEMAS

DEFAULT_MODEL = "gpt-5-nano"
MAX_DOCUMENT_CHARACTERS = 100_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract typed data from one source document.")
    parser.add_argument("input", nargs="?", help="source file path, or - to read stdin")
    parser.add_argument("--schema", help="named extraction schema")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model id")
    parser.add_argument("--list-schemas", action="store_true", help="list named schemas")
    parser.add_argument(
        "--debug", action="store_true", help="write the raw model message to stderr"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO = sys.stdin,
    port_factory: PortFactory = build_openai_port,
) -> int:
    args = _parser().parse_args(argv)
    if args.list_schemas:
        sys.stdout.write("\n".join(sorted(SCHEMAS)) + "\n")
        return 0
    if args.schema is None or args.input is None:
        sys.stderr.write("Input error: --schema and an input path are required.\n")
        return 1
    schema = SCHEMAS.get(args.schema)
    if schema is None:
        valid_names = ", ".join(sorted(SCHEMAS))
        sys.stderr.write(f"Unknown schema {args.schema!r}. Valid schemas: {valid_names}.\n")
        return 1
    try:
        document = (
            stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        )
    except OSError as error:
        sys.stderr.write(f"Input file error for {args.input!r}: {error.strerror or error}.\n")
        return 1
    document_size = len(document)
    if document_size > MAX_DOCUMENT_CHARACTERS:
        sys.stderr.write(
            "Document too large: "
            f"maximum is {MAX_DOCUMENT_CHARACTERS:,} characters; "
            f"received {document_size:,} characters.\n"
        )
        return 1
    try:
        extract = port_factory(args.model, sys.stderr if args.debug else None)
        outcome = extract(document, schema)
    except ConfigurationError as error:
        sys.stderr.write(f"Configuration error: {error}\n")
        return 1
    except Exception as error:
        sys.stderr.write(f"Unexpected error: {error}\n")
        return 1
    match outcome:
        case Extracted(value=value):
            sys.stdout.write(value.model_dump_json() + "\n")
            return 0
        case ValidationFailure(detail=detail):
            sys.stderr.write(f"Validation failure: {detail}\n")
            return 2
        case EmptyExtraction():
            sys.stderr.write("Empty extraction: the model returned no object.\n")
            return 3
        case Refusal(detail=detail):
            sys.stderr.write(f"Refusal: {detail}\n")
            return 4
        case unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    raise SystemExit(main())
