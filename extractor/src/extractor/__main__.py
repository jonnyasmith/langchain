import argparse
import sys
from collections.abc import Sequence
from enum import IntEnum
from typing import TextIO, assert_never

from extractor.extraction import (
    ConfigurationError,
    EmptyExtraction,
    Extracted,
    Extraction,
    PortFactory,
    ProviderFailure,
    ProviderRejectedRequest,
    Refusal,
    ValidationFailure,
    build_openai_port,
)
from extractor.intake import (
    IntakeFailure,
    OversizeDocument,
    UnreadableSource,
    load_source_document,
)
from extractor.schemas import SCHEMAS


class ExitCode(IntEnum):
    """The CLI's documented exit statuses. `README.md` publishes these numbers.

    `FAILURE` is the shared status for everything that is not an `Extraction` outcome:
    a bad invocation, an unreadable or oversize document, missing configuration, and
    any unexpected error. The outcome statuses are assigned by `_report`.
    """

    OK = 0
    FAILURE = 1
    VALIDATION_FAILURE = 2
    EMPTY_EXTRACTION = 3
    REFUSAL = 4
    PROVIDER_FAILURE = 5
    PROVIDER_REJECTED_REQUEST = 6


DEFAULT_MODEL = "gpt-5-nano"


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


def _report(outcome: Extraction) -> ExitCode:
    """Write an outcome's own output and return the exit code it earns.

    Every `Extraction` member's status and its diagnostic live here, together, and
    nowhere else. `assert_never` makes a new union member a type error, not a silent
    fall-through to an unnamed code.
    """
    match outcome:
        case Extracted(value=value):
            sys.stdout.write(value.model_dump_json() + "\n")
            return ExitCode.OK
        case ValidationFailure(detail=detail):
            sys.stderr.write(f"Validation failure: {detail}\n")
            return ExitCode.VALIDATION_FAILURE
        case EmptyExtraction():
            sys.stderr.write("Empty extraction: the model returned no object.\n")
            return ExitCode.EMPTY_EXTRACTION
        case Refusal(detail=detail):
            sys.stderr.write(f"Refusal: {detail}\n")
            return ExitCode.REFUSAL
        case ProviderFailure(detail=detail):
            sys.stderr.write(f"Provider failure: {detail}\n")
            return ExitCode.PROVIDER_FAILURE
        case ProviderRejectedRequest(detail=detail):
            sys.stderr.write(f"Provider-rejected request: {detail}\n")
            return ExitCode.PROVIDER_REJECTED_REQUEST
        case unreachable:
            assert_never(unreachable)


def _report_intake(failure: IntakeFailure) -> ExitCode:
    """Write an intake failure's diagnostic and return the exit code it earns.

    Both members share `FAILURE`, per the `ExitCode` docstring; only the wording differs.
    `assert_never` makes a new intake failure a type error rather than a silent no-op.
    """
    match failure:
        case UnreadableSource(spec=spec, detail=detail):
            sys.stderr.write(f"Input file error for {spec!r}: {detail}.\n")
        case OversizeDocument(characters=characters, ceiling=ceiling):
            sys.stderr.write(
                f"Document too large: maximum is {ceiling:,} characters; "
                f"received {characters:,} characters.\n"
            )
        case unreachable:
            assert_never(unreachable)
    return ExitCode.FAILURE


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO = sys.stdin,
    port_factory: PortFactory = build_openai_port,
) -> ExitCode:
    args = _parser().parse_args(argv)
    if args.list_schemas:
        sys.stdout.write("\n".join(sorted(SCHEMAS)) + "\n")
        return ExitCode.OK
    if args.schema is None or args.input is None:
        sys.stderr.write("Input error: --schema and an input path are required.\n")
        return ExitCode.FAILURE
    schema = SCHEMAS.get(args.schema)
    if schema is None:
        valid_names = ", ".join(sorted(SCHEMAS))
        sys.stderr.write(f"Unknown schema {args.schema!r}. Valid schemas: {valid_names}.\n")
        return ExitCode.FAILURE
    document = load_source_document(args.input, stdin)
    if not isinstance(document, str):
        return _report_intake(document)
    try:
        extract = port_factory(args.model, sys.stderr if args.debug else None)
        outcome = extract(document, schema)
    except ConfigurationError as error:
        sys.stderr.write(f"Configuration error: {error}\n")
        return ExitCode.FAILURE
    except Exception as error:
        sys.stderr.write(f"Unexpected error: {error}\n")
        return ExitCode.FAILURE
    return _report(outcome)


if __name__ == "__main__":
    raise SystemExit(main())
