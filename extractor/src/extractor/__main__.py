import sys
from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import TextIO, assert_never

from extractor.credentials import ConfigurationError
from extractor.extraction import (
    PROVIDERS,
    EmptyExtraction,
    Extracted,
    Extraction,
    Provider,
    ProviderFailure,
    ProviderRejectedRequest,
    Refusal,
    ValidationFailure,
)
from extractor.intake import (
    IntakeFailure,
    OversizeDocument,
    UnreadableSource,
    load_source_document,
)
from extractor.invocation import (
    BadInvocation,
    HelpRequested,
    Invocation,
    InvocationFailure,
    MissingArguments,
    SchemaListing,
    UnknownProvider,
    UnknownReasoningLevel,
    UnknownSchema,
    resolve,
)


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


def _report_invocation(failure: InvocationFailure) -> ExitCode:
    """Write an invocation failure's diagnostic and return the exit code it earns.

    All members share `FAILURE`, per the `ExitCode` docstring. `BadInvocation` writes nothing:
    the argument parser has already named the offending argument on stderr, and repeating it
    would print the same fault twice.
    """
    match failure:
        case BadInvocation():
            pass
        case MissingArguments():
            sys.stderr.write("Input error: --schema and an input path are required.\n")
        case UnknownProvider(name=name, valid=valid):
            sys.stderr.write(f"Unknown provider {name!r}. Valid providers: {', '.join(valid)}.\n")
        case UnknownReasoningLevel(name=name, valid=valid):
            sys.stderr.write(
                f"Unknown reasoning level {name!r}. Valid levels: {', '.join(valid)}.\n"
            )
        case UnknownSchema(name=name, valid=valid):
            sys.stderr.write(f"Unknown schema {name!r}. Valid schemas: {', '.join(valid)}.\n")
        case unreachable:
            assert_never(unreachable)
    return ExitCode.FAILURE


def _extract(invocation: Invocation, stdin: TextIO) -> ExitCode:
    """Run one resolved invocation: read the document, call the provider, report the outcome.

    Everything here is already validated, so this reads as the happy path it is. The port is
    constructed after intake, so an oversize document costs nothing.
    """
    document = load_source_document(invocation.source, stdin)
    if not isinstance(document, str):
        return _report_intake(document)
    try:
        extract = invocation.provider.build_port(invocation.settings)
        outcome = extract(document, invocation.schema)
    except ConfigurationError as error:
        sys.stderr.write(f"Configuration error: {error}\n")
        return ExitCode.FAILURE
    except Exception as error:
        sys.stderr.write(f"Unexpected error: {error}\n")
        return ExitCode.FAILURE
    return _report(outcome)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO = sys.stdin,
    providers: Mapping[str, Provider] = PROVIDERS,
) -> ExitCode:
    match resolve(argv, providers):
        case HelpRequested():
            return ExitCode.OK
        case SchemaListing(names=names):
            sys.stdout.write("\n".join(names) + "\n")
            return ExitCode.OK
        case Invocation() as invocation:
            return _extract(invocation, stdin)
        case failure:
            return _report_invocation(failure)


if __name__ == "__main__":
    raise SystemExit(main())
