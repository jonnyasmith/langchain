"""Turn one command line into a resolved invocation, or name why it is not one.

Argument validity was the last thing in this module that was a branch rather than a value: six
checks interleaved with the work they guard, each writing its own diagnostic on the spot. Here
they are one call returning one closed union, so `__main__` matches it the same way it matches
an extraction outcome, and the compiler proves it handled every case.

Failures carry their facts, not their prose. `__main__` renders them, the same split `intake`
and `extraction` already use.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from extractor.extraction import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    REASONING_LEVELS,
    PortSettings,
    Provider,
    ReasoningLevel,
)
from extractor.schemas import SCHEMAS


@dataclass(frozen=True, slots=True)
class Invocation:
    """A resolved command line: everything one extraction run needs, already checked.

    Holding the selected `Provider` and the built `PortSettings` rather than the raw strings is
    what makes this a resolution: no caller can reach a provider name that was never validated.
    """

    provider: Provider
    settings: PortSettings
    schema: type[BaseModel]
    source: str


@dataclass(frozen=True, slots=True)
class SchemaListing:
    """`--list-schemas`: the operator asked which schemas exist, not for an extraction."""

    names: list[str]


@dataclass(frozen=True, slots=True)
class HelpRequested:
    """The parser wrote usage and exited successfully. Not a failure."""


@dataclass(frozen=True, slots=True)
class BadInvocation:
    """The parser rejected the command line and has already written its own message.

    Carries no detail because there is nothing left to say: argparse names the offending
    argument itself, and repeating it would print the same fault twice.
    """


@dataclass(frozen=True, slots=True)
class MissingArguments:
    """An extraction needs both `--schema` and an input path, and one of them is absent."""


@dataclass(frozen=True, slots=True)
class UnknownProvider:
    """A `--provider` the registry does not hold."""

    name: str
    valid: list[str]


@dataclass(frozen=True, slots=True)
class UnknownReasoningLevel:
    """A `--reasoning` outside the provider-neutral levels."""

    name: str
    valid: list[str]


@dataclass(frozen=True, slots=True)
class UnknownSchema:
    """A `--schema` the registry does not hold."""

    name: str
    valid: list[str]


type InvocationFailure = (
    BadInvocation | MissingArguments | UnknownProvider | UnknownReasoningLevel | UnknownSchema
)
"""Why there is nothing to run. Named once here, so a new member reaches `__main__`'s match as
a type error rather than only as a widened call argument."""

type Resolution = Invocation | SchemaListing | HelpRequested | InvocationFailure


def parser() -> argparse.ArgumentParser:
    """The command line's shape. Public because `--help` is part of the published interface."""
    parser = argparse.ArgumentParser(description="Extract typed data from one source document.")
    parser.add_argument("input", nargs="?", help="source file path, or - to read stdin")
    parser.add_argument("--schema", help="named extraction schema")
    parser.add_argument("--model", help="provider model id; defaults to the provider's own")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="extraction provider")
    valid_reasoning = ", ".join(REASONING_LEVELS)
    parser.add_argument(
        "--reasoning",
        default=ReasoningLevel.MEDIUM.value,
        help=f"reasoning effort: {valid_reasoning}",
    )
    parser.add_argument("--list-schemas", action="store_true", help="list named schemas")
    parser.add_argument(
        "--debug", action="store_true", help="write the raw model message to stderr"
    )
    return parser


def resolve(
    argv: Sequence[str] | None,
    providers: Mapping[str, Provider] = PROVIDERS,
    schemas: Mapping[str, type[BaseModel]] = SCHEMAS,
) -> Resolution:
    """Resolve one command line, in the order the CLI's contract fixes.

    Every check that can fail runs before any that costs something: an unknown provider,
    reasoning level, or schema is reported without reading a document or constructing a port.
    `--list-schemas` still short-circuits after the value checks, so a bad `--provider` beside
    it is reported rather than silently ignored.
    """
    try:
        arguments = parser().parse_args(argv)
    except SystemExit as request:
        return HelpRequested() if request.code == 0 else BadInvocation()

    provider = providers.get(arguments.provider)
    if provider is None:
        return UnknownProvider(name=arguments.provider, valid=sorted(providers))

    reasoning = REASONING_LEVELS.get(arguments.reasoning)
    if reasoning is None:
        return UnknownReasoningLevel(name=arguments.reasoning, valid=list(REASONING_LEVELS))

    if arguments.list_schemas:
        return SchemaListing(names=sorted(schemas))

    if arguments.schema is None or arguments.input is None:
        return MissingArguments()

    schema = schemas.get(arguments.schema)
    if schema is None:
        return UnknownSchema(name=arguments.schema, valid=sorted(schemas))

    return Invocation(
        provider=provider,
        settings=PortSettings(
            model_id=arguments.model or provider.default_model,
            reasoning=reasoning,
            # Read now rather than captured at import, so a replaced stream is honoured.
            debug=sys.stderr if arguments.debug else None,
        ),
        schema=schema,
        source=arguments.input,
    )
