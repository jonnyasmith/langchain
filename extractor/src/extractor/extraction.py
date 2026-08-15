import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIRefusalError
from openai import APIError, BadRequestError, NotFoundError, UnprocessableEntityError
from pydantic import BaseModel


class ConfigurationError(Exception):
    """The extractor cannot construct its provider adapter."""


@dataclass(frozen=True, slots=True)
class Extracted:
    """The model returned one object that the schema accepts."""

    value: BaseModel


@dataclass(frozen=True, slots=True)
class EmptyExtraction:
    """The model answered but committed to no object."""


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    """The model returned an object that the schema rejects."""

    detail: str


@dataclass(frozen=True, slots=True)
class Refusal:
    """The model declined to extract at all."""

    detail: str


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """The provider could not serve a well-formed extraction request."""

    detail: str


@dataclass(frozen=True, slots=True)
class ProviderRejectedRequest:
    """The provider rejected the extraction request as malformed."""

    detail: str


type Extraction = (
    Extracted
    | EmptyExtraction
    | ValidationFailure
    | Refusal
    | ProviderFailure
    | ProviderRejectedRequest
)


class _RawStructuredOutput(TypedDict):
    """The envelope `with_structured_output(include_raw=True)` returns.

    `langchain_openai` types that call's result as a plain mapping, so this is the shape
    the one boundary `cast` asserts. Every key the adapter reads is checked against it.
    """

    raw: BaseMessage
    parsed: BaseModel | None
    parsing_error: BaseException | None


class ExtractionPort(Protocol):
    """One extraction attempt: document plus schema in, one named outcome out."""

    def __call__(self, document: str, schema: type[BaseModel]) -> Extraction: ...


type PortFactory = Callable[[str, TextIO | None], ExtractionPort]


def _load_env_file(path: Path) -> None:
    """Define any `KEY=value` the file declares that the environment does not already set.

    Ten lines rather than a dependency, per the coding standards. An exported variable always
    wins, so a stale file cannot shadow the shell. No interpolation, `export` prefixes, or
    multi-line values: the extractor reads one key.

    A file that exists but cannot be read is a misconfiguration, not an unexpected state, so
    it raises `ConfigurationError` rather than leaking an `OSError` into the top-level net.
    """
    if not path.is_file():
        return
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigurationError(f"cannot read {path}: {error}") from error
    for line in contents.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def build_openai_port(model_id: str, debug: TextIO | None) -> ExtractionPort:
    """Build the OpenAI-backed extraction port, or fail if it cannot be configured."""
    _load_env_file(Path(__file__).resolve().parents[2] / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ConfigurationError("missing OPENAI_API_KEY; set it in extractor/.env")
    model = ChatOpenAI(
        model=model_id,
        reasoning_effort="none",
        temperature=0,
        timeout=60,
        max_retries=2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract only facts stated in the source document. "
                "Do not infer or guess; use null when the source does not answer a field.",
            ),
            ("human", "{document}"),
        ]
    )

    def extract(document: str, schema: type[BaseModel]) -> Extraction:
        structured_model = model.with_structured_output(
            schema,
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        chain = prompt | structured_model
        try:
            result = cast(_RawStructuredOutput, chain.invoke({"document": document}))
        except OpenAIRefusalError as error:
            return Refusal(detail=str(error))
        except (BadRequestError, NotFoundError, UnprocessableEntityError) as error:
            return ProviderRejectedRequest(detail=str(error))
        except APIError as error:
            return ProviderFailure(detail=str(error))
        if debug is not None:
            debug.write(f"Raw model message: {result['raw']!r}\n")
        parsing_error = result["parsing_error"]
        if isinstance(parsing_error, OpenAIRefusalError):
            return Refusal(detail=str(parsing_error))
        if parsing_error is not None:
            return ValidationFailure(detail=str(parsing_error))
        parsed = result["parsed"]
        if parsed is None:
            return EmptyExtraction()
        return Extracted(value=parsed)

    return extract
