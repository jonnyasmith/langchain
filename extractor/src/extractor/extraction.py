import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIRefusalError
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


type Extraction = Extracted | EmptyExtraction | ValidationFailure | Refusal


class ExtractionPort(Protocol):
    """One extraction attempt: document plus schema in, one named outcome out."""

    def __call__(self, document: str, schema: type[BaseModel]) -> Extraction: ...


type PortFactory = Callable[[str, TextIO | None], ExtractionPort]


def build_openai_port(model_id: str, debug: TextIO | None) -> ExtractionPort:
    """Build the OpenAI-backed extraction port, or fail if it cannot be configured."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ConfigurationError("missing OPENAI_API_KEY; set it in extractor/.env")
    model = ChatOpenAI(model=model_id, reasoning_effort="none", temperature=0)

    def extract(document: str, schema: type[BaseModel]) -> Extraction:
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
        structured_model = model.with_structured_output(
            schema,
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        chain = prompt | structured_model
        try:
            result = cast(dict[str, Any], chain.invoke({"document": document}))
        except OpenAIRefusalError as error:
            return Refusal(detail=str(error))
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
