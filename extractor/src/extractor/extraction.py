from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, TextIO, TypedDict, cast

from anthropic import APIError as AnthropicAPIError
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic import NotFoundError as AnthropicNotFoundError
from anthropic import UnprocessableEntityError as AnthropicUnprocessableEntityError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIRefusalError
from openai import APIError, BadRequestError, NotFoundError, UnprocessableEntityError
from pydantic import BaseModel, SecretStr

from extractor.credentials import required_key


class ReasoningLevel(StrEnum):
    """Provider-neutral reasoning effort exposed by the command line."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


REASONING_LEVELS = {level.value: level for level in ReasoningLevel}


@dataclass(frozen=True, slots=True)
class PortSettings:
    """Everything a provider adapter needs to construct an extraction port."""

    model_id: str
    reasoning: ReasoningLevel
    debug: TextIO | None


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

    Both integrations type that call's result as a plain mapping, so this is the shape the
    one `cast` per exception funnel asserts. Every key an adapter reads is checked against it.
    """

    raw: BaseMessage
    parsed: BaseModel | None
    parsing_error: BaseException | None


class ExtractionPort(Protocol):
    """One extraction attempt: document plus schema in, one named outcome out."""

    def __call__(self, document: str, schema: type[BaseModel]) -> Extraction: ...


type PortFactory = Callable[[PortSettings], ExtractionPort]

# `Any` in the output position only: `with_structured_output` is typed as returning a plain
# mapping, and reshaping it is what the funnel's `cast` does. This is the boundary rule 6 allows.
type _StructuredChain = Runnable[dict[str, str], Any]

type _RefusalReader = Callable[[_RawStructuredOutput], str | None]

_REQUEST_TIMEOUT_SECONDS = 60
_MAX_RETRIES = 2

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The fail-closed half of the aggregator contract: select only endpoints that honour every
# parameter sent, so no endpoint incapable of enforcing the schema can serve the request.
_ROUTING_GUARD = {"require_parameters": True}

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract only facts stated in the source document. "
            "Do not infer or guess; use null when the source does not answer a field.",
        ),
        ("human", "{document}"),
    ]
)


def _classify(
    result: _RawStructuredOutput, debug: TextIO | None, refused: _RefusalReader
) -> Extraction:
    """Turn one structured-output envelope into one outcome, dumping the raw message first.

    Shared by every adapter: only how a provider reports a refusal differs, and that is
    `refused`. A refusal is read before a parsing error because a refused call carries no
    object to validate, so reporting it as a validation failure would name the wrong cause.
    """
    if debug is not None:
        debug.write(f"Raw model message: {result['raw']!r}\n")
    declined = refused(result)
    if declined is not None:
        return Refusal(detail=declined)
    parsing_error = result["parsing_error"]
    if parsing_error is not None:
        return ValidationFailure(detail=str(parsing_error))
    parsed = result["parsed"]
    if parsed is None:
        return EmptyExtraction()
    return Extracted(value=parsed)


def _openai_family_refusal(result: _RawStructuredOutput) -> str | None:
    """An OpenAI-format refusal arrives as a refusal error in the parsing-error slot.

    Reachable through OpenRouter but not guaranteed there: the error is raised off a message
    field any OpenAI-format response may carry, so it depends on the upstream provider
    passing it through. See ADR-0004.
    """
    error = result["parsing_error"]
    if isinstance(error, OpenAIRefusalError):
        return str(error)
    return None


def _anthropic_refusal(result: _RawStructuredOutput) -> str | None:
    """An Anthropic refusal is a stop reason on the raw message, not an exception or a field.

    This is the one raw-message inspection sanctioned inside an adapter: there is nowhere else
    the provider reports it, and a refusal read as an empty extraction names the wrong cause.
    """
    if result["raw"].response_metadata.get("stop_reason") != "refusal":
        return None
    return "the model stopped with stop reason 'refusal'"


def _through_openai_family(
    chain: _StructuredChain, document: str, debug: TextIO | None
) -> Extraction:
    """The funnel OpenAI and OpenRouter share, because they share one integration.

    Specific rejected-request classes are caught before `APIError`, so a newly added SDK
    subclass degrades to `ProviderFailure` rather than escaping. Credit exhaustion needs no
    case of its own: it is a status error with no named subclass, so it lands there too, and
    that is correct — re-running after topping up succeeds.
    """
    try:
        result = cast(_RawStructuredOutput, chain.invoke({"document": document}))
    except OpenAIRefusalError as error:
        return Refusal(detail=str(error))
    except (BadRequestError, NotFoundError, UnprocessableEntityError) as error:
        return ProviderRejectedRequest(detail=str(error))
    except APIError as error:
        return ProviderFailure(detail=str(error))
    return _classify(result, debug, _openai_family_refusal)


def _through_anthropic(chain: _StructuredChain, document: str, debug: TextIO | None) -> Extraction:
    """The Anthropic funnel, separate because its SDK shares no ancestor with OpenAI's.

    The class names coincide and the ancestry does not, so one classifier would have to import
    both SDKs to name them. ADR-0004 keeps each mapping inside its own adapter instead.
    """
    try:
        result = cast(_RawStructuredOutput, chain.invoke({"document": document}))
    except (
        AnthropicBadRequestError,
        AnthropicNotFoundError,
        AnthropicUnprocessableEntityError,
    ) as error:
        return ProviderRejectedRequest(detail=str(error))
    except AnthropicAPIError as error:
        return ProviderFailure(detail=str(error))
    return _classify(result, debug, _anthropic_refusal)


type _OpenAIReasoningEffort = Literal["none", "low", "medium", "high"]

_OPENAI_REASONING: dict[ReasoningLevel, _OpenAIReasoningEffort] = {
    ReasoningLevel.OFF: "none",
    ReasoningLevel.LOW: "low",
    ReasoningLevel.MEDIUM: "medium",
    ReasoningLevel.HIGH: "high",
}

type _AnthropicReasoningEffort = Literal["low", "medium", "high"]

# Anthropic spells "off" as a disabled thinking configuration and sets no effort at all: an
# effort re-enables adaptive thinking, so the two are mutually exclusive. The named levels set
# an effort and leave `thinking` unset, which is what turns adaptive thinking on.
_ANTHROPIC_REASONING: dict[
    ReasoningLevel, tuple[dict[str, str] | None, _AnthropicReasoningEffort | None]
] = {
    ReasoningLevel.OFF: ({"type": "disabled"}, None),
    ReasoningLevel.LOW: (None, "low"),
    ReasoningLevel.MEDIUM: (None, "medium"),
    ReasoningLevel.HIGH: (None, "high"),
}

type _OpenRouterReasoningEffort = Literal["none", "low", "medium", "high"]

# The aggregator's own spelling: an effort inside a `reasoning` object, not a flattened field.
# Reasoning is a cost lever and enforcement is the correctness contract, but on this provider the
# lever is not free of the contract: the routing guard requires every parameter sent to be
# honoured, so `reasoning` narrows endpoint selection too. Against a model that does not advertise
# reasoning, `--provider openrouter` therefore reports a rejected request rather than quietly
# ignoring the level. The guard cannot be scoped to one parameter, and losing enforcement is the
# worse trade, so this is accepted and recorded rather than worked around.
_OPENROUTER_REASONING: dict[ReasoningLevel, _OpenRouterReasoningEffort] = {
    ReasoningLevel.OFF: "none",
    ReasoningLevel.LOW: "low",
    ReasoningLevel.MEDIUM: "medium",
    ReasoningLevel.HIGH: "high",
}


def _openai_family_port(model: ChatOpenAI, debug: TextIO | None) -> ExtractionPort:
    """The extraction port OpenAI and OpenRouter share: strict json schema, one funnel.

    Only the model construction differs between those two providers. Everything after it —
    the binding arguments, the prompt, and the exception mapping — is one integration's
    behaviour and belongs in one place.
    """

    def extract(document: str, schema: type[BaseModel]) -> Extraction:
        structured_model = model.with_structured_output(
            schema,
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        return _through_openai_family(_PROMPT | structured_model, document, debug)

    return extract


def build_openai_port(settings: PortSettings) -> ExtractionPort:
    """Build the OpenAI-backed extraction port, or fail if it cannot be configured."""
    required_key("OPENAI_API_KEY")
    return _openai_family_port(
        ChatOpenAI(
            model=settings.model_id,
            reasoning_effort=_OPENAI_REASONING[settings.reasoning],
            temperature=0,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            max_retries=_MAX_RETRIES,
        ),
        settings.debug,
    )


def build_anthropic_port(settings: PortSettings) -> ExtractionPort:
    """Build the Anthropic-backed extraction port, or fail if it cannot be configured.

    No temperature is set. Anthropic rejects a modified temperature whenever thinking is on,
    and all three named reasoning levels turn it on, so pinning it would break every default
    run instead of buying repeatability.
    """
    required_key("ANTHROPIC_API_KEY")
    thinking, effort = _ANTHROPIC_REASONING[settings.reasoning]
    model = ChatAnthropic(
        model=settings.model_id,
        thinking=thinking,
        reasoning_effort=effort,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )

    def extract(document: str, schema: type[BaseModel]) -> Extraction:
        # `json_schema` is the enforcement here; this integration has no `strict` argument.
        # `function_calling` must never be used: it forces tool choice, which the provider
        # rejects when thinking is enabled — and thinking is on at every named level.
        structured_model = model.with_structured_output(
            schema,
            method="json_schema",
            include_raw=True,
        )
        return _through_anthropic(_PROMPT | structured_model, document, settings.debug)

    return extract


def build_openrouter_port(settings: PortSettings) -> ExtractionPort:
    """Build the OpenRouter-backed extraction port, or fail if it cannot be configured.

    The aggregator is reached through the OpenAI-compatible chat model pointed at its base
    URL, so it adds no dependency and inherits that integration's exception classes.

    `extra_body` is load-bearing: the SDK hoists it to the top level of the request, which is
    where the aggregator reads the routing guard and the reasoning setting. The flattened
    model-arguments channel would put a top-level `reasoning` key where it selects a different
    API surface the aggregator does not serve; `use_responses_api=False` refuses that same
    selection being inferred from the model id.

    No temperature is set. The routing guard turns every parameter sent into a routing
    constraint, and the default model's endpoint does not advertise temperature, so sending it
    would leave the aggregator with no endpoint able to enforce the schema.
    """
    key = required_key("OPENROUTER_API_KEY")
    return _openai_family_port(
        ChatOpenAI(
            model=settings.model_id,
            base_url=_OPENROUTER_BASE_URL,
            api_key=SecretStr(key),
            use_responses_api=False,
            extra_body={
                "provider": _ROUTING_GUARD,
                "reasoning": {"effort": _OPENROUTER_REASONING[settings.reasoning]},
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
            max_retries=_MAX_RETRIES,
        ),
        settings.debug,
    )


@dataclass(frozen=True, slots=True)
class Provider:
    """A registered provider: the model it uses unless overridden, and its port factory."""

    default_model: str
    build_port: PortFactory


PROVIDERS: dict[str, Provider] = {
    "openai": Provider("gpt-5-nano", build_openai_port),
    # Deliberately not a Haiku tier: Haiku 4.5 and Sonnet 4.5 reject the effort parameter
    # server-side, which would make the default reasoning level fail on every run.
    "anthropic": Provider("claude-sonnet-5", build_anthropic_port),
    # Deliberately a model also reachable directly, so comparing the two paths isolates
    # routing rather than changing two variables at once.
    "openrouter": Provider("openai/gpt-5-nano", build_openrouter_port),
}
DEFAULT_PROVIDER = next(iter(PROVIDERS))
