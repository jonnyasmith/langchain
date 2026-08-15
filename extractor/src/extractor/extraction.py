from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, TextIO, TypedDict, assert_never, cast

from anthropic import APIError as AnthropicAPIError
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic import NotFoundError as AnthropicNotFoundError
from anthropic import UnprocessableEntityError as AnthropicUnprocessableEntityError
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel, LanguageModelInput
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


class Provider(Protocol):
    """The seam `main` depends on: a default model, and a port built from settings.

    Narrower than the registry's record on purpose — it names the two members the CLI uses and
    nothing else, so a test satisfies it without a model builder or an integration.
    `ProviderAdapter` satisfies it structurally; neither has to know about the other.
    """

    @property
    def default_model(self) -> str: ...

    def build_port(self, settings: PortSettings) -> ExtractionPort: ...


# `Any` in the output position only: `with_structured_output` is typed as returning a plain
# mapping, and reshaping it is what the funnel's `cast` does. This is the boundary rule 6 allows.
type _StructuredChain = Runnable[dict[str, str], Any]

# The bound model before the prompt is piped into it. `Any` stays in the output position only,
# as it does for `_StructuredChain`: the input is the integration's own named message union.
type _StructuredModel = Runnable[LanguageModelInput, Any]

type _RefusalReader = Callable[[_RawStructuredOutput], str | None]

type SchemaBinder = Callable[[BaseChatModel, type[BaseModel]], _StructuredModel]
"""Bind a schema through one integration's enforced structured-output path."""

type IntegrationCall = Callable[[_StructuredChain, str, TextIO | None], Extraction]
"""Invoke one chain and return an outcome. Never raises: this is where exceptions stop."""

type ModelBuilder = Callable[[PortSettings], BaseChatModel]
"""Construct one provider's chat model, or raise `ConfigurationError` before anything is sent."""

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


type _Effort = Literal["none", "low", "medium", "high"]


def _openai_family_effort(level: ReasoningLevel) -> _Effort:
    """The OpenAI-format spelling: `off` is an effort of `none`, and the rest carry over.

    Shared with the aggregator, which uses the same four words — but not the same channel, so
    only the vocabulary is common. A `match` rather than a table: a fifth reasoning level then
    fails type checking here instead of raising a `KeyError` at the first run that selects it.
    """
    match level:
        case ReasoningLevel.OFF:
            return "none"
        case ReasoningLevel.LOW:
            return "low"
        case ReasoningLevel.MEDIUM:
            return "medium"
        case ReasoningLevel.HIGH:
            return "high"
        case unreachable:
            assert_never(unreachable)


class _AnthropicReasoning(TypedDict):
    """Anthropic's two mutually exclusive reasoning controls, spelled as constructor arguments.

    Keyed by parameter name so the builder unpacks it rather than reading fields back out;
    there is no order to transpose and no half of it the builder can forget to pass on.
    """

    thinking: dict[str, str] | None
    reasoning_effort: Literal["low", "medium", "high"] | None


def _anthropic_reasoning(level: ReasoningLevel) -> _AnthropicReasoning:
    """`off` is a disabled thinking configuration and no effort at all.

    An effort re-enables adaptive thinking, so the two controls cannot both be set. The named
    levels set an effort and leave `thinking` unset, which is what turns adaptive thinking on.
    """
    match level:
        case ReasoningLevel.OFF:
            return _AnthropicReasoning(thinking={"type": "disabled"}, reasoning_effort=None)
        case ReasoningLevel.LOW:
            return _AnthropicReasoning(thinking=None, reasoning_effort="low")
        case ReasoningLevel.MEDIUM:
            return _AnthropicReasoning(thinking=None, reasoning_effort="medium")
        case ReasoningLevel.HIGH:
            return _AnthropicReasoning(thinking=None, reasoning_effort="high")
        case unreachable:
            assert_never(unreachable)


def _openrouter_reasoning(level: ReasoningLevel) -> dict[str, _Effort]:
    """The aggregator's own channel: an effort inside a `reasoning` object, not a flat field.

    Reasoning is a cost lever and enforcement is the correctness contract, but on this provider
    the lever is not free of the contract: the routing guard requires every parameter sent to be
    honoured, so `reasoning` narrows endpoint selection too. Against a model that does not
    advertise reasoning, `--provider openrouter` therefore reports a rejected request rather than
    quietly ignoring the level. The guard cannot be scoped to one parameter, and losing
    enforcement is the worse trade, so this is accepted and recorded rather than worked around.
    """
    return {"effort": _openai_family_effort(level)}


def _bind_openai_family(model: BaseChatModel, schema: type[BaseModel]) -> _StructuredModel:
    """ADR-0001: enforcement is provider-side, so these arguments are the contract.

    `strict=False` or a `method` of `function_calling` degrades enforcement to a polite request
    with no error and no warning.
    """
    return model.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)


def _bind_anthropic(model: BaseChatModel, schema: type[BaseModel]) -> _StructuredModel:
    """`json_schema` is the enforcement here; this integration has no `strict` argument.

    `function_calling` must never be used: it forces tool choice, which the provider rejects
    when thinking is enabled — and thinking is on at every named reasoning level.
    """
    return model.with_structured_output(schema, method="json_schema", include_raw=True)


@dataclass(frozen=True, slots=True)
class Integration:
    """One LangChain chat integration: how it binds a schema, and how its SDK reports failure.

    The two travel together because they come from the same package. Two integrations serve
    three providers — OpenAI and OpenRouter share this one — and ADR-0004 requires the two
    exception mappings stay apart, because the SDKs' identically named error classes share no
    ancestor. Holding each mapping on its own record is what keeps them apart.
    """

    bind: SchemaBinder
    call: IntegrationCall


OPENAI_FAMILY = Integration(bind=_bind_openai_family, call=_through_openai_family)
"""Serves every OpenAI-format provider, direct or through an aggregator."""

ANTHROPIC = Integration(bind=_bind_anthropic, call=_through_anthropic)
"""Serves Anthropic, whose SDK shares no exception ancestor with the OpenAI family's."""


def _build_openai_model(settings: PortSettings) -> BaseChatModel:
    """Build the OpenAI chat model, or fail before anything is sent.

    Temperature is pinned here and nowhere else; the other two providers cannot accept it.
    """
    required_key("OPENAI_API_KEY")
    return ChatOpenAI(
        model=settings.model_id,
        reasoning_effort=_openai_family_effort(settings.reasoning),
        temperature=0,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )


def _build_anthropic_model(settings: PortSettings) -> BaseChatModel:
    """Build the Anthropic chat model, or fail before anything is sent.

    No temperature is set. Anthropic rejects a modified temperature whenever thinking is on, and
    all three named reasoning levels turn it on, so pinning it would break every default run
    instead of buying repeatability.
    """
    required_key("ANTHROPIC_API_KEY")
    return ChatAnthropic(
        model=settings.model_id,
        **_anthropic_reasoning(settings.reasoning),
        timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )


def _build_openrouter_model(settings: PortSettings) -> BaseChatModel:
    """Build the aggregator's chat model, or fail before anything is sent.

    The aggregator is reached through the OpenAI-compatible chat model pointed at its base URL,
    so it adds no dependency and inherits that integration's exception classes.

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
    return ChatOpenAI(
        model=settings.model_id,
        base_url=_OPENROUTER_BASE_URL,
        api_key=SecretStr(key),
        use_responses_api=False,
        extra_body={
            "provider": _ROUTING_GUARD,
            "reasoning": _openrouter_reasoning(settings.reasoning),
        },
        timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )


@dataclass(frozen=True, slots=True)
class ProviderAdapter:
    """A registered provider: everything that varies between adapters, in one record.

    Reading one entry tells you the whole adapter — the model it uses unless overridden, how it
    constructs one, and which integration serves it. Adding a fourth provider is filling in this
    record; nothing else in the module learns its name.
    """

    default_model: str
    build_model: ModelBuilder
    integration: Integration

    def build_port(self, settings: PortSettings) -> ExtractionPort:
        """Construct the extraction port, or fail before anything is sent.

        The prompt, the debug dump, and outcome classification are invariant across providers,
        so they live here once. The model is built now and the schema is bound per call, because
        only the schema arrives with the document.
        """
        model = self.build_model(settings)
        integration = self.integration

        def extract(document: str, schema: type[BaseModel]) -> Extraction:
            return integration.call(
                _PROMPT | integration.bind(model, schema), document, settings.debug
            )

        return extract


PROVIDERS: dict[str, ProviderAdapter] = {
    "openai": ProviderAdapter("gpt-5-nano", _build_openai_model, OPENAI_FAMILY),
    # Deliberately not a Haiku tier: Haiku 4.5 and Sonnet 4.5 reject the effort parameter
    # server-side, which would make the default reasoning level fail on every run.
    "anthropic": ProviderAdapter("claude-sonnet-5", _build_anthropic_model, ANTHROPIC),
    # Deliberately a model also reachable directly, so comparing the two paths isolates
    # routing rather than changing two variables at once. It shares OpenAI's integration,
    # which the registry now states rather than leaving to a shared builder's name.
    "openrouter": ProviderAdapter("openai/gpt-5-nano", _build_openrouter_model, OPENAI_FAMILY),
}
DEFAULT_PROVIDER = next(iter(PROVIDERS))
