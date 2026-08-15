"""Offline adapter tests. No test here reaches a provider or the network.

Two seams, deliberately.

Outcome tests build a `ProviderAdapter` whose `build_model` returns a canned chat model. That
is the seam `main` itself crosses, so they need no API key, no environment, and no patching of
the module's namespace. The canned model is a real `ChatOpenAI` or `ChatAnthropic` subclass,
because the structured-output parsing that decides an outcome runs inside the provider package
and only runs on a real subclass.

Configuration tests are the exception: they exist to pin the arguments handed to the SDK class,
which is visible nowhere else, so they substitute that class in this module's namespace and call
the registry's model builder directly. The one test that needs a real HTTP request binds a
loopback server of its own.

OpenAI and OpenRouter share one integration, so at the outcome seam they share their outcomes;
what differs is the model each builds, which the configuration tests cover. The two SDKs do not
even share an HTTP library — `openai` carries `httpx2` and `anthropic` carries `httpx` — so
provider errors are built with the flavour their own SDK expects.
"""

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIRefusalError
from openai import (
    APIError,
    APIStatusError,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)
from pydantic import SecretStr

from extractor import extraction
from extractor.credentials import ConfigurationError
from extractor.extraction import (
    ANTHROPIC,
    OPENAI_FAMILY,
    PROVIDERS,
    EmptyExtraction,
    Extracted,
    Extraction,
    Integration,
    PortSettings,
    ProviderAdapter,
    ProviderFailure,
    ProviderRejectedRequest,
    ReasoningLevel,
    Refusal,
    ValidationFailure,
)
from extractor.schemas import TermsOfService

VALID_FACTS = {
    "governing_law": "State of New York",
    "arbitration_required": True,
    "arbitration_clause": "Disputes must be resolved by binding arbitration.",
    "liability_cap": "$100",
    "termination_notice_period": "30 days",
    "data_retention_period": None,
    "effective_date": "2026-01-01",
}

ALL_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")


class CannedOpenAIChatModel(ChatOpenAI):
    """A real `ChatOpenAI` whose single response is supplied, so no request leaves the process.

    The adapter's outcomes are decided by `langchain_openai`'s own structured-output
    parsing, which only runs on a `ChatOpenAI` subclass, so the double has to be one.
    """

    # `Any`, not a union: pydantic would try to coerce an exception into an `AIMessage`.
    response: Any

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(self.response, BaseException):
            raise self.response
        return ChatResult(generations=[ChatGeneration(message=self.response)])


class CannedAnthropicChatModel(ChatAnthropic):
    """The same double for Anthropic, whose json-schema parsing also runs in its own package."""

    response: Any

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(self.response, BaseException):
            raise self.response
        return ChatResult(generations=[ChatGeneration(message=self.response)])


class StubbedModelClass:
    """Substitutes the chat-model class a builder constructs, and records its arguments.

    Only the configuration tests need this. Everything that asserts an *outcome* goes through
    `canned_provider` instead, which needs no patching because the model builder is a field on
    the provider record.
    """

    def __init__(
        self,
        *,
        target: str = "ChatOpenAI",
        canned: Callable[..., Any] = CannedOpenAIChatModel,
        api_key: str = "OPENAI_API_KEY",
    ) -> None:
        self.target = target
        self.canned = canned
        self.api_key = api_key
        self.configuration: dict[str, Any] = {}
        self.built = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(self.api_key, "test-key")

        def build(**configuration: Any) -> Any:
            self.configuration = configuration
            self.built = True
            # The builder may already have supplied a key; only fill one in when it did not.
            return self.canned(
                **{"api_key": SecretStr("test-key"), **configuration},
                response=AIMessage(content=""),
            )

        monkeypatch.setattr(f"extractor.extraction.{self.target}", build)


def canned_provider(
    response: AIMessage | BaseException,
    *,
    integration: Integration = OPENAI_FAMILY,
    canned: Callable[..., BaseChatModel] = CannedOpenAIChatModel,
) -> ProviderAdapter:
    """A provider record whose model builder returns a canned chat model.

    This is the seam `main` crosses, so an outcome test needs no API key, no environment, and
    no patching of this module's namespace — it supplies a `build_model` and reads the outcome.
    """

    def build_model(settings: PortSettings) -> BaseChatModel:
        return canned(model=settings.model_id, api_key=SecretStr("test-key"), response=response)

    return ProviderAdapter("canned-model", build_model, integration)


def extract_through(provider: ProviderAdapter, *, debug: StringIO | None = None) -> Extraction:
    port = provider.build_port(PortSettings(provider.default_model, ReasoningLevel.MEDIUM, debug))
    return port("Terms of Service source", TermsOfService)


def extract_through_openai(
    response: AIMessage | BaseException, *, debug: StringIO | None = None
) -> Extraction:
    return extract_through(canned_provider(response), debug=debug)


def extract_through_anthropic(
    response: AIMessage | BaseException, *, debug: StringIO | None = None
) -> Extraction:
    return extract_through(
        canned_provider(response, integration=ANTHROPIC, canned=CannedAnthropicChatModel),
        debug=debug,
    )


def parsed_message(facts: dict[str, object]) -> AIMessage:
    """The shape OpenAI strict structured output returns: facts on the raw message."""
    return AIMessage(content="", additional_kwargs={"parsed": facts})


def anthropic_message(facts: dict[str, object]) -> AIMessage:
    """The shape Anthropic json-schema output returns: the object as the message text."""
    return AIMessage(content=json.dumps(facts))


def openai_error(kind: type[APIStatusError], message: str, status: int) -> APIStatusError:
    """Build an `openai` status error, which carries `httpx2`."""
    request = httpx2.Request("POST", "https://provider.test")
    return kind(message, response=httpx2.Response(status, request=request), body=None)


def anthropic_error(
    kind: type[anthropic.APIStatusError], message: str, status: int
) -> anthropic.APIStatusError:
    """Build an `anthropic` status error, which carries `httpx`."""
    request = httpx.Request("POST", "https://provider.test")
    return kind(message, response=httpx.Response(status, request=request), body=None)


def test_a_parsed_object_is_an_extracted_outcome() -> None:
    outcome = extract_through_openai(parsed_message(VALID_FACTS))

    assert outcome == Extracted(TermsOfService.model_validate(VALID_FACTS))


def test_an_answer_carrying_no_object_is_an_empty_extraction_and_never_raises() -> None:
    answered_without_committing = AIMessage(
        content="",
        tool_calls=[{"name": "TermsOfService", "args": {}, "id": "call-1", "type": "tool_call"}],
    )

    outcome = extract_through_openai(answered_without_committing)

    assert outcome == EmptyExtraction()


def test_an_object_the_schema_rejects_is_a_validation_failure() -> None:
    outcome = extract_through_openai(parsed_message({**VALID_FACTS, "effective_date": "nope"}))

    assert isinstance(outcome, ValidationFailure)
    assert "effective_date" in outcome.detail


def test_a_refusal_carried_by_the_raw_message_is_a_refusal_not_a_validation_failure() -> None:
    refused = AIMessage(
        content="",
        additional_kwargs={"refusal": "The provider declined this extraction."},
    )

    outcome = extract_through_openai(refused)

    assert isinstance(outcome, Refusal)
    assert "declined" in outcome.detail


def test_a_refusal_raised_by_the_provider_is_a_refusal() -> None:
    outcome = extract_through_openai(OpenAIRefusalError("The provider declined this extraction."))

    assert isinstance(outcome, Refusal)
    assert "declined" in outcome.detail


def test_a_refusal_carried_on_the_message_is_read_by_the_openai_family_refusal_reader() -> None:
    """Reachable through OpenRouter, not guaranteed: the refusal is read off a message field
    any OpenAI-format response may carry, so it depends on the upstream provider sending it.
    ADR-0004. The aggregator inherits this reader whole, which the registry test above pins."""
    refused = AIMessage(
        content="",
        additional_kwargs={"refusal": "The upstream provider declined this extraction."},
    )

    outcome = extract_through_openai(refused)

    assert isinstance(outcome, Refusal)
    assert "declined" in outcome.detail


def test_an_anthropic_stop_reason_of_refusal_is_a_refusal() -> None:
    """Anthropic reports a refusal nowhere else. `stop_reason` reaches `response_metadata`
    because `langchain_core` merges the generation's `llm_output` into it."""
    refused = AIMessage(content="", response_metadata={"stop_reason": "refusal"})

    outcome = extract_through_anthropic(refused)

    assert outcome == Refusal(detail="the model stopped with stop reason 'refusal'")


def test_an_anthropic_json_object_is_an_extracted_outcome() -> None:
    outcome = extract_through_anthropic(anthropic_message(VALID_FACTS))

    assert outcome == Extracted(TermsOfService.model_validate(VALID_FACTS))


def test_an_anthropic_object_the_schema_rejects_is_a_validation_failure() -> None:
    outcome = extract_through_anthropic(
        anthropic_message({**VALID_FACTS, "effective_date": "nope"})
    )

    assert isinstance(outcome, ValidationFailure)
    assert "effective_date" in outcome.detail


def test_the_aggregator_is_registered_with_the_openai_family_integration() -> None:
    """This is the whole of what OpenRouter shares: one binding, one exception funnel, one
    refusal reader. Asserting it here is what lets every OpenAI outcome test count for the
    aggregator too — re-running those tests against an identically built canned model would
    assert nothing, because at this seam the two are the same code.
    """
    assert PROVIDERS["openrouter"].integration is OPENAI_FAMILY
    assert PROVIDERS["openai"].integration is OPENAI_FAMILY
    assert PROVIDERS["anthropic"].integration is ANTHROPIC


@pytest.mark.parametrize(
    ("extract", "response"),
    [
        (extract_through_openai, parsed_message(VALID_FACTS)),
        (extract_through_anthropic, anthropic_message(VALID_FACTS)),
    ],
)
def test_the_debug_stream_receives_the_raw_provider_message_on_every_integration(
    monkeypatch: pytest.MonkeyPatch,
    extract: Callable[..., Extraction],
    response: AIMessage,
) -> None:
    debug = StringIO()

    extract(response, debug=debug)

    dumped = debug.getvalue()
    assert dumped.startswith("Raw model message: ")
    assert "State of New York" in dumped


@pytest.mark.parametrize(
    ("extract", "response"),
    [
        (extract_through_openai, parsed_message(VALID_FACTS)),
        (extract_through_anthropic, anthropic_message(VALID_FACTS)),
    ],
)
def test_without_a_debug_stream_no_adapter_writes_to_stdout_or_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extract: Callable[..., Extraction],
    response: AIMessage,
) -> None:
    """`_report` is the sole writer of extraction output, on every provider."""
    extract(response)

    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    ("reasoning", "provider_spelling"),
    [
        (ReasoningLevel.OFF, "none"),
        (ReasoningLevel.LOW, "low"),
        (ReasoningLevel.MEDIUM, "medium"),
        (ReasoningLevel.HIGH, "high"),
    ],
)
def test_each_reasoning_level_reaches_openai_in_its_own_spelling(
    monkeypatch: pytest.MonkeyPatch,
    reasoning: ReasoningLevel,
    provider_spelling: str,
) -> None:
    model_class = StubbedModelClass()
    model_class.install(monkeypatch)

    PROVIDERS["openai"].build_model(PortSettings("gpt-5-mini", reasoning, None))

    assert model_class.configuration == {
        "model": "gpt-5-mini",
        "reasoning_effort": provider_spelling,
        "temperature": 0,
        "timeout": 60,
        "max_retries": 2,
    }


@pytest.mark.parametrize(
    ("reasoning", "thinking", "effort"),
    [
        # "Off" is a disabled thinking configuration and no effort at all: an effort would
        # re-enable adaptive thinking, and the two are mutually exclusive.
        (ReasoningLevel.OFF, {"type": "disabled"}, None),
        (ReasoningLevel.LOW, None, "low"),
        (ReasoningLevel.MEDIUM, None, "medium"),
        (ReasoningLevel.HIGH, None, "high"),
    ],
)
def test_each_reasoning_level_reaches_anthropic_in_its_own_spelling(
    monkeypatch: pytest.MonkeyPatch,
    reasoning: ReasoningLevel,
    thinking: dict[str, str] | None,
    effort: str | None,
) -> None:
    model_class = StubbedModelClass(
        target="ChatAnthropic", canned=CannedAnthropicChatModel, api_key="ANTHROPIC_API_KEY"
    )
    model_class.install(monkeypatch)

    PROVIDERS["anthropic"].build_model(PortSettings("claude-sonnet-5", reasoning, None))

    assert model_class.configuration == {
        "model": "claude-sonnet-5",
        "thinking": thinking,
        "reasoning_effort": effort,
        "timeout": 60,
        "max_retries": 2,
    }


@pytest.mark.parametrize(
    ("reasoning", "provider_spelling"),
    [
        (ReasoningLevel.OFF, "none"),
        (ReasoningLevel.LOW, "low"),
        (ReasoningLevel.MEDIUM, "medium"),
        (ReasoningLevel.HIGH, "high"),
    ],
)
def test_each_reasoning_level_reaches_openrouter_through_the_extra_body_channel(
    monkeypatch: pytest.MonkeyPatch,
    reasoning: ReasoningLevel,
    provider_spelling: str,
) -> None:
    model_class = StubbedModelClass(api_key="OPENROUTER_API_KEY")
    model_class.install(monkeypatch)

    PROVIDERS["openrouter"].build_model(PortSettings("openai/gpt-5-nano", reasoning, None))

    assert model_class.configuration["extra_body"] == {
        "provider": {"require_parameters": True},
        "reasoning": {"effort": provider_spelling},
    }


def test_the_openrouter_model_is_configured_for_the_aggregator_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole configuration, exactly: the routing guard and the reasoning setting travel in
    `extra_body`, there is no flattened model-arguments channel, and the alternative API
    surface is refused explicitly rather than inferred from the model id.

    No temperature: the routing guard makes every parameter a routing constraint, and the
    default model's endpoint does not advertise temperature, so sending it would leave the
    aggregator with no endpoint able to enforce the schema.
    """
    model_class = StubbedModelClass(api_key="OPENROUTER_API_KEY")
    model_class.install(monkeypatch)

    PROVIDERS["openrouter"].build_model(
        PortSettings("openai/gpt-5-nano", ReasoningLevel.MEDIUM, None)
    )

    assert model_class.configuration == {
        "model": "openai/gpt-5-nano",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": SecretStr("test-key"),
        "use_responses_api": False,
        "extra_body": {
            "provider": {"require_parameters": True},
            "reasoning": {"effort": "medium"},
        },
        "timeout": 60,
        "max_retries": 2,
    }


@pytest.mark.parametrize(
    ("provider", "key"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_each_adapter_checks_only_its_own_key_before_building_the_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    key: str,
) -> None:
    """Every other provider's key is present and does not help, and the message names both the
    missing key and where to put it. Nothing is constructed, so a misconfigured run is free."""
    monkeypatch.setattr("extractor.credentials.ENV_FILE", tmp_path / "absent.env")
    for other in ALL_KEYS:
        monkeypatch.setenv(other, "test-key")
    monkeypatch.delenv(key, raising=False)
    built: list[str] = []
    for target in ("ChatOpenAI", "ChatAnthropic"):
        monkeypatch.setattr(f"extractor.extraction.{target}", lambda **_: built.append("built"))

    with pytest.raises(ConfigurationError, match=rf"{key}.*extractor/\.env"):
        PROVIDERS[provider].build_model(PortSettings("any-model", ReasoningLevel.MEDIUM, None))

    assert built == []


@pytest.mark.parametrize(
    ("provider", "key"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_each_adapter_takes_its_key_from_the_credentials_file_not_the_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    key: str,
) -> None:
    """No key is exported, so the adapter can only build if it resolved one from the file.

    `test_credentials.py` pins where that file lives; this pins that every adapter goes
    through it rather than reading the environment directly.
    """
    for name in ALL_KEYS:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{key}=file-key\n", encoding="utf-8")
    monkeypatch.setattr("extractor.credentials.ENV_FILE", env_file)
    built: list[str] = []
    for target in ("ChatOpenAI", "ChatAnthropic"):
        monkeypatch.setattr(f"extractor.extraction.{target}", lambda **_: built.append("built"))

    PROVIDERS[provider].build_model(PortSettings("any-model", ReasoningLevel.MEDIUM, None))

    assert built == ["built"]


def test_the_openai_binding_asks_the_provider_to_enforce_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0001: enforcement is provider-side, so the binding arguments are the contract.

    `strict=False` or a `method` of `function_calling` degrades enforcement to a polite
    request with no error and no warning, and every other offline test still passes.
    """
    bindings = record_bindings(monkeypatch, ChatOpenAI)

    extract_through_openai(parsed_message(VALID_FACTS))

    assert bindings == [
        {
            "schema": TermsOfService,
            "method": "json_schema",
            "strict": True,
            "include_raw": True,
        }
    ]


def test_the_anthropic_binding_enforces_through_json_schema_never_function_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """There is no `strict` argument on this integration: the method *is* the enforcement.

    `function_calling` must never appear here. It forces tool choice, which the provider
    rejects when thinking is enabled — and thinking is on at every named reasoning level.
    """
    bindings = record_bindings(monkeypatch, ChatAnthropic)

    extract_through_anthropic(anthropic_message(VALID_FACTS))

    assert bindings == [
        {
            "schema": TermsOfService,
            "method": "json_schema",
            "include_raw": True,
        }
    ]


def record_bindings(
    monkeypatch: pytest.MonkeyPatch, model_class: type[Any]
) -> list[dict[str, Any]]:
    """Record every `with_structured_output` call on a chat-model class, still calling through."""
    bindings: list[dict[str, Any]] = []
    bind = model_class.with_structured_output

    def record(model: Any, schema: Any = None, **kwargs: Any) -> Any:
        bindings.append({"schema": schema, **kwargs})
        return bind(model, schema, **kwargs)

    monkeypatch.setattr(model_class, "with_structured_output", record)
    return bindings


@pytest.fixture
def aggregator_stub() -> Iterator[dict[str, Any]]:
    """A local chat-completions server that records the request the adapter actually emits.

    The seam above records how the model was *configured*; only the emitted request proves the
    SDK hoists the extra body to the top level of the request, which is where the aggregator
    reads it. Loopback only, so the default run stays offline.
    """
    captured: dict[str, Any] = {}
    answer = json.dumps(VALID_FACTS)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            captured["path"] = self.path
            captured["body"] = json.loads(self.rfile.read(length))
            body = json.dumps(
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "created": 0,
                    "model": captured["body"]["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": answer},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the handler: stderr belongs to the extractor's own diagnostics."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    captured["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_the_routing_guard_and_reasoning_arrive_as_top_level_request_fields(
    monkeypatch: pytest.MonkeyPatch, aggregator_stub: dict[str, Any]
) -> None:
    """A chat-completions request, never the alternative API surface, carrying the routing
    guard and the reasoning setting at the top level beside a strict json-schema format."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(extraction, "_OPENROUTER_BASE_URL", aggregator_stub["base_url"])

    port = PROVIDERS["openrouter"].build_port(
        PortSettings("openai/gpt-5-nano", ReasoningLevel.MEDIUM, None)
    )
    outcome = port("Terms of Service source", TermsOfService)

    assert outcome == Extracted(TermsOfService.model_validate(VALID_FACTS))
    assert aggregator_stub["path"] == "/v1/chat/completions"
    body = aggregator_stub["body"]
    assert body["provider"] == {"require_parameters": True}
    assert body["reasoning"] == {"effort": "medium"}
    assert sorted(body["response_format"]["json_schema"]) == ["name", "schema", "strict"]
    assert body["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize(
    "error",
    [
        openai_error(BadRequestError, "bad request", 400),
        openai_error(NotFoundError, "not found", 404),
        openai_error(UnprocessableEntityError, "unprocessable", 422),
    ],
)
def test_each_rejected_request_class_becomes_a_provider_rejected_request(
    monkeypatch: pytest.MonkeyPatch,
    error: APIError,
) -> None:
    outcome = extract_through_openai(error)

    assert outcome == ProviderRejectedRequest(detail=str(error))


@pytest.mark.parametrize(
    "error",
    [
        anthropic_error(anthropic.BadRequestError, "bad request", 400),
        anthropic_error(anthropic.NotFoundError, "not found", 404),
        anthropic_error(anthropic.UnprocessableEntityError, "unprocessable", 422),
    ],
)
def test_each_anthropic_rejected_request_class_becomes_a_provider_rejected_request(
    monkeypatch: pytest.MonkeyPatch,
    error: anthropic.APIError,
) -> None:
    outcome = extract_through_anthropic(error)

    assert outcome == ProviderRejectedRequest(detail=str(error))


def test_the_provider_family_base_class_becomes_a_provider_failure() -> None:
    error = APIError(
        "provider unavailable",
        request=httpx2.Request("POST", "https://provider.test"),
        body=None,
    )

    outcome = extract_through_openai(error)

    assert outcome == ProviderFailure(detail=str(error))


def test_the_anthropic_family_base_class_becomes_a_provider_failure() -> None:
    error = anthropic.APIError(
        "provider unavailable",
        request=httpx.Request("POST", "https://provider.test"),
        body=None,
    )

    outcome = extract_through_anthropic(error)

    assert outcome == ProviderFailure(detail=str(error))


def test_exhausted_aggregator_credit_becomes_a_provider_failure() -> None:
    """It has no named subclass, so it falls through to the base class — which is right, since
    re-running after topping up succeeds. Raised by the aggregator, funnelled by the OpenAI
    family integration it is registered with."""
    credit_exhausted = openai_error(APIStatusError, "insufficient credits", 402)

    outcome = extract_through_openai(credit_exhausted)

    assert outcome == ProviderFailure(detail=str(credit_exhausted))


@pytest.mark.parametrize(
    ("extract", "error"),
    [
        (extract_through_openai, openai_error(NotFoundError, "model does not exist", 404)),
        (
            extract_through_anthropic,
            anthropic_error(anthropic.NotFoundError, "model does not exist", 404),
        ),
    ],
)
def test_an_unknown_model_id_is_a_provider_rejected_request_on_every_integration(
    monkeypatch: pytest.MonkeyPatch,
    extract: Callable[..., Extraction],
    error: BaseException,
) -> None:
    outcome = extract(error)

    assert outcome == ProviderRejectedRequest(detail=str(error))
