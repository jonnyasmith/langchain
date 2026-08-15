"""Offline adapter tests, all through the substituted chat-model seam.

No test here reaches the network. Each adapter's provider chat-model class is replaced by a
recording builder that returns a canned real subclass, because the structured-output parsing
that decides an outcome runs inside the provider package and only runs on a real subclass.

OpenAI and OpenRouter share `ChatOpenAI`, so their tests are told apart by what the recorded
configuration contains: the base URL, the routing guard, and the reasoning channel.

The two SDKs do not even share an HTTP library — `openai` carries `httpx2` and `anthropic`
carries `httpx` — so provider errors are built with the flavour their own SDK expects.
"""

import json
import os
import re
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
from extractor.extraction import (
    ConfigurationError,
    EmptyExtraction,
    Extracted,
    Extraction,
    PortFactory,
    PortSettings,
    ProviderFailure,
    ProviderRejectedRequest,
    ReasoningLevel,
    Refusal,
    ValidationFailure,
    _load_env_file,
    build_anthropic_port,
    build_openai_port,
    build_openrouter_port,
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


class StubbedProvider:
    """Substitutes the chat-model class an adapter builds, and records how it was configured."""

    def __init__(
        self,
        response: AIMessage | BaseException,
        *,
        target: str = "ChatOpenAI",
        canned: Callable[..., Any] = CannedOpenAIChatModel,
        api_key: str = "OPENAI_API_KEY",
    ) -> None:
        self.response = response
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
            # The adapter may already have supplied a key; only fill one in when it did not.
            return self.canned(
                **{"api_key": SecretStr("test-key"), **configuration}, response=self.response
            )

        monkeypatch.setattr(f"extractor.extraction.{self.target}", build)


def openai_stub(response: AIMessage | BaseException) -> StubbedProvider:
    return StubbedProvider(response)


def openrouter_stub(response: AIMessage | BaseException) -> StubbedProvider:
    return StubbedProvider(response, api_key="OPENROUTER_API_KEY")


def anthropic_stub(response: AIMessage | BaseException) -> StubbedProvider:
    return StubbedProvider(
        response,
        target="ChatAnthropic",
        canned=CannedAnthropicChatModel,
        api_key="ANTHROPIC_API_KEY",
    )


def extract_through_openai(
    monkeypatch: pytest.MonkeyPatch,
    response: AIMessage | BaseException,
    *,
    debug: StringIO | None = None,
) -> Extraction:
    openai_stub(response).install(monkeypatch)
    port = build_openai_port(PortSettings("gpt-5-nano", ReasoningLevel.MEDIUM, debug))
    return port("Terms of Service source", TermsOfService)


def extract_through_openrouter(
    monkeypatch: pytest.MonkeyPatch,
    response: AIMessage | BaseException,
    *,
    debug: StringIO | None = None,
) -> Extraction:
    openrouter_stub(response).install(monkeypatch)
    port = build_openrouter_port(PortSettings("openai/gpt-5-nano", ReasoningLevel.MEDIUM, debug))
    return port("Terms of Service source", TermsOfService)


def extract_through_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    response: AIMessage | BaseException,
    *,
    debug: StringIO | None = None,
) -> Extraction:
    anthropic_stub(response).install(monkeypatch)
    port = build_anthropic_port(PortSettings("claude-sonnet-5", ReasoningLevel.MEDIUM, debug))
    return port("Terms of Service source", TermsOfService)


def parsed_message(facts: dict[str, object]) -> AIMessage:
    """The shape OpenAI strict structured output returns: facts on the raw message."""
    return AIMessage(content="", additional_kwargs={"parsed": facts})


def anthropic_message(facts: dict[str, object]) -> AIMessage:
    """The shape Anthropic json-schema output returns: the object as the message text."""
    return AIMessage(content=json.dumps(facts))


def status_error(kind: type[Any], message: str, status: int) -> Any:
    """Build one SDK's status error with the HTTP flavour that SDK actually carries."""
    library = httpx if kind.__module__.startswith("anthropic") else httpx2
    request = library.Request("POST", "https://provider.test")
    return kind(message, response=library.Response(status, request=request), body=None)


def test_a_parsed_object_is_an_extracted_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = extract_through_openai(monkeypatch, parsed_message(VALID_FACTS))

    assert outcome == Extracted(TermsOfService.model_validate(VALID_FACTS))


def test_an_answer_carrying_no_object_is_an_empty_extraction_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answered_without_committing = AIMessage(
        content="",
        tool_calls=[{"name": "TermsOfService", "args": {}, "id": "call-1", "type": "tool_call"}],
    )

    outcome = extract_through_openai(monkeypatch, answered_without_committing)

    assert outcome == EmptyExtraction()


def test_an_object_the_schema_rejects_is_a_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = extract_through_openai(
        monkeypatch, parsed_message({**VALID_FACTS, "effective_date": "nope"})
    )

    assert isinstance(outcome, ValidationFailure)
    assert "effective_date" in outcome.detail


def test_a_refusal_carried_by_the_raw_message_is_a_refusal_not_a_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = AIMessage(
        content="",
        additional_kwargs={"refusal": "The provider declined this extraction."},
    )

    outcome = extract_through_openai(monkeypatch, refused)

    assert isinstance(outcome, Refusal)
    assert "declined" in outcome.detail


def test_a_refusal_raised_by_the_provider_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = extract_through_openai(
        monkeypatch, OpenAIRefusalError("The provider declined this extraction.")
    )

    assert isinstance(outcome, Refusal)
    assert "declined" in outcome.detail


def test_a_refusal_passed_through_by_the_aggregator_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reachable through OpenRouter, not guaranteed: the refusal is read off a message field
    any OpenAI-format response may carry, so it depends on the upstream provider sending it."""
    refused = AIMessage(
        content="",
        additional_kwargs={"refusal": "The upstream provider declined this extraction."},
    )

    outcome = extract_through_openrouter(monkeypatch, refused)

    assert isinstance(outcome, Refusal)
    assert "declined" in outcome.detail


def test_an_anthropic_stop_reason_of_refusal_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic reports a refusal nowhere else. `stop_reason` reaches `response_metadata`
    because `langchain_core` merges the generation's `llm_output` into it."""
    refused = AIMessage(content="", response_metadata={"stop_reason": "refusal"})

    outcome = extract_through_anthropic(monkeypatch, refused)

    assert outcome == Refusal(detail="the model stopped with stop reason 'refusal'")


def test_an_anthropic_json_object_is_an_extracted_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = extract_through_anthropic(monkeypatch, anthropic_message(VALID_FACTS))

    assert outcome == Extracted(TermsOfService.model_validate(VALID_FACTS))


def test_an_anthropic_object_the_schema_rejects_is_a_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = extract_through_anthropic(
        monkeypatch, anthropic_message({**VALID_FACTS, "effective_date": "nope"})
    )

    assert isinstance(outcome, ValidationFailure)
    assert "effective_date" in outcome.detail


def test_an_openrouter_parsed_object_is_an_extracted_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = extract_through_openrouter(monkeypatch, parsed_message(VALID_FACTS))

    assert outcome == Extracted(TermsOfService.model_validate(VALID_FACTS))


@pytest.mark.parametrize(
    ("extract", "response"),
    [
        (extract_through_openai, parsed_message(VALID_FACTS)),
        (extract_through_anthropic, anthropic_message(VALID_FACTS)),
        (extract_through_openrouter, parsed_message(VALID_FACTS)),
    ],
)
def test_the_debug_stream_receives_the_raw_provider_message_on_every_provider(
    monkeypatch: pytest.MonkeyPatch,
    extract: Callable[..., Extraction],
    response: AIMessage,
) -> None:
    debug = StringIO()

    extract(monkeypatch, response, debug=debug)

    dumped = debug.getvalue()
    assert dumped.startswith("Raw model message: ")
    assert "State of New York" in dumped


@pytest.mark.parametrize(
    ("extract", "response"),
    [
        (extract_through_openai, parsed_message(VALID_FACTS)),
        (extract_through_anthropic, anthropic_message(VALID_FACTS)),
        (extract_through_openrouter, parsed_message(VALID_FACTS)),
    ],
)
def test_without_a_debug_stream_no_adapter_writes_to_stdout_or_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extract: Callable[..., Extraction],
    response: AIMessage,
) -> None:
    """`_report` is the sole writer of extraction output, on every provider."""
    extract(monkeypatch, response)

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
    provider = openai_stub(parsed_message(VALID_FACTS))
    provider.install(monkeypatch)

    build_openai_port(PortSettings("gpt-5-mini", reasoning, None))

    assert provider.configuration == {
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
    provider = anthropic_stub(anthropic_message(VALID_FACTS))
    provider.install(monkeypatch)

    build_anthropic_port(PortSettings("claude-sonnet-5", reasoning, None))

    assert provider.configuration == {
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
    provider = openrouter_stub(parsed_message(VALID_FACTS))
    provider.install(monkeypatch)

    build_openrouter_port(PortSettings("openai/gpt-5-nano", reasoning, None))

    assert provider.configuration["extra_body"] == {
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
    provider = openrouter_stub(parsed_message(VALID_FACTS))
    provider.install(monkeypatch)

    build_openrouter_port(PortSettings("openai/gpt-5-nano", ReasoningLevel.MEDIUM, None))

    assert provider.configuration == {
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
    ("build", "key", "model_id"),
    [
        (build_openai_port, "OPENAI_API_KEY", "gpt-5-nano"),
        (build_anthropic_port, "ANTHROPIC_API_KEY", "claude-sonnet-5"),
        (build_openrouter_port, "OPENROUTER_API_KEY", "openai/gpt-5-nano"),
    ],
)
def test_each_adapter_checks_only_its_own_key_before_building_the_model(
    monkeypatch: pytest.MonkeyPatch,
    build: PortFactory,
    key: str,
    model_id: str,
) -> None:
    """Every other provider's key is present and does not help, and the message names both the
    missing key and where to put it. Nothing is constructed, so a misconfigured run is free."""
    monkeypatch.setattr("extractor.extraction._load_env_file", lambda _: None)
    for other in ALL_KEYS:
        monkeypatch.setenv(other, "test-key")
    monkeypatch.delenv(key, raising=False)
    built: list[str] = []
    for target in ("ChatOpenAI", "ChatAnthropic"):
        monkeypatch.setattr(f"extractor.extraction.{target}", lambda **_: built.append("built"))

    with pytest.raises(ConfigurationError, match=rf"{key}.*extractor/\.env"):
        build(PortSettings(model_id, ReasoningLevel.MEDIUM, None))

    assert built == []


@pytest.mark.parametrize(
    ("build", "key", "model_id"),
    [
        (build_openai_port, "OPENAI_API_KEY", "gpt-5-nano"),
        (build_anthropic_port, "ANTHROPIC_API_KEY", "claude-sonnet-5"),
        (build_openrouter_port, "OPENROUTER_API_KEY", "openai/gpt-5-nano"),
    ],
)
def test_each_adapter_reads_its_key_from_the_app_local_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    build: PortFactory,
    key: str,
    model_id: str,
) -> None:
    """`extractor/.env`, so no key has to be exported into every shell."""
    monkeypatch.setenv(key, "test-key")
    for target in ("ChatOpenAI", "ChatAnthropic"):
        monkeypatch.setattr(f"extractor.extraction.{target}", lambda **_: None)
    loaded: list[Path] = []
    monkeypatch.setattr("extractor.extraction._load_env_file", loaded.append)

    build(PortSettings(model_id, ReasoningLevel.MEDIUM, None))

    assert loaded == [Path(__file__).resolve().parents[1] / ".env"]


def test_the_openai_binding_asks_the_provider_to_enforce_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0001: enforcement is provider-side, so the binding arguments are the contract.

    `strict=False` or a `method` of `function_calling` degrades enforcement to a polite
    request with no error and no warning, and every other offline test still passes.
    """
    openai_stub(parsed_message(VALID_FACTS)).install(monkeypatch)
    bindings = record_bindings(monkeypatch, ChatOpenAI)

    port = build_openai_port(PortSettings("gpt-5-nano", ReasoningLevel.MEDIUM, None))
    port("Terms of Service source", TermsOfService)

    assert bindings == [
        {
            "schema": TermsOfService,
            "method": "json_schema",
            "strict": True,
            "include_raw": True,
        }
    ]


def test_the_openrouter_binding_asks_the_aggregator_to_enforce_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openrouter_stub(parsed_message(VALID_FACTS)).install(monkeypatch)
    bindings = record_bindings(monkeypatch, ChatOpenAI)

    port = build_openrouter_port(PortSettings("openai/gpt-5-nano", ReasoningLevel.MEDIUM, None))
    port("Terms of Service source", TermsOfService)

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
    anthropic_stub(anthropic_message(VALID_FACTS)).install(monkeypatch)
    bindings = record_bindings(monkeypatch, ChatAnthropic)

    port = build_anthropic_port(PortSettings("claude-sonnet-5", ReasoningLevel.MEDIUM, None))
    port("Terms of Service source", TermsOfService)

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

    port = build_openrouter_port(PortSettings("openai/gpt-5-nano", ReasoningLevel.MEDIUM, None))
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
        status_error(BadRequestError, "bad request", 400),
        status_error(NotFoundError, "not found", 404),
        status_error(UnprocessableEntityError, "unprocessable", 422),
    ],
)
def test_each_rejected_request_class_becomes_a_provider_rejected_request(
    monkeypatch: pytest.MonkeyPatch,
    error: APIError,
) -> None:
    outcome = extract_through_openai(monkeypatch, error)

    assert outcome == ProviderRejectedRequest(detail=str(error))


@pytest.mark.parametrize(
    "error",
    [
        status_error(anthropic.BadRequestError, "bad request", 400),
        status_error(anthropic.NotFoundError, "not found", 404),
        status_error(anthropic.UnprocessableEntityError, "unprocessable", 422),
    ],
)
def test_each_anthropic_rejected_request_class_becomes_a_provider_rejected_request(
    monkeypatch: pytest.MonkeyPatch,
    error: anthropic.APIError,
) -> None:
    outcome = extract_through_anthropic(monkeypatch, error)

    assert outcome == ProviderRejectedRequest(detail=str(error))


def test_the_provider_family_base_class_becomes_a_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = APIError(
        "provider unavailable",
        request=httpx2.Request("POST", "https://provider.test"),
        body=None,
    )

    outcome = extract_through_openai(monkeypatch, error)

    assert outcome == ProviderFailure(detail=str(error))


def test_the_anthropic_family_base_class_becomes_a_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = anthropic.APIError(
        "provider unavailable",
        request=httpx.Request("POST", "https://provider.test"),
        body=None,
    )

    outcome = extract_through_anthropic(monkeypatch, error)

    assert outcome == ProviderFailure(detail=str(error))


def test_exhausted_aggregator_credit_becomes_a_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It has no named subclass, so it falls through to the base class — which is right, since
    re-running after topping up succeeds."""
    credit_exhausted = status_error(APIStatusError, "insufficient credits", 402)

    outcome = extract_through_openrouter(monkeypatch, credit_exhausted)

    assert outcome == ProviderFailure(detail=str(credit_exhausted))


@pytest.mark.parametrize(
    ("extract", "error"),
    [
        (extract_through_openai, status_error(NotFoundError, "model does not exist", 404)),
        (extract_through_openrouter, status_error(NotFoundError, "model does not exist", 404)),
        (
            extract_through_anthropic,
            status_error(anthropic.NotFoundError, "model does not exist", 404),
        ),
    ],
)
def test_an_unknown_model_id_is_a_provider_rejected_request_on_every_provider(
    monkeypatch: pytest.MonkeyPatch,
    extract: Callable[..., Extraction],
    error: BaseException,
) -> None:
    outcome = extract(monkeypatch, error)

    assert outcome == ProviderRejectedRequest(detail=str(error))


def test_the_env_file_defines_a_key_the_environment_lacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('OPENAI_API_KEY="file-key"\n', encoding="utf-8")

    _load_env_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "file-key"


def test_an_exported_key_beats_the_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shell is the operator's override; a stale file must not shadow it."""
    monkeypatch.setenv("OPENAI_API_KEY", "exported-key")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")

    _load_env_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "exported-key"


def test_the_env_file_ignores_blank_lines_and_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n# OPENAI_API_KEY=commented-out\n\nOPENAI_API_KEY=file-key\n", encoding="utf-8"
    )

    _load_env_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "file-key"


def test_a_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    """Returning rather than raising is the assertion: a first run has no `.env` and the key
    may be exported instead, so an absent file must not fail the extractor."""
    _load_env_file(tmp_path / ".env")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode 000 file regardless")
def test_an_unreadable_env_file_is_a_configuration_error(tmp_path: Path) -> None:
    """A present-but-unreadable file is a misconfiguration, not an `Unexpected error`."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    env_file.chmod(0o000)

    with pytest.raises(ConfigurationError, match=re.escape(str(env_file))):
        _load_env_file(env_file)


def test_a_non_utf8_env_file_is_a_configuration_error(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"OPENAI_API_KEY=caf\xe9\n")

    with pytest.raises(ConfigurationError, match=re.escape(str(env_file))):
        _load_env_file(env_file)
