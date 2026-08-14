from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIRefusalError
from pydantic import SecretStr

from extractor.extraction import (
    ConfigurationError,
    EmptyExtraction,
    Extracted,
    Extraction,
    Refusal,
    ValidationFailure,
    build_openai_port,
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


class StubbedProvider:
    """Substitutes the model the adapter builds, and records how it was configured."""

    def __init__(self, response: AIMessage | BaseException) -> None:
        self.response = response
        self.configuration: dict[str, Any] = {}
        self.built = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        def build(**configuration: Any) -> ChatOpenAI:
            self.configuration = configuration
            self.built = True
            return CannedOpenAIChatModel(
                api_key=SecretStr("test-key"), response=self.response, **configuration
            )

        monkeypatch.setattr("extractor.extraction.ChatOpenAI", build)


def extract_from(
    monkeypatch: pytest.MonkeyPatch,
    response: AIMessage | BaseException,
    *,
    debug: StringIO | None = None,
) -> Extraction:
    StubbedProvider(response).install(monkeypatch)
    port = build_openai_port("gpt-5-nano", debug)
    return port("Terms of Service source", TermsOfService)


def parsed_message(facts: dict[str, object]) -> AIMessage:
    """The shape OpenAI strict structured output returns: facts on the raw message."""
    return AIMessage(content="", additional_kwargs={"parsed": facts})


def test_a_parsed_object_is_an_extracted_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    outcome = extract_from(monkeypatch, parsed_message(VALID_FACTS))

    assert outcome == Extracted(TermsOfService.model_validate(VALID_FACTS))


def test_an_answer_carrying_no_object_is_an_empty_extraction_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answered_without_committing = AIMessage(
        content="",
        tool_calls=[{"name": "TermsOfService", "args": {}, "id": "call-1", "type": "tool_call"}],
    )

    outcome = extract_from(monkeypatch, answered_without_committing)

    assert outcome == EmptyExtraction()


def test_an_object_the_schema_rejects_is_a_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = extract_from(monkeypatch, parsed_message({**VALID_FACTS, "effective_date": "nope"}))

    assert isinstance(outcome, ValidationFailure)
    assert "effective_date" in outcome.detail


def test_a_refusal_carried_by_the_raw_message_is_a_refusal_not_a_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = AIMessage(
        content="",
        additional_kwargs={"refusal": "The provider declined this extraction."},
    )

    outcome = extract_from(monkeypatch, refused)

    assert isinstance(outcome, Refusal)
    assert "declined" in outcome.detail


def test_a_refusal_raised_by_the_provider_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = extract_from(
        monkeypatch, OpenAIRefusalError("The provider declined this extraction.")
    )

    assert isinstance(outcome, Refusal)
    assert "declined" in outcome.detail


def test_the_debug_stream_receives_the_raw_provider_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    debug = StringIO()

    extract_from(monkeypatch, parsed_message(VALID_FACTS), debug=debug)

    dumped = debug.getvalue()
    assert dumped.startswith("Raw model message: ")
    assert "State of New York" in dumped


def test_without_a_debug_stream_the_raw_message_is_not_dumped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    extract_from(monkeypatch, parsed_message(VALID_FACTS))

    assert capsys.readouterr() == ("", "")


def test_the_model_is_the_named_cheap_deterministic_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = StubbedProvider(parsed_message(VALID_FACTS))
    provider.install(monkeypatch)

    build_openai_port("gpt-5-mini", None)

    assert provider.configuration == {
        "model": "gpt-5-mini",
        "reasoning_effort": "none",
        "temperature": 0,
        "timeout": 60,
        "max_retries": 2,
    }


def test_a_missing_api_key_fails_before_the_model_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = StubbedProvider(parsed_message(VALID_FACTS))
    provider.install(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("extractor.extraction.load_dotenv", lambda _: False)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_openai_port("gpt-5-nano", None)

    assert not provider.built


def test_the_binding_asks_the_provider_to_enforce_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0001: enforcement is provider-side, so the binding arguments are the contract.

    `strict=False` or a `method` of `function_calling` degrades enforcement to a polite
    request with no error and no warning, and every other offline test still passes.
    """
    StubbedProvider(parsed_message(VALID_FACTS)).install(monkeypatch)
    bindings: list[dict[str, Any]] = []
    bind = ChatOpenAI.with_structured_output

    def record(model: ChatOpenAI, schema: Any = None, **kwargs: Any) -> Any:
        bindings.append({"schema": schema, **kwargs})
        return bind(model, schema, **kwargs)

    monkeypatch.setattr(ChatOpenAI, "with_structured_output", record)

    port = build_openai_port("gpt-5-nano", None)
    port("Terms of Service source", TermsOfService)

    assert bindings == [
        {
            "schema": TermsOfService,
            "method": "json_schema",
            "strict": True,
            "include_raw": True,
        }
    ]


def test_the_key_is_read_from_the_app_local_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """`extractor/.env`, so the key does not have to be exported into every shell."""
    StubbedProvider(parsed_message(VALID_FACTS)).install(monkeypatch)
    loaded: list[Path] = []

    def record(path: Path) -> bool:
        loaded.append(path)
        return True

    monkeypatch.setattr("extractor.extraction.load_dotenv", record)

    build_openai_port("gpt-5-nano", None)

    assert loaded == [Path(__file__).resolve().parents[1] / ".env"]
