from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai.chat_models.base import OpenAIRefusalError

from extractor.__main__ import main


class ToolCallingFakeChatModel(BaseChatModel):
    response: Any

    @property
    def _llm_type(self) -> str:
        return "tool-calling-fake"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if isinstance(self.response, Exception):
            raise self.response
        return ChatResult(generations=[ChatGeneration(message=self.response)])


def tool_response(arguments: Mapping[str, object], *, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": "TermsOfService",
                "args": dict(arguments),
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


def configured_model(
    *,
    model: str,
    reasoning_effort: str,
    temperature: int,
) -> BaseChatModel:
    extracted = {
        "governing_law": model,
        "arbitration_required": reasoning_effort == "none",
        "arbitration_clause": reasoning_effort,
        "liability_cap": str(temperature),
        "termination_notice_period": None,
        "data_retention_period": None,
        "effective_date": None,
    }
    return ToolCallingFakeChatModel(response=tool_response(extracted))


def test_a_valid_extraction_is_the_only_stdout_and_exits_zero() -> None:
    expected = {
        "governing_law": "State of New York",
        "arbitration_required": True,
        "arbitration_clause": "Disputes must be resolved by binding arbitration.",
        "liability_cap": "$100",
        "termination_notice_period": "30 days",
        "data_retention_period": None,
        "effective_date": "2026-01-01",
    }
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", "-"],
        stdin=StringIO("Terms of Service source"),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=tool_response(expected)),
    )

    assert exit_code == 0
    assert json.loads(stdout.getvalue()) == expected
    assert stderr.getvalue() == ""


def test_an_invalid_model_value_is_a_validation_failure() -> None:
    invalid = {
        "governing_law": None,
        "arbitration_required": None,
        "arbitration_clause": None,
        "liability_cap": None,
        "termination_notice_period": None,
        "data_retention_period": None,
        "effective_date": "not-a-date",
    }
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", "-"],
        stdin=StringIO("Terms of Service source"),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=tool_response(invalid)),
    )

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert "validation failure" in stderr.getvalue().lower()
    assert "effective_date" in stderr.getvalue()


def test_a_model_answer_without_an_object_is_an_empty_extraction() -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", "-"],
        stdin=StringIO("Terms of Service source"),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=AIMessage(content="No extracted facts")),
    )

    assert exit_code == 3
    assert stdout.getvalue() == ""
    assert "empty extraction" in stderr.getvalue().lower()


def test_a_provider_refusal_is_reported_separately() -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", "-"],
        stdin=StringIO("Terms of Service source"),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(
            response=OpenAIRefusalError("The provider declined this extraction.")
        ),
    )

    assert exit_code == 4
    assert stdout.getvalue() == ""
    assert "refusal" in stderr.getvalue().lower()
    assert "declined" in stderr.getvalue().lower()


def test_an_unexpected_provider_error_uses_the_generic_failure_exit() -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", "-"],
        stdin=StringIO("Terms of Service source"),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=ConnectionError("network unavailable")),
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "unexpected error" in stderr.getvalue().lower()
    assert "network unavailable" in stderr.getvalue().lower()


def test_debug_dumps_the_raw_model_message_to_stderr() -> None:
    extracted = {
        "governing_law": None,
        "arbitration_required": None,
        "arbitration_clause": None,
        "liability_cap": None,
        "termination_notice_period": None,
        "data_retention_period": None,
        "effective_date": None,
    }
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--debug", "--schema", "tos", "-"],
        stdin=StringIO("Terms of Service source"),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(
            response=tool_response(extracted, content="raw provider message")
        ),
    )

    assert exit_code == 0
    assert json.loads(stdout.getvalue()) == extracted
    assert "raw provider message" in stderr.getvalue()


def test_a_source_file_is_read_for_extraction() -> None:
    expected = {
        "governing_law": "State of New York",
        "arbitration_required": True,
        "arbitration_clause": "Binding arbitration administered by the AAA.",
        "liability_cap": "USD 100",
        "termination_notice_period": "30 days",
        "data_retention_period": None,
        "effective_date": "2026-01-01",
    }
    stdout = StringIO()
    stderr = StringIO()
    fixture = Path(__file__).parent / "fixtures" / "terms.html"

    exit_code = main(
        ["--schema", "tos", str(fixture)],
        stdin=StringIO(),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=tool_response(expected)),
    )

    assert exit_code == 0
    assert json.loads(stdout.getvalue()) == expected
    assert stderr.getvalue() == ""


def test_listing_schemas_needs_no_input_or_model() -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--list-schemas"],
        stdin=StringIO(),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue() == "tos\n"
    assert stderr.getvalue() == ""


def test_an_unknown_schema_name_lists_valid_names() -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "contract", "-"],
        stdin=StringIO("source"),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=AIMessage(content="unused")),
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "unknown schema" in stderr.getvalue().lower()
    assert "tos" in stderr.getvalue()


def test_a_missing_input_file_is_reported_as_an_input_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.html"
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", str(missing)],
        stdin=StringIO(),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=AIMessage(content="unused")),
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "input file" in stderr.getvalue().lower()
    assert str(missing) in stderr.getvalue()
    assert "validation failure" not in stderr.getvalue().lower()


def test_an_unreadable_input_path_is_reported_as_an_input_error(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", str(tmp_path)],
        stdin=StringIO(),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=AIMessage(content="unused")),
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "input file error" in stderr.getvalue().lower()
    assert str(tmp_path) in stderr.getvalue()


def test_a_missing_api_key_is_a_named_configuration_error(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", "-"],
        stdin=StringIO("source"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "configuration error" in stderr.getvalue().lower()
    assert "openai_api_key" in stderr.getvalue().lower()


def test_an_oversize_document_is_refused_before_calling_the_model() -> None:
    stdout = StringIO()
    stderr = StringIO()
    document = "x" * 100_001

    exit_code = main(
        ["--schema", "tos", "-"],
        stdin=StringIO(document),
        stdout=stdout,
        stderr=stderr,
        model=ToolCallingFakeChatModel(response=AssertionError("the provider must not be called")),
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert "100,000" in stderr.getvalue()
    assert "100,001" in stderr.getvalue()


def test_model_override_keeps_deterministic_cheap_settings(monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("extractor.__main__.ChatOpenAI", configured_model)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--model", "gpt-5", "--schema", "tos", "-"],
        stdin=StringIO("source"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert json.loads(stdout.getvalue()) == {
        "governing_law": "gpt-5",
        "arbitration_required": True,
        "arbitration_clause": "none",
        "liability_cap": "0",
        "termination_notice_period": None,
        "data_retention_period": None,
        "effective_date": None,
    }
    assert stderr.getvalue() == ""


def test_default_model_loads_the_app_local_dotenv(monkeypatch: Any) -> None:
    app_dotenv = Path(__file__).parents[1] / ".env"

    def load_app_env(dotenv_path: str | Path) -> bool:
        if Path(dotenv_path) == app_dotenv:
            monkeypatch.setenv("OPENAI_API_KEY", "dotenv-test-key")
            return True
        return False

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("extractor.__main__.load_dotenv", load_app_env)
    monkeypatch.setattr("extractor.__main__.ChatOpenAI", configured_model)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--schema", "tos", "-"],
        stdin=StringIO("source"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert json.loads(stdout.getvalue())["governing_law"] == "gpt-5-nano"
    assert stderr.getvalue() == ""
