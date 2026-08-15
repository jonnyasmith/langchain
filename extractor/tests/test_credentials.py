"""Credential resolution, exercised through `required_key` — the module's whole interface.

Every test points `ENV_FILE` at a temporary file rather than reaching for the private loader,
so what is asserted is the guarantee the adapters depend on: a returned key is usable, and
anything else raises `ConfigurationError` before a model is constructed.
"""

import os
import re
from pathlib import Path

import pytest

from extractor.credentials import ENV_FILE, ConfigurationError, required_key


def test_the_credentials_file_is_the_app_local_dotenv() -> None:
    """`extractor/.env`, so no key has to be exported into every shell. The adapters name this
    path in their diagnostics, so moving it silently would misdirect every operator."""
    app_local_dotenv = Path(__file__).resolve().parents[1] / ".env"

    assert app_local_dotenv == ENV_FILE


@pytest.fixture(autouse=True)
def unexported_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a shell that does not already define the key under test."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def point_env_file_at(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr("extractor.credentials.ENV_FILE", path)


def test_the_env_file_defines_a_key_the_environment_lacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('OPENAI_API_KEY="file-key"\n', encoding="utf-8")
    point_env_file_at(monkeypatch, env_file)

    assert required_key("OPENAI_API_KEY") == "file-key"


def test_an_exported_key_beats_the_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shell is the operator's override; a stale file must not shadow it."""
    monkeypatch.setenv("OPENAI_API_KEY", "exported-key")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    point_env_file_at(monkeypatch, env_file)

    assert required_key("OPENAI_API_KEY") == "exported-key"


def test_the_env_file_ignores_blank_lines_and_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n# OPENAI_API_KEY=commented-out\n\nOPENAI_API_KEY=file-key\n", encoding="utf-8"
    )
    point_env_file_at(monkeypatch, env_file)

    assert required_key("OPENAI_API_KEY") == "file-key"


def test_a_missing_env_file_is_not_an_error_when_the_key_is_exported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A first run has no `.env` and the key may be exported instead, so an absent file must
    not fail the extractor."""
    monkeypatch.setenv("OPENAI_API_KEY", "exported-key")
    point_env_file_at(monkeypatch, tmp_path / ".env")

    assert required_key("OPENAI_API_KEY") == "exported-key"


def test_an_absent_key_names_the_key_and_where_to_put_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The diagnostic is the operator's only instruction, so both halves are the contract."""
    point_env_file_at(monkeypatch, tmp_path / ".env")

    with pytest.raises(ConfigurationError, match=r"OPENAI_API_KEY.*extractor/\.env"):
        required_key("OPENAI_API_KEY")


def test_an_empty_value_is_treated_as_an_absent_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty key reaches the provider as an authentication failure, which names the wrong
    cause; this is a misconfiguration and must be reported as one."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    point_env_file_at(monkeypatch, env_file)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        required_key("OPENAI_API_KEY")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode 000 file regardless")
def test_an_unreadable_env_file_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A present-but-unreadable file is a misconfiguration, not an `Unexpected error`."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    env_file.chmod(0o000)
    point_env_file_at(monkeypatch, env_file)

    with pytest.raises(ConfigurationError, match=re.escape(str(env_file))):
        required_key("OPENAI_API_KEY")


def test_a_non_utf8_env_file_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"OPENAI_API_KEY=caf\xe9\n")
    point_env_file_at(monkeypatch, env_file)

    with pytest.raises(ConfigurationError, match=re.escape(str(env_file))):
        required_key("OPENAI_API_KEY")
