"""Provider credentials: the app-local `.env` and the per-provider key check.

Not provider-shaped, so it does not belong in `extraction.py`. The interface is one call:
`required_key(name)` either returns a usable key or raises before anything is constructed
or sent. Where the key came from — the shell or the file — is implementation.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
"""The app-local credentials file, so no key has to be exported into every shell."""


class ConfigurationError(Exception):
    """The extractor cannot construct its provider adapter."""


def _load_env_file(path: Path) -> None:
    """Define any `KEY=value` the file declares that the environment does not already set.

    Ten lines rather than a dependency, per the coding standards. An exported variable always
    wins, so a stale file cannot shadow the shell. No interpolation, `export` prefixes, or
    multi-line values: the extractor reads one key per provider.

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


def required_key(name: str) -> str:
    """Read one provider's key, or fail before anything is constructed or sent.

    `ENV_FILE` is read at every call rather than at import, so no credential is loaded by the
    act of importing the module, and an absent file is not an error — the key may be exported.
    """
    _load_env_file(ENV_FILE)
    key = os.getenv(name)
    if not key:
        raise ConfigurationError(f"missing {name}; set it in extractor/.env")
    return key
