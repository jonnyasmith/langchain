# Each app is an independent project, not a uv workspace member

The five apps share a repository but nothing else: they are separate exercises in different LangChain internals, and any shared helper layer would hide the very wiring each one exists to teach. Each directory therefore carries its own `pyproject.toml` and its own lockfile, resolved independently by `uv`, with no root workspace and no shared package.

## Consequences

Dependency versions can drift between apps, and an upgrade has to be applied five times. That is accepted — it also means each app pins whatever the exercise needs, and one app's breakage never blocks another.
