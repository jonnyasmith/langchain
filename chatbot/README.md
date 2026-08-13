# chatbot

Terminal chatbot that remembers the conversation and streams its answer token by token.

History is injected into the prompt via `MessagesPlaceholder` and persisted per session by `RunnableWithMessageHistory` over a local SQLite store. Responses are consumed with `.stream()`, so text appears as it is generated instead of after a long pause.

## Features (planned)

- Multi-turn history keyed by session id
- SQLite-backed message store, surviving restarts
- Typewriter streaming that never blocks the input loop
- Modern LCEL history injection (no deprecated memory classes)

## Status

Not implemented.

## Requirements

- Python (version TBD when a `pyproject.toml` is added)
- [uv](https://docs.astral.sh/uv/) for dependency management
- A model provider API key in `.env`

## Install

```bash
uv sync
```

## Test

```bash
uv run pytest
```

## Run

```bash
uv run python -m chatbot
```
