# researcher

Tool-calling ReAct agent that decides which external function to reach for.

Three tools: web search, arithmetic, and local file reads. Each is declared with the `@tool` decorator over a strict Pydantic input schema, bound to the model with `.bind_tools()`, and driven by `create_tool_calling_agent` inside an `AgentExecutor`. Tool descriptions are the steering surface — they tell the model when and how to call each function. A failing tool returns an error to the loop instead of taking the executor down.

## Features (planned)

- Web search, calculator, and file-read tools
- Pydantic input schemas per tool
- Agent executor loop with a step limit
- Per-tool error handling, so one failure does not end the run
- Traceable intermediate steps

## Status

Not implemented.

## Requirements

- Python (version TBD when a `pyproject.toml` is added)
- [uv](https://docs.astral.sh/uv/) for dependency management
- A model provider API key in `.env`
- A search provider API key in `.env`

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
uv run python -m researcher
```
