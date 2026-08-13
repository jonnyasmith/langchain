# synthesizer

Takes a topic, analyses it from several angles at once, and writes one executive summary.

`RunnableParallel` fires the economic, psychological, and environmental chains concurrently; `RunnableBranch` picks which analyses a given topic even warrants. The branch outputs are mapped into a final synthesis prompt, so the interesting work is dictionary discipline: every key the summary prompt expects has to survive the trip through the pipeline.

## Features (planned)

- Concurrent analysis branches via `RunnableParallel`
- Conditional routing with `RunnableBranch` / `RunnableLambda`
- Explicit key mapping into the synthesis prompt
- Final executive summary with per-branch sections

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
uv run python -m synthesizer
```
