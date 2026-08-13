# rag

Document Q&A CLI that goes past naive top-k retrieval.

Documents are loaded, split with `RecursiveCharacterTextSplitter`, and embedded into a vector store. Retrieval is composed rather than singular: a `MultiQueryRetriever` rewrites the question several ways to catch semantic misses, and a `ContextualCompressionRetriever` strips the useless paragraphs before anything reaches the prompt. Context is wired in explicitly with `RunnablePassthrough.assign()`.

## Features (planned)

- Document loaders and recursive chunking
- Local vector store with persisted index
- Multi-query and contextual compression retrievers, composed
- Explicit LCEL pipeline end to end (no `RetrievalQA` wrapper)
- Cited source documents in the answer

## Status

Not implemented.

## Requirements

- Python (version TBD when a `pyproject.toml` is added)
- [uv](https://docs.astral.sh/uv/) for dependency management
- A model provider API key in `.env` (chat and embeddings)

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
uv run python -m rag
```
