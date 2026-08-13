# RAG vocabulary

Terms specific to this module. Solution-wide vocabulary lives in the root `docs/agents/domain.md`.

| Term | What to understand |
| --- | --- |
| Document loader | Turns a source file into `Document` objects carrying `page_content` and `metadata`. Metadata is what makes citation possible later. |
| `RecursiveCharacterTextSplitter` | Splits on a descending list of separators so chunks break at natural boundaries. Chunk size and overlap decide what the retriever can possibly find. |
| Embedding distance | Retrieval ranks by vector similarity, not meaning. Near-duplicates crowd the top-k; paraphrases can miss entirely. |
| Vector store | Persisted index of chunk embeddings, exposing a base retriever. |
| `MultiQueryRetriever` | Asks the model to rewrite the question several ways and unions the results, covering semantic misses a single phrasing would drop. |
| `ContextualCompressionRetriever` | Filters or trims retrieved documents before they reach the prompt, so irrelevant paragraphs do not consume context. |
| `RunnablePassthrough.assign()` | Adds retrieved context as a new key while keeping the keys already in flight. How context enters the chain explicitly. |
| Naive RAG | The baseline this module deliberately exceeds. The `RetrievalQA.from_chain_type` wrapper MUST NOT be used — the pipeline is built from explicit LCEL components so document formatting stays visible. |
