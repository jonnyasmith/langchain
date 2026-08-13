# Domain vocabulary

What you need in order to steer work on this solution. Each module README states the problem and the LangChain internals that module exercises.

| Domain | What to understand | Where it shows up |
| --- | --- | --- |
| LCEL composition | How the `\|` operator wires runnables, and how dicts flow between them without breaking | every module |
| Runnable protocol | `invoke` / `stream` / `batch` and their async twins as one interface over every component | every module |
| Structured output | Binding a Pydantic schema with `.with_structured_output()` so types are guaranteed, not parsed out of prose | `extractor/` |
| Prompt templating | `ChatPromptTemplate` roles and `MessagesPlaceholder` for injecting prior turns | `extractor/`, `chatbot/` |
| Message history | `RunnableWithMessageHistory`, session ids, and a persistent store — not the deprecated memory classes | `chatbot/` |
| Streaming | Iterating chunks as they are generated instead of blocking on the full response | `chatbot/` |
| Retrieval mechanics | Chunking strategy, embedding distance, and how retrieved documents are formatted into the prompt | `rag/` |
| Retriever composition | Multi-query and contextual compression retrievers stacked over a base vector store | `rag/` |
| Passthrough wiring | `RunnablePassthrough.assign()` for adding keys without dropping the ones already in flight | `rag/`, `synthesizer/` |
| Tool binding | How a model reads a JSON schema to decide which Python function to call, and why the docstring is the contract | `researcher/` |
| Parallelism and routing | `RunnableParallel` for concurrent branches, `RunnableBranch` / `RunnableLambda` for conditional paths | `synthesizer/` |
