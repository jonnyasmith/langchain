# Chatbot vocabulary

Terms specific to this module. Solution-wide vocabulary lives in the root `docs/agents/domain.md`.

| Term | What to understand |
| --- | --- |
| `MessagesPlaceholder` | The slot in the prompt template where prior turns are injected. Without it, history has nowhere to land. |
| `RunnableWithMessageHistory` | Wraps a chain, resolves a session id to a history store, and appends both the input and the response after each turn. This is the modern replacement for `ConversationBufferMemory` and the other deprecated memory classes — those MUST NOT be used. |
| Session id | The key that separates one conversation from another in the store. Configured per invocation, not baked into the chain. |
| Chat message history store | Where turns persist. Local SQLite here, so a restart does not lose the conversation. |
| `.stream()` | Yields chunks as the model generates them. The iterator must not block the terminal input loop; partial chunks are printed as they arrive, then committed to history once complete. |
