# Extractor vocabulary

Terms specific to this module. Solution-wide vocabulary lives in the root `docs/agents/domain.md`.

| Term | What to understand |
| --- | --- |
| Structured output | `.with_structured_output(Schema)` binds a Pydantic model to the call, so the provider returns a validated object. It is the extraction contract — never post-hoc string parsing or regex repair of malformed JSON. |
| `ChatPromptTemplate` | Holds the system and human messages as roles. System message carries the extraction rules; human message carries the raw text. |
| Pydantic schema | Field names, types, and descriptions are prompt surface as well as validation: the model reads them. `Optional` means "may be absent from the source", not "nice to have". |
| Conversational filler | The failure mode this module exists to eliminate. Native structured output bindings remove it by construction; a parser bolted on afterwards does not. |
| Parser | `StrOutputParser` and friends unwrap a message into a plain value. Reach for one only where no schema applies. |
