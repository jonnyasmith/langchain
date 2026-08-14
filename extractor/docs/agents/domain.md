# Extractor vocabulary

Terms specific to this module. Solution-wide vocabulary lives in the root `docs/agents/domain.md`.

| Term | What to understand |
| --- | --- |
| Structured output | `.with_structured_output(Schema)` binds a Pydantic model to the call, so the provider returns a validated object. It is the extraction contract — never post-hoc string parsing or regex repair of malformed JSON. |
| `ChatPromptTemplate` | Holds the system and human messages as roles. System message carries the extraction rules; human message carries the raw text. |
| Pydantic schema | Field names, types, and descriptions are prompt surface as well as validation: the model reads them. `Optional` means "may be absent from the source", not "nice to have". |
| Conversational filler | The failure mode this module exists to eliminate. Native structured output bindings remove it by construction; a parser bolted on afterwards does not. |
| Parser | `StrOutputParser` and friends unwrap a message into a plain value. Reach for one only where no schema applies. |
| Source document | The single messy input handed to one run — a scraped page, an email thread. One document per invocation; splitting a document is `rag/`'s lesson, not this one. |
| Named schema | An extraction target the tool ships with, chosen by name on the command line. The set of names is fixed and small; this is not a tool for loading arbitrary user schemas. _Avoid_: template, model, profile. |
| Extraction | One run: source document plus named schema in, one validated object out. Either it produces that object or it fails — there is no partial result. |
| Empty extraction | The model answered but committed to nothing, so no object comes back. A failure, not a success with a null value — it must never reach stdout. |
| Refusal | The model declined to extract at all. Distinct from an empty extraction and from a validation failure, and reported as such. |
| Validation failure | The model produced an object that the schema rejects. One of the three named ways an extraction fails, alongside an empty extraction and a refusal; each is reported distinctly rather than collapsed into a generic error. |
| Absent field | A field the source document genuinely does not answer. Recorded as null — never inferred, never filled with a plausible value. Distinct from a validation failure, which is the model getting a field wrong rather than leaving it open. |
| Extraction outcome | The single value one extraction attempt returns: exactly one of an extracted object, an empty extraction, a validation failure, or a refusal. A closed union, so every caller handles all four and the compiler proves it — never a bag of nullable fields the caller has to interrogate in the right order. |
| Extraction port | The seam `main` depends on: document plus named schema in, one extraction outcome out. A narrow interface, not a provider handle — production satisfies it with the OpenAI adapter, which owns prompt assembly, the strict structured-output binding, and the raw-message debug dump; tests satisfy it by staging an outcome directly. Its depth is that no provider type crosses it. |