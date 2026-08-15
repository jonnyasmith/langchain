# Extractor coding standards

Rules for changing code in this module. Every rule holds in the code today. Where an ADR is named,
it is authoritative — `docs/adr/`.

## Errors and outcomes

- MUST return a predictable failure as a value, never raise it: `load_source_document -> str | InputFailure`.
- MUST model mutually exclusive results as a closed union of `@dataclass(frozen=True, slots=True)`
  types, one per state, each with a docstring naming the state in domain terms. ADR-0002.
- NEVER model states as one type with nullable fields a caller interrogates in order. ADR-0002.
- MUST end every `match` over a union with `case unreachable: assert_never(unreachable)`.
- MUST extend every `match` when adding a union member. NEVER delete an `assert_never` case as cleanup.
- MUST store rendered detail as `str` on an outcome. NEVER store an exception instance.
- MUST raise only for construction failure (`ConfigurationError`) or programmer error. An outcome is
  never an exception.
- NEVER write `except Exception` anywhere except the existing top-level net in `main`.
- MUST make a return type carry its own guarantee, so no caller has to remember an ordering rule: a
  `str` from intake has already cleared `MAX_DOCUMENT_CHARACTERS`.

## Seams

- MUST declare a seam as a `Protocol` in the module that consumes it, not the module that implements it.
- NEVER name a provider, vendor, or framework type in a seam's signature.
- MUST keep a seam to one call. A second method means the boundary is wrong; do not widen the Protocol.
- MUST accept the narrowest interface (`TextIO`, a factory `Callable`) and return a concrete type.
- NEVER add a seam for an implementation that does not exist yet. ADR-0001.
- NEVER add a second `ExtractionPort` implementation: it typechecks while silently dropping
  `strict=True`. ADR-0001.
- MUST keep a single-call-site, single-implementation helper a plain function — not a seam.

## Framework containment

- MUST keep `with_structured_output`, `include_raw`, `parsing_error`, `OpenAIRefusalError`, prompt
  assembly, and `strict=True` inside `extraction.py`. ADR-0002.
- NEVER import LangChain, OpenAI, or provider types in `__main__.py`.
- MUST confine `Any` and `cast` to the framework boundary, and reduce the result to a domain type
  before returning it. The `cast(dict[str, Any], ...)` around `chain.invoke` is the only allowance.
- MUST write every schema field as required and nullable with no default (`str | None`, never
  `Optional[str] = None`) and give every field a description. ADR-0001.

## Layout, naming, visibility

- MUST keep `src/extractor/` flat: one module per responsibility. NEVER add `internal/`, `pkg/`,
  `cmd/`, or packages by layer.
- MUST prefix a module-private name with `_`. An unprefixed name is API.
- MUST follow PEP 8 and the spellings in `docs/agents/domain.md`. NEVER use `Err`-prefixed error
  names, `I`-prefixed interfaces, or abbreviated parameters.
- MUST add a new domain term to `docs/agents/domain.md` in the change that introduces it.
- MUST write a docstring that states a contract, an invariant, or a reason. NEVER restate the signature.
- NEVER add module-level mutable state or import-time I/O. `load_dotenv` runs inside `build_openai_port`.

## Published contracts

- MUST treat `ExitCode` numbers as published: `README.md` documents them and a test pins them.
  Renumbering a member is a breaking change.
- MUST assign an outcome's status and diagnostic in `_report` and nowhere else.
- MUST write the extracted object to stdout and every diagnostic to stderr.
- MUST update `ARCHITECTURE.md` in the change that moves ownership between modules, reverses a
  dependency, adds or removes an `Extraction` member, alters an exit number, or changes a hard
  limit. It describes the module as it is; a stale description is read with the same trust as
  a rule.

## Tests

- MUST write one test per contract, named as a sentence stating the behaviour.
- MUST reach `main` through its injected seams — `port_factory`, `StringIO`, `tmp_path`. NEVER
  subclass a framework type to reach `main`.
- MUST mark any test that calls a provider `live`, and MUST keep the default run passing offline with
  no API key.
- MUST skip a `live` test with no key, with a message naming what went unchecked. NEVER let it fail
  or silently pass. ADR-0003.
- MUST assert a decision's invariant directly where behaviour would not catch it: the strict binding
  arguments, the required-and-nullable field rule, the exit numbers.
- NEVER reduce `tests/test_live.py` to a shape check. Assert field presence (an unanswered field is
  null) and values with one faithful rendering; do not assert model wording. ADR-0003.

## Rejected — do not propose these

| Rejected | Use instead | Reason |
| --- | --- | --- |
| `(value, error)` pairs, generic `Result` | closed union + `match` + `assert_never` | Tuple returns substitute for absent sum types; this module has them. ADR-0002. |
| Error check after every call | one funnel in the adapter | Framework code raises across the boundary anyway; per-call wrappers add noise, not safety. |
| Wrapped error chains (`%w`, `errors.As`) | render detail to `str` at capture | No consumer inspects a cause; exception instances break outcome value-equality in tests. |
| Discard writer replacing an optional debug stream | keep the `None` branch | The branch elides formatting a large raw message. |
| `context.Context`-style threading | client configuration | One document, one call, one process. A timeout belongs on `ChatOpenAI`. |
| Concurrency primitives | nothing | One document per invocation; splitting a document is `rag/`'s concern. |
| Table-driven tests replacing named ones | one named test per contract | A sentence in the failure line beats a parametrised id. |
| Vendoring a dependency | depend on it | LangChain and Pydantic are this module's product surface. |
| A second provider adapter | nothing | Silently drops `strict=True`. ADR-0001. |

## Verification

Commands, dependency-group rules, the lockfile rule, and the single-config-file rule live in
`AGENTS.md`. Style is settled by `uv run ruff format .` and `uv run ruff check .`; this document adds
no rule a formatter enforces.
