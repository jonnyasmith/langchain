# Coding Standards

We write Python. These are the architectural and coding standards that keep our systems maintainable, readable, and resilient.

## 1. Dependency Management: Own Your Logic

**Rule:** A little copying is better than a little dependency.

* **Approach:** Before adding a third-party package for a minor utility, or a shared internal module for a handful of dataclasses, check whether the functionality fits in 20-50 lines of our own code.
* **Rationale:** Every dependency adds supply chain risk, maintenance overhead, and tight coupling. We prefer isolated, duplicated utility logic over brittle, tightly coupled shared modules.

## 2. Control Flow: Maintain "Line of Sight"

**Rule:** Return early and avoid deep nesting.

* **Approach:** Handle errors, edge cases, and preconditions at the top of the function using guard clauses. The happy path stays un-indented, running down the left of the screen.
* **Rationale:** Deep `if`/`else` nesting increases cognitive load. Guard clauses make it immediately obvious what causes a function to exit.

## 3. Type Design: Robust Signatures

**Rule:** Accept the most general type, return the most specific one.

* **Approach:** Parameters should demand only the behaviour the function actually uses — take a `Protocol`, an `Iterable`, or a `Sequence` rather than a concrete class or a `list`. Return types should be concrete: a specific dataclass, a `list`, a named union.
* **Rationale:** A narrow parameter type maximises flexibility for the caller, who can pass anything matching the shape. A concrete return type lets us add fields and methods later without breaking callers.
* **Closures:** A function that returns a closure is the exception. A closure has no denotable concrete type, so a callable `Protocol` is the most specific type available and is the correct return annotation.
* **Note:** This is about coupling, not input tolerance. We do not accept malformed input — see rule 6.

## 4. Architecture: Favour Composition Over Inheritance

**Rule:** Build complex behaviour by assembling small, focused components.

* **Approach:** Avoid deep class hierarchies and abstract base classes. Inject single-purpose callables or `Protocol`-typed collaborators to combine behaviours.
* **Rationale:** Inheritance trees become rigid and brittle. Composition lets us swap or fake a behaviour without untangling a family tree.

## 5. Error Handling: Predictable and Explicit Failures

**Rule:** Treat errors as expected values, not invisible control flow.

* **Approach:** Reserve `raise` for genuinely unrecoverable states — a missing API key, a broken invariant. Model expected failures as named dataclasses in a union return type, one variant per outcome the caller must handle.
* **Rationale:** The caller reads the signature and knows every way the call can fail, which forces them to acknowledge each one.
* **Exhaustiveness:** Python does not check unions for us. Consume them with `match` and close the block with `case _ as unreachable: assert_never(unreachable)`, so mypy fails the build when a new variant appears.

### Third-party boundaries

Libraries raise. LangChain and the OpenAI SDK signal expected failures — refusals, malformed requests, upstream errors — as exceptions, so the only way to honour this rule is to catch them and map them into our own outcomes.

* **Approach:** Wrap the third-party call in a thin translation layer that catches its exceptions and returns our variants. Keep the layer thin: catch, map, return. No business logic. Everything above the boundary sees only the union.
* **This is compliance, not an exemption.** The boundary function is where exceptions stop, which is precisely what lets the rest of the module treat errors as values.
* **The union is only as complete as the `except` clauses.** Nothing verifies that we caught everything the library can raise, and the surface shifts on upgrade. Where an escaped exception would be worse than a swallowed bug, close the boundary with a final `except Exception` mapping to a generic failure variant. Make that call deliberately.

## 6. Type Safety: Avoid the Escape Hatch

**Rule:** The empty type says nothing.

* **Approach:** Do not reach for `Any`, a bare `dict[str, Any]`, or `cast` to silence the type checker. Where a value is genuinely unknown, narrow it with `isinstance` before use. Where a library forces our hand — an untyped return we must reshape — confine the `cast` to the boundary and give the result a typed shape immediately.
* **Rationale:** Bypassing the type system converts build-time errors into production crashes.

## 7. Resource Management: Clean Up Deterministically

**Rule:** Acquisition and release belong in the same lexical scope.

* **Approach:** Open files, acquire locks, and establish connections with `with`. Where setup and teardown are non-trivial, write a `@contextmanager` so both halves sit in one function and callers get a single `with` block.
* **Rationale:** Relying on reference counting for unmanaged resources, or putting cleanup at the bottom of a long function, guarantees leaks on the error path.

## 8. Testing: One Behaviour Per Test

**Rule:** Name the behaviour in the test; parametrise only over data.

* **Approach:** Default to a separate test per behaviour, named as a sentence stating what must hold — `test_a_missing_path_is_a_named_input_failure`. Reach for `@pytest.mark.parametrize` when the cases differ only in their inputs and share one assertion, such as boundary values or encoding variants.
* **Rationale:** pytest reports the failing test by name and introspects the failing assertion, so a descriptive test tells us what broke without opening the file. Collapsing behaviours that assert different things into one table hides that signal and couples unrelated cases to a single assertion.
