# Synthesizer vocabulary

Terms specific to this module. Solution-wide vocabulary lives in the root `docs/agents/domain.md`.

| Term | What to understand |
| --- | --- |
| `RunnableParallel` | Runs several chains concurrently over the same input and returns a dict keyed by branch name. The reason three analyses cost roughly one analysis of wall time. |
| Analysis branch | One perspective on the topic — economic, psychological, environmental — as an independent chain. |
| `RunnableBranch` | Conditional routing: the first matching predicate wins, with a mandatory default. Picks which chains a given topic warrants. |
| `RunnableLambda` | Lifts a plain function into the chain, for routing logic and key reshaping that no built-in covers. |
| Key mapping | The real difficulty of this module. Every key the synthesis prompt expects must survive the trip from the parallel branches; a renamed or dropped key fails at prompt formatting, not at type-check time. |
| Synthesis prompt | The final template that reads the branch outputs and writes one executive summary. |
