# Researcher vocabulary

Terms specific to this module. Solution-wide vocabulary lives in the root `docs/agents/domain.md`.

| Term | What to understand |
| --- | --- |
| `@tool` | Declares a Python function as callable by the model. The docstring is the contract: it must state when to use the tool and what the arguments mean, because that text is all the model sees. |
| Tool input schema | A Pydantic model per tool. Strict types stop the model from inventing argument shapes. |
| `.bind_tools()` | Attaches the tool JSON schemas to the model so it can emit tool calls rather than prose. |
| `create_tool_calling_agent` | Builds the runnable that decides the next action from the prompt, tools, and scratchpad. |
| `AgentExecutor` | Runs the loop: call the agent, execute the chosen tool, feed the observation back, repeat. Needs a step limit to stay bounded. |
| Agent scratchpad | The running record of prior tool calls and observations, fed back on each iteration. |
| Tool error handling | A failed API call MUST return an error observation to the loop, not raise through the executor and kill the run. |
| Intermediate steps | The executor's trace of tool calls. The thing to inspect when the agent picks the wrong tool. |
