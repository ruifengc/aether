# Aether

**English** · [简体中文](README.zh-CN.md)

A small, message-driven runtime for LLM loops. Aether provides asynchronous submission, internal processing, queries and subscriptions, with continuous context, automatic compaction, tool registration and bounded model-error recovery. Python 3.11+; the core uses only the standard library. The terminal application lives in a separate repository.

[API reference (Chinese)](docs/api.md) · [Architecture (Chinese)](docs/architecture.md) · [Collaboration example](examples/collaboration.py)

## Run with a real LLM

Clone the repository and install the optional HTTP adapter:

```bash
git clone https://github.com/ruifengc/aether.git
cd aether
uv sync --extra live --no-dev
```

Your application supplies model credentials and settings. The library does not discover environment variables or configuration files. In the example below, the application reads `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` from its environment. `LLM_BASE_URL` is the API root, such as `https://example.invalid/v1`.

Save this as `chat.py`, configure those variables for your provider, and run `uv run --extra live --no-dev python chat.py`:

```python
import asyncio
import os

from aether import Agent
from aether.chat import ChatModel, Settings


async def main():
    settings = Settings(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        provider="openai-compatible",
        reasoning=None,  # Set a reasoning level if your provider supports one.
    )
    async with Agent(ChatModel(settings), system_prompt="Keep exact IDs.") as agent:
        agent.register_prompt("Answer concisely.")
        first = await agent.submit("Remember the fictional project ID ORBIT-17.")
        await agent.wait(first.run_id)
        second = await agent.submit("What was the project ID?",
                                    conversation_id=first.conversation_id)
        result = await agent.wait(second.run_id)
        print(result["state"], result["output"])


asyncio.run(main())
```

This path makes real model requests and requires a reachable provider and its credentials. Set `provider="tencent"` to enable TokenHub-specific cache routing. Host instructions are registered before the first successful submission, count toward the input budget, and remain outside compressed history. Default library prompts are English; host text, user messages and tool results are preserved as supplied.

## Scripted runtime example

The collaboration example defaults to **fixed Python decisions, with no LLM inference**. `ScriptedAssistant` requests inventory for `SKU-A`, and a deterministic inventory service returns a quantity of 7. It checks the path from submission through tool execution, another participant's `submit`/`wait`, result delivery and final output. It does not demonstrate natural-language understanding or model quality.

After installing the environment above, run it without network access:

```bash
uv run --offline --no-sync python examples/collaboration.py
```

Expected output: `7`. Installing or synchronizing dependencies for the first time may require network access; the scripted example itself makes no network requests.

To replace the scripted assistant with a real TokenHub model, set `TOKENHUB_API_KEY` in the application's environment, then run:

```bash
uv run --extra live --no-dev python examples/collaboration.py --live
```

This live path calls `deepseek/deepseek-flash`. `TOKENHUB_BASE_URL` optionally overrides the API root; the example does not read another application's configuration file.

## Models and tools

A model implements `async decide(snapshot, tools) -> Decision`; inheritance is not required. `Model` and `Summarizer` protocols are exported from `aether`. A model reads `context` and returns `Call` objects; the runtime executes tools and returns their results. Compaction uses the model's `summarize` method or an explicit callable with the signature `async (previous, fragment, max_bytes) -> Summary`.

Register ordinary functions as tools:

```python
from aether import Agent, Tool
from aether.tools import local_tools


def multiply(a: int, b: int):
    return a * b


custom = Tool("multiply", "Multiply two integers.", {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"], "additionalProperties": False,
}, multiply)
agent = Agent(my_model, tools=[custom])
for tool in local_tools(".", include=("read_file", "search_files")):
    agent.register_tool(tool)
```

The library starts with no tools. `agent.tools` is read-only; add tools through `register_tool`. Synchronous handlers run in a thread pool; asynchronous handlers are awaited. A decision waits for its new tool batch by default. `Decision.wait_for` selects a subset, or `()` makes the batch nonblocking.

`local_tools` provides `terminal`, `read_file`, `write_file`, `edit_file` and `search_files` when `include` is omitted; an empty sequence selects none. Construction does not execute commands, change the working directory or register tools. The path base is not a sandbox. Process tools currently target POSIX, with Bash for commands and `rg` for search.

## Layout and development

Importable code lives in `src/aether`; tests, examples and documentation stay outside the installed package. This is the standard [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/), which helps expose packaging errors instead of accidentally importing code from the repository root. `uv` installs the package before running it.

```bash
bash scripts/check.sh
uv build
```

Checks cover worktree and index credential scans, Ruff, 105 runtime and tool tests, and the scripted collaboration example. They do not require the CLI or the original experiment workspace. Initial dependency installation may need network access; these checks make no model requests. Source distributions include both READMEs, documentation, examples, tests and check scripts.

Context budgets use `ContextPolicy(input_bytes=24000, summary_bytes=1200, overhead_bytes=2048)`: conservative UTF-8 byte counts. Model output limits (`max_tokens`/`max_output_tokens`) and actual `usage` remain token counts.

Aether currently runs in one process with in-memory state. Summaries are lossy, and cancellation cannot undo external effects. Cross-process delivery and restart recovery remain future, demand-driven work.
