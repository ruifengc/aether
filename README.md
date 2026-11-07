# Aether

**English** · [简体中文](README.zh-CN.md)

Aether is a message-driven kernel for LLM agents. It runs the loop between a model, tools and incoming messages, so an application can submit work, add information while it is running, and observe the result.

The model decides what to do next. Aether manages execution: conversation history, concurrent tool calls, waiting, cancellation, context compaction and bounded recovery. Each run processes state changes in order; model and tool requests execute outside that queue. A new message can invalidate an unfinished model decision before it produces output or starts tools.

[Design](docs/design.md) · [API](docs/api.md) · [Model integration](docs/models.md) · [Working example](examples/follow_up.py)

## Getting started

Requires Python 3.11+. The runtime uses the standard library; the optional OpenAI-compatible connection uses HTTPX.

```sh
git clone https://github.com/ruifengc/aether.git
cd aether
uv sync --extra live
```

Set `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` for your provider. The application reads these settings explicitly:

```python
import asyncio
import os

from aether import Agent
from aether.adapters import OpenAIModel, Settings


async def main():
    model = OpenAIModel(Settings(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
    ))
    async with Agent(model) as agent:
        receipt = await agent.submit("Explain how a heat pump works.")
        await agent.submit("Focus on heating a home in winter.", run_id=receipt.run_id)
        result = await agent.wait(receipt.run_id)
        print(result["state"], result["output"])


asyncio.run(main())
```

`submit` returns once the input is accepted. `wait` waits for that run to finish. Queries and subscriptions let the application observe work without taking over its execution. Aether manages the model's HTTP connections and closes them when the Agent closes.

## Adding tools

A tool is a name, a description, a JSON parameter definition and an ordinary Python function. Register only the capabilities your application needs:

```python
from aether import Tool

async def stock(sku: str):
    return {"sku": sku, "available": 7}

agent.register_tool(Tool("stock", "Read stock for a SKU.", {
    "type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"],
}, stock))
```

The [follow-up example](examples/follow_up.py) uses a real model to select a device, then adds a weight requirement while its catalog tool is running. It demonstrates the same submission path for the initial task and the later requirement. It requires model credentials and makes billable API calls.

## Scope

The kernel runs in one process and keeps state in memory. A conversation can continue across runs; each run has its own execution limits. Context compaction preserves recent messages and summarizes older history, with some loss of detail. Cancellation stops further processing but cannot undo completed external actions.

Models implement the public `Model` interface. Provider request formatting belongs in model adapters; terminal interaction and application configuration belong in applications. The separate [Aether CLI](https://github.com/ruifengc/aether-cli) provides a terminal interface.

## Development

```sh
bash scripts/check.sh
uv build
```

The checks cover runtime behavior, model protocol handling, local tools, code style and credential scanning. They do not call a real model. Public interfaces are in `contracts.py`; implementation is in `runtime.py`, `context.py` and `adapters/`.
