"""Add a requirement while a real LLM is waiting for a tool result.

Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL, then run:
uv run --extra live python examples/follow_up.py
"""
from __future__ import annotations

import asyncio
import os

from aether import Agent, Limits, Tool
from aether.adapters import OpenAIModel, Settings


async def main():
    started, release = asyncio.Event(), asyncio.Event()

    async def catalog():
        started.set()
        await release.wait()
        return [
            {"id": "ALPHA", "mass_g": 180, "runtime_h": 12},
            {"id": "BETA", "mass_g": 120, "runtime_h": 8},
            {"id": "GAMMA", "mass_g": 140, "runtime_h": 10},
        ]

    model = OpenAIModel(Settings(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        provider=os.environ.get("LLM_PROVIDER", "openai-compatible"),
    ))
    tool = Tool("catalog", "Get device IDs, weights in grams, and battery life in hours.",
                {"type": "object", "properties": {}, "additionalProperties": False}, catalog)
    async with Agent(model, [tool], limits=Limits(model_calls=5, tool_calls=3),
                     system_prompt="Use the catalog. Follow the latest requirements.") as agent:
        receipt = await agent.submit("Choose the device with the longest battery life. Reply only with its ID.")
        await asyncio.wait_for(started.wait(), timeout=60)
        await agent.submit("It must weigh no more than 150 grams.", run_id=receipt.run_id)
        release.set()
        result = await agent.wait(receipt.run_id, timeout=180)
        print(result["output"] if result["state"] == "succeeded" else result["error"])


if __name__ == "__main__":
    asyncio.run(main())
