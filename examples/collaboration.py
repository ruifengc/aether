"""Runtime smoke check with scripted decisions; no LLM is called by default.

After installing the environment:
uv run --offline --no-sync python examples/collaboration.py

Use a real model instead (requires TOKENHUB_API_KEY and provider access):
uv run --extra live --no-dev python examples/collaboration.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from aether import Agent, Call, Decision, Tool
from aether.chat import ChatModel, Settings


class Inventory:
    """An ordinary service's decision is deterministic business code."""

    async def decide(self, snapshot, tools):
        request = snapshot["inputs"][-1]["content"]
        return Decision(json.dumps({"sku": request["sku"], "available": 7}))


class ScriptedAssistant:
    async def decide(self, snapshot, tools):
        results = [i for i in snapshot["inputs"] if i["source"] == "tool"]
        if results:
            return Decision(str(results[-1]["content"]["available"]))
        return Decision(calls=(Call("inventory", {"sku": "SKU-A"}),))


async def main(live=False):
    async with Agent(Inventory()) as service:
        async def inventory(sku):
            receipt = await service.submit({"sku": sku})
            result = await service.wait(receipt.run_id)
            if result["state"] != "succeeded":
                raise RuntimeError("inventory request failed")
            return json.loads(result["output"])

        tools = [Tool("inventory", "查询 SKU 库存", {"type": "object", "properties": {
            "sku": {"type": "string"}}, "required": ["sku"]}, inventory)]
        model = ChatModel(Settings(
            api_key=os.environ["TOKENHUB_API_KEY"],
            base_url=os.environ.get("TOKENHUB_BASE_URL", "https://tokenhub.tencentmaas.com/v1"),
            model="deepseek/deepseek-flash", provider="tencent")) if live else ScriptedAssistant()
        async with Agent(model, tools) as assistant:
            receipt = await assistant.submit("查询 SKU-A 库存，严格只回答数量。")
            result = await assistant.wait(receipt.run_id)
            print(result["output"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Use TokenHub DeepSeek instead of scripted decisions; requires TOKENHUB_API_KEY")
    asyncio.run(main(parser.parse_args().live))
