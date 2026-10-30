"""Scripted inventory generation/review/revision workflow; no real LLM requests.

Every researcher/reviewer decision passes through the runtime's decision accounting.
The staging tool keeps the candidate inside the unfinished run. Replace the
two scripted models with real Model implementations and provide a summarizer
explicitly if they need compaction.
"""
from dataclasses import replace

from aether import Agent, Call, Decision, Limits, Model, Tool


async def stage_candidate(content: str):
    return {"candidate": content}


STAGE = Tool("stage_candidate", "Stage a candidate for a separate review decision.", {
    "type": "object", "properties": {"content": {"type": "string"}},
    "required": ["content"], "additionalProperties": False,
}, stage_candidate)


class ReviewedModel:
    """Example-only composition for one blocking candidate at a time."""

    def __init__(self, researcher: Model, reviewer: Model):
        self.researcher = researcher
        self.reviewer = reviewer

    async def decide(self, snapshot, tools):
        visible = [tool for tool in tools if tool["function"]["name"] != STAGE.name]
        latest = snapshot["inputs"][-1]
        if latest["source"] == "tool" and latest.get("name") == STAGE.name:
            return await self.reviewer.decide(snapshot, visible)
        result = await self.researcher.decide(snapshot, visible)
        if (result.finish_reason in {"stop", "tool_calls"} and result.content and not result.calls
                and not result.wait and not result.wait_for and not snapshot["pending"]):
            return replace(result, content="", calls=(Call(STAGE.name, {
                "content": result.content}),), wait_for=None)
        return result


class Researcher:
    async def decide(self, snapshot, tools):
        inventory = [i for i in snapshot["inputs"] if i.get("name") == "read_inventory"]
        return Decision(str(inventory[-1]["content"]["available"]) if inventory else "0")


class Reviewer:
    async def decide(self, snapshot, tools):
        candidate = snapshot["inputs"][-1]["content"]["candidate"]
        if candidate == "7":
            return Decision(candidate)
        # A review decision may re-read evidence with the ordinary registered tools.
        return Decision(calls=(Call("read_inventory"),))


async def run_demo(*, model_calls=4):
    async def read_inventory():
        return {"sku": "SKU-A", "available": 7}

    model = ReviewedModel(Researcher(), Reviewer())
    async with Agent(model, [STAGE, Tool("read_inventory", "Read SKU-A inventory.", {}, read_inventory)],
                     limits=Limits(model_calls=model_calls, tool_calls=3)) as agent:
        receipt = await agent.submit("Report the available quantity for SKU-A.")
        return await agent.wait(receipt.run_id)


if __name__ == "__main__":
    import asyncio
    import json

    result = asyncio.run(run_demo())
    print(json.dumps({key: result[key] for key in ("state", "output", "model_calls", "tool_calls")}))
