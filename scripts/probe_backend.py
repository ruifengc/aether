"""Bounded live backend audit using synthetic data; opt-in, billable model requests.

Run with TOKENHUB_API_KEY in the environment:
uv run --extra live python scripts/probe_backend.py --output /tmp/aether-audit.json
No credentials, request headers, provider error bodies or reasoning text are saved.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx

from aether import Agent, ContextPolicy, Limits, Tool
from aether.chat import ChatModel, Settings
from aether.errors import ModelError

EXPECTED = {"quantity": 7, "receipt": "RCPT-829", "valid": True}
TASK = ("Look up SKU-A, then verify the returned receipt. After verification output only "
        "a JSON object with quantity, receipt, valid. Do not invent tool data.")


async def lookup(sku):
    return {"sku": sku, "quantity": 7, "receipt": "RCPT-829"}


async def verify(receipt):
    return {"receipt": receipt, "valid": receipt == "RCPT-829"}


TOOLS = [Tool("lookup", "Get authoritative quantity and receipt for a SKU.", {
    "type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}, lookup),
    Tool("verify", "Verify a receipt returned by lookup.", {
        "type": "object", "properties": {"receipt": {"type": "string"}},
        "required": ["receipt"]}, verify)]


def matches(text):
    try:
        return json.loads(text) == EXPECTED
    except (ValueError, TypeError):
        return False


class Probe(ChatModel):
    def __init__(self, *args, rows, label, request_limit=33, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows, self.label = rows, label
        self.request_limit, self.requests = request_limit, 0

    async def _request(self, payload):
        if self.requests >= self.request_limit:
            raise ModelError("probe request limit", kind="fatal")
        self.requests += 1
        started = time.monotonic()
        body = await super()._request(payload)
        choice = body["choices"][0]
        message = choice["message"]
        self.rows.append({"case": self.label, "elapsed_s": time.monotonic() - started,
                          "returned_model": body.get("model"), "usage": body.get("usage", {}),
                          "finish_reason": choice.get("finish_reason"),
                          "reasoning_chars": len(message.get("reasoning_content") or ""),
                          "max_tokens": payload.get("max_tokens")})
        return body


async def native(model, *, preserve):
    # Experimental sequential reference, not an alternative runtime implementation.
    messages = [{"role": "system", "content": "Follow the task; concise final JSON only."},
                {"role": "user", "content": TASK}]
    tools = [t.schema() for t in TOOLS]
    handlers = {t.name: t for t in TOOLS}
    names = []
    conversation = "probe-" + uuid4().hex
    for _ in range(5):
        payload = {"model": model.settings.model, "messages": messages, "tools": tools,
                   "max_tokens": model.max_tokens}
        if model.settings.provider == "tencent":
            payload["prompt_cache_key"] = conversation
        body = await model._request(payload)
        choice = body["choices"][0]
        if choice["finish_reason"] not in {"stop", "tool_calls"}:
            return {"state": choice["finish_reason"], "passed": False, "tools": names}
        raw = choice["message"]
        message = {"role": "assistant", "content": raw.get("content")}
        if preserve and "reasoning_content" in raw:
            message["reasoning_content"] = raw["reasoning_content"]
        calls = raw.get("tool_calls") or []
        if not calls:
            return {"state": "completed", "passed": matches(raw.get("content"))
                    and names == ["lookup", "verify"], "tools": names}
        if len(names) + len(calls) > 2:
            return {"state": "unexpected_tools", "passed": False, "tools": names}
        message["tool_calls"] = calls
        messages.append(message)
        for call in calls:
            name = call["function"]["name"]
            if name not in handlers or (not names and name != "lookup"):
                return {"state": "unexpected_tool", "passed": False, "tools": names}
            names.append(name)
            result = await handlers[name].execute(json.loads(call["function"]["arguments"]))
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result)})
    return {"state": "call_limit", "passed": False, "tools": names}


async def audit(args, rows):
    prefix = "TOKENHUB" if args.provider == "tencent" else "DEEPSEEK"
    key = os.environ.get(prefix + "_API_KEY", "")
    if not key:
        raise ValueError("missing " + prefix + "_API_KEY")
    url = os.environ.get(prefix + "_BASE_URL", "https://tokenhub.tencentmaas.com/v1"
                         if args.provider == "tencent" else "https://api.deepseek.com/v1")
    settings = Settings(key, url, args.model, args.provider, args.reasoning,
                        {"auto": None, "enabled": True, "disabled": False}[args.thinking])
    async with httpx.AsyncClient() as client:
        model = Probe(settings, client=client, rows=rows, label="cache",
                      request_limit=3 + args.trials * 15,
                      max_tokens=args.max_tokens, max_output_tokens=max(8192, args.max_tokens))
        history = "\n".join(f"Item {i:04d}: quantity {i % 17}, shelf {i % 23}." for i in range(250))
        state = {"conversation_id": "probe-" + uuid4().hex,
                 "context": {"summary": "", "through": 0, "messages": [
                     {"id": 1, "source": "user", "content": history}]},
                 "pending": [], "completed": []}
        for i in range(3):
            model.label = f"cache-{i}"
            state["context"]["messages"].append({"id": i + 2, "source": "user",
                                                "content": f"Reply exactly OK-{i}; no tools."})
            decision = await model.decide(state, [])
            rows.append({"case": model.label + "-result", "passed": decision.content == f"OK-{i}"})
        for trial in range(args.trials):
            model.label = f"snapshot-{trial}"
            async with Agent(model, TOOLS, limits=Limits(model_calls=5, model_retries=1),
                             context=ContextPolicy(input_bytes=50000),
                             system_prompt="Follow the task; concise final JSON only.") as agent:
                receipt = await agent.submit(TASK)
                result = await agent.wait(receipt.run_id, timeout=300)
                started = [e.data["name"] for e in agent.query(receipt.run_id, after=0)["events"]
                           if e.type == "tool.started"]
                rows.append({"case": model.label + "-result", "state": result["state"],
                             "passed": result["state"] == "succeeded" and matches(result["output"])
                             and started == ["lookup", "verify"], "tools": started,
                             "usage": result["usage"], "usage_reports": result["usage_reports"]})
            for preserve in (True, False):
                model.label = f"native-{'preserved' if preserve else 'omitted'}-{trial}"
                try:
                    result = await native(model, preserve=preserve)
                except ModelError as exc:
                    result = {"state": str(exc), "passed": False}
                rows.append({"case": model.label + "-result", **result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("tencent", "deepseek"), default="tencent")
    parser.add_argument("--model", default="deepseek/deepseek-flash")
    parser.add_argument("--thinking", choices=("auto", "enabled", "disabled"), default="auto")
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--trials", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 16384:
        parser.error("--max-tokens must be between 1 and 16384")
    rows = []
    status = "completed"
    try:
        asyncio.run(audit(args, rows))
    except (ModelError, httpx.HTTPError, TimeoutError) as exc:
        # Adapter errors contain classifications only; raw HTTP exceptions may contain URLs.
        status = str(exc) if isinstance(exc, ModelError) else type(exc).__name__
    except ValueError:
        status = "configuration_or_response_error"
    report = {"status": status, "requested_model": args.model, "provider": args.provider,
              "thinking": args.thinking, "reasoning": args.reasoning, "requests": rows,
              "limits": {"cache_requests": 3, "trials": args.trials,
                         "max_requests": 3 + args.trials * 15,
                         "max_tokens_per_request": max(8192, args.max_tokens)}}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": status, "recorded_rows": len(rows)}, ensure_ascii=False))
    return int(status != "completed" or any(r.get("passed") is False for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
