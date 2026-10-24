"""Conversation context and bounded prefix compaction, independent of transport."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field

from .errors import ModelError


def json_bytes(value) -> int:
    """Conservative UTF-8 byte estimate; no vendor tokenizer dependency."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class ContextPolicy:
    # An input allowance, with room for model output left outside this allowance.
    input_bytes: int = 24000
    summary_bytes: int = 1200
    keep_recent: int = 6
    overhead_bytes: int = 2048

    def __post_init__(self):
        for value in (self.input_bytes, self.summary_bytes, self.overhead_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("context budgets must be positive integers")
        if type(self.keep_recent) is not int or self.keep_recent < 0:
            raise ValueError("keep_recent must be a nonnegative integer")
        if self.input_bytes <= self.overhead_bytes + 2 * self.summary_bytes + 512:
            raise ValueError("input allowance must leave space for summary and messages")

    def size(self, context: dict, tools: list, pending: list, feedback: str = "",
             completed: list | None = None, system_prompt: str = "") -> int:
        content = {
            "context": context, "tools": tools, "pending": pending, "feedback": feedback,
            "completed": completed or [],
        }
        if system_prompt:
            content["system_prompt"] = system_prompt
        return self.overhead_bytes + json_bytes(content)


@dataclass
class Context:
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    through: int = 0
    sequence: int = 0

    def append(self, source: str, content, **meta):
        self.sequence += 1
        self.messages.append({"id": self.sequence, "source": source,
                              "content": deepcopy(content), **meta})

    def view(self) -> dict:
        return deepcopy({"summary": self.summary, "through": self.through,
                         "messages": self.messages})

    def apply(self, update: dict):
        # Only an immutable prefix was summarized. Messages received meanwhile survive.
        if update["base"] != self.through:
            raise ValueError("context prefix changed during compaction")
        self.summary, self.through = update["summary"], update["through"]
        retained = set(update.get("retained", ()))
        self.messages = [m for m in self.messages if m["id"] > self.through or m["id"] in retained]


def _summary_fragments(messages: list[dict], policy: ContextPolicy, budget: int):
    """Keep ordinary messages whole; label every piece of an oversized record."""
    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def fits(fragment):
        return policy.overhead_bytes + 256 + json_bytes({
            "previous_summary": "x" * policy.summary_bytes,
            "next_fragment": fragment,
        }) <= budget

    def piece(message, record, offset, length):
        return encode({"id": message["id"], "source": message["source"],
                       "offset": offset, "fragment": record[offset:offset + length]})

    batch = []
    for message in messages:
        if fits(encode([*batch, message])):
            batch.append(message)
            continue
        if batch:
            yield encode(batch)
            batch = []
        if fits(encode([message])):
            batch = [message]
            continue
        # Only an individually oversized message is split. Offsets refer to the
        # serialized record's characters, so every Unicode character survives.
        record = encode(message)
        offset = 0
        while offset < len(record):
            low, high = 0, len(record) - offset
            while low < high:
                middle = (low + high + 1) // 2
                if fits(piece(message, record, offset, middle)):
                    low = middle
                else:
                    high = middle - 1
            if not low:
                raise ModelError("summary wrapper exceeds context allowance", kind="fatal")
            yield piece(message, record, offset, low)
            offset += low
    if batch:
        yield encode(batch)


async def compact(view: dict, policy: ContextPolicy, tools: list, pending: list,
                  summarize, *, budget: int, feedback: str = "", max_calls: int = 12,
                  completed: list | None = None, system_prompt: str = "") -> dict:
    """Fold an old prefix in bounded chunks. Never slice content out of existence."""
    messages = view["messages"]
    if not messages:
        raise ModelError("context cannot fit without messages", kind="fatal")
    keep = min(policy.keep_recent, len(messages) - 1)
    while True:
        cut = len(messages) - keep
        # Small user messages carry the original requirements and amendments.
        # Retain them verbatim within a quarter of the input allowance; large
        # documents still undergo summarization. No task-specific fields exist.
        through = max(view["through"], messages[cut-1]["id"])
        candidate = {"summary": "x" * policy.summary_bytes, "through": through, "messages": []}
        available = budget - policy.size(candidate, tools, pending, feedback, completed, system_prompt)
        retained_budget = min(budget // 4, max(0, available))
        retained = []
        for message in messages[:cut]:
            if message["source"] == "user" and json_bytes([*retained, message]) <= retained_budget:
                retained.append(message)
        candidate["messages"] = [*retained, *messages[cut:]]
        if policy.size(candidate, tools, pending, feedback, completed, system_prompt) <= budget:
            break
        if keep == 0:
            raise ModelError("system prompt, tools or pending operations exceed context allowance",
                             kind="fatal")
        keep -= 1
    summary = view["summary"]
    for index, chunk in enumerate(_summary_fragments(messages[:cut], policy, budget)):
        if index >= max_calls:
            raise ModelError("summary_call_limit", kind="fatal")
        result = await summarize(summary, chunk, policy.summary_bytes)
        if not result.content.strip() or json_bytes(result.content) > policy.summary_bytes:
            raise ModelError("summary empty or exceeds allowance", kind="invalid_response")
        summary = result.content
    return {"base": view["through"], "through": through, "summary": summary,
            "retained": [m["id"] for m in retained]}
