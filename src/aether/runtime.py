"""One mailbox per run; only slow effects run outside the mailbox.

No broker, persistence, approval layer, or model-vendor dependency. Models consume
state snapshots, so they can decide while tools are still running.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .context import Context, ContextPolicy, compact, json_bytes
from .contracts import Call, Decision, Event, Limits, Model, Receipt, Summarizer, Summary, Tool
from .errors import ModelError

TERMINAL = {"succeeded", "failed", "cancelled"}


class SubscriptionOverflow(Exception):
    """The subscriber must query/replay before trusting its view again."""


class Subscription:
    def __init__(self, run: _Run, buffer: int):
        if buffer < 1:
            raise ValueError("subscription buffer must be positive")
        self._run: _Run | None = run
        self._queue: asyncio.Queue = asyncio.Queue(buffer)
        self._closed = False

    def _put(self, event: Event):
        if self._closed:
            return
        try:
            self._queue.put_nowait(deepcopy(event))
        except asyncio.QueueFull:
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queue.put_nowait(SubscriptionOverflow("query state and replay from a cursor"))
            self.close()

    def __aiter__(self):
        return self

    async def __anext__(self) -> Event:
        if self._closed and self._queue.empty():
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, Exception):
            raise item
        if item is None:
            raise StopAsyncIteration
        return item

    def close(self):
        if self._run is not None:
            self._run.subscribers.discard(self)
            self._run = None
        if self._closed:
            return
        self._closed = True
        if self._queue.empty():
            self._queue.put_nowait(None)


@dataclass
class _Conversation:
    context: Context = field(default_factory=Context)
    active: str | None = None


@dataclass
class _Run:
    id: str
    conversation_id: str
    context: Context
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    state: str = "ready"
    inputs: list[dict] = field(default_factory=list)
    accepted: int = 0
    consumed: int = 0
    events: list[Event] = field(default_factory=list)
    subscribers: set[Subscription] = field(default_factory=set)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    finished_at: float = 0
    operations: dict[str, dict] = field(default_factory=dict)
    waiting_for: set[str] = field(default_factory=set)
    completed: list[dict] = field(default_factory=list)
    effects: dict[str, asyncio.Task] = field(default_factory=dict)
    model_id: str | None = None
    model_calls: int = 0
    summary_calls: int = 0
    summary_cache: dict[tuple[str, str, int], str] = field(default_factory=dict)
    recoveries: int = 0
    feedback: str | dict = ""
    context_budget: int = 0
    phase: str | None = None
    tool_calls: int = 0
    discarded: int = 0
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    usage_known: bool = True
    output: str | None = None
    error: str | None = None
    dirty: bool = False
    cancel_requested: bool = False
    pump: asyncio.Task | None = None


class Agent:
    def __init__(self, model: Model, tools: tuple[Tool, ...] | list[Tool] = (), *,
                 limits: Limits | None = None, freshness_guard: bool = True,
                 context: ContextPolicy | None = None, summarizer: Summarizer | None = None, system_prompt: str = ""):
        self.model = model
        self._closed = False
        self._system_prompt = ""
        self._prompts_frozen = False
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be text")
        if system_prompt:
            self.register_prompt(system_prompt)
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register_tool(tool)
        self.limits = limits or Limits()
        self.context_policy = context or ContextPolicy()
        self.summarizer = summarizer or getattr(model, "summarize", None)
        # Exposed only to test the ablation; normal callers should retain the guard.
        self.freshness_guard = freshness_guard
        self._runs: dict[str, _Run] = {}
        self._conversations: dict[str, _Conversation] = {}
        self._receipts: dict[str, tuple[Any, Receipt]] = {}

    @property
    def tools(self) -> Mapping[str, Tool]:
        """Read-only view of registered tools; add tools through register_tool()."""
        return MappingProxyType(self._tools)

    def register_prompt(self, prompt: str) -> None:
        """Append host instructions before the first successful submission."""
        if self._closed:
            raise RuntimeError("agent is closed")
        if self._prompts_frozen:
            raise RuntimeError("register prompts before the first submission")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be text")
        if not prompt.strip():
            raise ValueError("prompt must be nonempty text")
        self._system_prompt += ("\n\n" if self._system_prompt else "") + prompt

    def register_tool(self, tool: Tool) -> None:
        """Append a tool for subsequent decisions; registration does not trigger inference."""
        if self._closed:
            raise RuntimeError("agent is closed")
        if not isinstance(tool, Tool):
            raise TypeError("expected a Tool")
        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ValueError("tool name must be nonempty text")
        if tool.name in self._tools:
            raise ValueError("tool names must be unique")
        if not callable(tool.handler):
            raise TypeError("tool handler must be callable")
        self._tools[tool.name] = tool

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def submit(self, content: Any, *, run_id: str | None = None,
                     conversation_id: str | None = None, message_id: str | None = None,
                     source: str = "user") -> Receipt:
        """Acknowledge local admission, never wait for model/tool completion."""
        if self._closed:
            raise RuntimeError("agent is closed")
        if source not in {"user", "external"}:
            raise ValueError("input source must be user or external")
        if conversation_id is not None and (not isinstance(conversation_id, str)
                                             or not conversation_id):
            raise ValueError("conversation_id must be nonempty text")
        content = json.loads(json.dumps(content, ensure_ascii=False))
        message = {"source": source, "content": content}
        message_id = message_id or uuid4().hex
        if previous := self._receipts.get(message_id):
            old_content, receipt = previous
            if (old_content != message or run_id not in (None, receipt.run_id)
                    or conversation_id not in (None, receipt.conversation_id)):
                raise ValueError("message_id reused with different content/run")
            return Receipt(receipt.run_id, message_id, receipt.input_version, True,
                           receipt.conversation_id)
        if run_id is not None:
            run = self._runs[run_id]
            if conversation_id not in (None, run.conversation_id):
                raise ValueError("run belongs to another conversation")
        elif conversation_id is not None and conversation_id in self._conversations:
            active = self._conversations[conversation_id].active
            if active:
                run_id = active
        if run_id is None:
            conversation_id = conversation_id or uuid4().hex
            conversation = self._conversations.setdefault(conversation_id, _Conversation())
            run_id = uuid4().hex
            # Fork on continuation: the previous run keeps its unchanged context.
            conversation.context = deepcopy(conversation.context)
            run = _Run(run_id, conversation_id, conversation.context,
                       context_budget=self.context_policy.input_bytes)
            conversation.active = run_id
            self._runs[run_id] = run
            run.pump = asyncio.create_task(self._pump(run), name=f"run:{run_id}")
        else:
            run = self._runs[run_id]
        if run.state in TERMINAL or run.cancel_requested:
            raise ValueError("run is closed or cancellation requested; submit a new run")
        version = self._admit(run, source, content, message_id=message_id)
        self._prompts_frozen = True
        receipt = Receipt(run_id, message_id, version, conversation_id=run.conversation_id)
        self._receipts[message_id] = (message, receipt)
        return receipt

    def query(self, run_id: str, *, after: int | None = None) -> dict:
        """Read a snapshot without advancing the run or invoking the model."""
        run = self._runs[run_id]
        if after is not None:
            self._check_cursor(run, after)
        snapshot = self._snapshot(run)
        if after is not None:
            snapshot["events"] = deepcopy(run.events[after:])
        return snapshot

    def subscribe(self, run_id: str, *, after: int | None = None,
                  buffer: int = 128) -> Subscription:
        """Atomic local snapshot/replay plus live registration (no await gap)."""
        run = self._runs[run_id]
        if after is not None:
            self._check_cursor(run, after)
        sub = Subscription(run, buffer)
        run.subscribers.add(sub)
        if after is None:
            sub._put(Event(run.id, len(run.events), "snapshot", self._snapshot(run), time.time()))
        else:
            for event in run.events[after:]:
                sub._put(event)
        if run.state in TERMINAL:
            sub.close()
        return sub

    async def wait(self, run_id: str, *, timeout: float | None = None) -> dict:  # noqa: ASYNC109
        run = self._runs[run_id]
        async with asyncio.timeout(timeout):
            await run.finished.wait()
        # Keep the captured run usable even if another caller pruned its record meanwhile.
        return self._snapshot(run)

    def prune(self, *, keep_runs: int = 1, conversation_id: str | None = None) -> list[str]:
        """Release older terminal run records; preserve conversation context and active work."""
        if type(keep_runs) is not int or keep_runs < 0:
            raise ValueError("keep_runs must be a nonnegative integer")
        if conversation_id is not None:
            self._conversations[conversation_id]  # Reject an unknown scope rather than ignoring it.
        finished = sorted((run for run in self._runs.values() if run.state in TERMINAL and
                           (conversation_id is None or run.conversation_id == conversation_id)),
                          key=lambda run: run.finished_at)
        older = finished[:-keep_runs] if keep_runs else finished
        return self._drop_runs([run.id for run in older if self._can_release(run)])

    def forget(self, conversation_id: str) -> None:
        """Release a finished conversation, its run records and associated deduplication."""
        conversation = self._conversations[conversation_id]
        runs = [run for run in self._runs.values() if run.conversation_id == conversation_id]
        if conversation.active or any(not self._can_release(run) for run in runs):
            raise ValueError("conversation still has active work or unfinished background tasks")
        self._drop_runs([run.id for run in runs])
        del self._conversations[conversation_id]

    @staticmethod
    def _can_release(run: _Run) -> bool:
        return (run.state in TERMINAL and (run.pump is None or run.pump.done())
                and all(task.done() for task in run.effects.values()))

    def _drop_runs(self, run_ids: list[str]) -> list[str]:
        if run_ids:
            for run_id in run_ids:
                del self._runs[run_id]
            removed = set(run_ids)
            self._receipts = {key: value for key, value in self._receipts.items()
                              if value[1].run_id not in removed}
        return run_ids

    async def cancel(self, run_id: str):
        run = self._runs[run_id]
        if run.state not in TERMINAL:
            run.cancel_requested = True
            self._emit(run, "run.cancel_requested")
            run.inbox.put_nowait(("cancel", None))
        # This is admission of cancellation, not confirmation that remote effects stopped.

    async def close(self):
        if self._closed:
            return
        self._closed = True
        tasks = []
        for run in self._runs.values():
            if run.state not in TERMINAL:
                self._finish(run, "cancelled")
            if run.pump:
                run.pump.cancel()
                tasks.append(run.pump)
            for task in run.effects.values():
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _check_cursor(run: _Run, cursor: int):
        if type(cursor) is not int or not 0 <= cursor <= len(run.events):
            raise ValueError("cursor outside available event history")

    def _snapshot(self, run: _Run) -> dict:
        snapshot = deepcopy({
            "run_id": run.id, "state": run.state, "cursor": len(run.events),
            "conversation_id": run.conversation_id,
            "input_version": run.accepted, "consumed_version": run.consumed,
            "inputs": run.inputs, "pending": list(run.operations.values()),
            "waiting_for": sorted(run.waiting_for),
            "completed": run.completed,
            "model_pending": run.model_id is not None,
            "cancel_requested": run.cancel_requested,
            "output": run.output, "error": run.error,
            "model_calls": run.model_calls, "tool_calls": run.tool_calls,
            "summary_calls": run.summary_calls, "recoveries": run.recoveries,
            "phase": run.phase, "feedback": run.feedback,
            "discarded_decisions": run.discarded,
            "usage": {**run.usage, "known": run.usage_known},
        })
        snapshot["context"] = run.context.view()
        snapshot["budget"] = self._budget(run)
        if self._system_prompt:
            snapshot["system_prompt"] = self._system_prompt
        return snapshot

    def _budget(self, run: _Run, *, next_model_call: bool = False) -> dict:
        counters = (
            ("model_calls", self.limits.model_calls, run.model_calls + next_model_call, "attempts"),
            ("tool_calls", self.limits.tool_calls, run.tool_calls, "calls"),
            ("summary_calls", self.limits.summary_calls, run.summary_calls, "attempts"),
            ("recoveries", self.limits.model_retries, run.recoveries, "recoveries"),
        )
        return {name: {"max": maximum, "used": used, "remaining": max(0, maximum - used),
                       "unit": unit} for name, maximum, used, unit in counters}

    def _emit(self, run: _Run, kind: str, **data):
        event = Event(run.id, len(run.events) + 1, kind, deepcopy(data), time.time())
        run.events.append(event)
        for sub in tuple(run.subscribers):
            sub._put(event)

    def _admit(self, run: _Run, source: str, content: Any, **meta) -> int:
        # Runs on the event-loop thread with no await; admission order is explicit.
        run.accepted += 1
        item = {"seq": run.accepted, "source": source, "content": deepcopy(content), **meta}
        self._emit(run, "input.accepted", **item)
        run.inbox.put_nowait(("input", item))
        return run.accepted

    async def _pump(self, run: _Run):
        try:
            while True:
                kind, payload = await run.inbox.get()
                if run.state in TERMINAL:
                    self._emit(run, "message.ignored", reason="terminal", message_type=kind)
                    continue
                if kind == "input":
                    run.summary_cache.clear()
                    run.inputs.append(payload)
                    run.context.append(payload["source"], payload["content"], run_id=run.id,
                                       input_seq=payload["seq"], **{k: v for k, v in payload.items()
                                       if k not in {"source", "content", "seq"}})
                    run.consumed = payload["seq"]
                    operation_id = payload.get("operation_id")
                    if operation_id:
                        run.waiting_for.discard(operation_id)
                        operation = run.operations.pop(operation_id, None)
                        if operation:
                            failed = (isinstance(payload["content"], dict) and
                                      "error" in payload["content"])
                            run.completed.append({**operation, "status": "error" if failed else
                                                  "completed", "context_message_id":
                                                  run.context.sequence})
                    run.dirty = True
                    self._emit(run, "input.consumed", seq=payload["seq"], source=payload["source"])
                elif kind == "model":
                    self._model_result(run, *payload)
                elif kind == "context":
                    operation_id, update = payload
                    if operation_id == run.model_id:
                        run.model_id = None
                        run.phase = None
                        run.context.apply(update)
                        run.summary_cache.clear()
                        run.dirty = True
                        self._emit(run, "context.compacted", through=update["through"],
                                   retained=len(run.context.messages))
                elif kind == "summary_started":
                    run.summary_calls += 1
                    self._emit(run, "context.summary_started", call=run.summary_calls)
                elif kind == "summary_usage":
                    self._usage(run, payload)
                elif kind == "summary_cached":
                    operation_id, version, key, content = payload
                    if operation_id == run.model_id and version == run.accepted:
                        run.summary_cache[key] = content
                elif kind == "summary_reused":
                    self._emit(run, "context.summary_reused")
                elif kind == "retry":
                    if payload == run.model_id:
                        run.model_id = None
                        run.phase = None
                        run.dirty = True
                elif kind == "cancel":
                    self._finish(run, "cancelled")
                self._drive(run)
                if run.state in TERMINAL:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._finish(run, "failed", error=f"runtime:{type(exc).__name__}:{exc}")

    def _drive(self, run: _Run):
        if (run.state in TERMINAL or run.cancel_requested or run.model_id or run.waiting_for
                or not run.dirty
                or run.consumed != run.accepted):
            return
        if run.model_calls >= self.limits.model_calls:
            self._finish(run, "failed", error="model_call_limit")
            return
        run.state = "running"
        run.dirty = False
        operation_id = uuid4().hex
        run.model_id = operation_id
        version = run.accepted
        view = self._snapshot(run)
        # Reserve the prospective decision in its input estimate. Compaction
        # itself does not consume a decision attempt.
        view["budget"] = self._budget(run, next_model_call=True)
        schemas = [t.schema() for t in self._tools.values()]
        if self.context_policy.size(view["context"], schemas, view["pending"],
                                    run.feedback, view["completed"],
                                    self._system_prompt, budget=view["budget"]) > run.context_budget:
            run.phase = "compacting"
            self._emit(run, "context.compacting", through=run.context.through,
                       input_version=version)
            run.state = "waiting"
            self._spawn(run, operation_id, self._compact(run, operation_id, view, schemas))
            return
        run.model_calls += 1
        view["model_calls"] = run.model_calls
        run.phase = "deciding"
        self._emit(run, "model.started", operation_id=operation_id, input_version=version)
        run.state = "waiting"
        self._spawn(run, operation_id, self._decide(run, operation_id, version, view))

    def _spawn(self, run: _Run, operation_id: str, coroutine):
        task = asyncio.create_task(coroutine)
        run.effects[operation_id] = task
        task.add_done_callback(lambda _: run.effects.pop(operation_id, None))

    async def _decide(self, run: _Run, operation_id: str, version: int, view: dict):
        try:
            async with asyncio.timeout(self.limits.model_timeout):
                decision = await self.model.decide(view, [t.schema() for t in self._tools.values()])
            result = decision
        except asyncio.CancelledError:
            return
        except TimeoutError:
            result = ModelError("model TimeoutError", kind="transient", usage={})
        except ModelError as exc:
            result = exc
        except Exception as exc:
            result = f"model:{type(exc).__name__}:{exc}"
        if run.state in TERMINAL:
            self._emit(run, "model.late", operation_id=operation_id)
            return
        run.inbox.put_nowait(("model", (operation_id, version, result)))

    async def _compact(self, run: _Run, operation_id: str, view: dict, schemas: list):
        cached = run.summary_cache.copy()
        remaining = self.limits.summary_calls - run.summary_calls
        calls = 0

        async def summarize(previous, fragment, max_bytes):
            nonlocal calls
            key = (previous, fragment, max_bytes)
            if key in cached:
                run.inbox.put_nowait(("summary_reused", None))
                return Summary(cached[key])
            if calls >= remaining:
                raise ModelError("summary_call_limit", kind="fatal")
            calls += 1
            run.inbox.put_nowait(("summary_started", None))
            async with asyncio.timeout(self.limits.model_timeout):
                result = await self.summarizer(previous, fragment, max_bytes)
            run.inbox.put_nowait(("summary_usage", result.usage))
            # Keep only accepted pieces. A later failure must not spend the
            # remaining request budget redoing the same successful prefix.
            if result.content.strip() and json_bytes(result.content) <= max_bytes:
                run.inbox.put_nowait(("summary_cached", (
                    operation_id, view["input_version"], key, result.content)))
            return result

        try:
            if self.summarizer is None:
                raise ModelError("context limit reached; model needs a summarizer")
            update = await compact(view["context"], self.context_policy, schemas, view["pending"],
                                   summarize, budget=run.context_budget, feedback=run.feedback,
                                   max_calls=self.limits.summary_calls,
                                   completed=view["completed"], system_prompt=self._system_prompt,
                                   runtime_budget=view["budget"])
        except asyncio.CancelledError:
            return
        except TimeoutError:
            error = ModelError("summary TimeoutError", kind="transient", usage={})
        except ModelError as exc:
            error = exc
        except Exception as exc:
            error = ModelError(f"summary:{type(exc).__name__}")
        else:
            if run.state not in TERMINAL:
                run.inbox.put_nowait(("context", (operation_id, update)))
            return
        if run.state not in TERMINAL:
            run.inbox.put_nowait(("model", (operation_id, view["input_version"], error)))

    def _usage(self, run: _Run, usage: dict):
        for key in run.usage:
            value = usage.get(key)
            if type(value) is int:
                run.usage[key] += value
            else:
                run.usage_known = False

    def _recover(self, run: _Run, error: ModelError, *, feedback: dict | None = None):
        if error.usage is not None:
            self._usage(run, error.usage)
        if error.kind not in {"transient", "context", "invalid_response", "length"}:
            self._finish(run, "failed", error=str(error))
            return
        if run.recoveries >= self.limits.model_retries:
            self._finish(run, "failed", error=f"recovery_exhausted:{error}")
            return
        run.recoveries += 1
        if error.kind == "context":
            run.summary_cache.clear()
            minimum = (self.context_policy.overhead_bytes +
                       2 * self.context_policy.summary_bytes + 513)
            budget = max(minimum, int(run.context_budget * .65))
            if budget >= run.context_budget:
                self._finish(run, "failed", error="context allowance cannot be reduced further")
                return
            run.context_budget = budget
        if error.kind in {"invalid_response", "length"}:
            run.feedback = feedback if feedback is not None else (
                "The previous request did not produce a complete executable response. "
                "Correct the format and tool arguments; call only registered tools. "
                "For a direct reply, provide a complete, concise answer.")
        delay = min(30, max(error.retry_after,
                           self.limits.retry_delay * 2 ** (run.recoveries - 1)))
        run.model_id = uuid4().hex
        run.phase = "retrying"
        run.state = "waiting"
        self._emit(run, "model.retrying", reason=str(error), category=error.kind,
                   attempt=run.recoveries, delay=delay)
        self._spawn(run, run.model_id, self._retry(run, run.model_id, delay))

    async def _retry(self, run: _Run, operation_id: str, delay: float):
        await asyncio.sleep(delay)
        run.inbox.put_nowait(("retry", operation_id))

    def _model_result(self, run: _Run, operation_id: str, version: int,
                      result: Decision | str | ModelError):
        if operation_id != run.model_id:
            return
        run.model_id = None
        run.phase = None
        if run.cancel_requested:
            self._finish(run, "cancelled")
            return
        if isinstance(result, str):
            self._finish(run, "failed", error=result)
            return
        if isinstance(result, ModelError):
            self._recover(run, result)
            return
        self._usage(run, result.usage)
        self._emit(run, "model.completed", input_version=version,
                   finish_reason=result.finish_reason, usage=result.usage)
        if self.freshness_guard and version != run.accepted:
            run.discarded += 1
            run.dirty = True
            self._emit(run, "decision.discarded", input_version=version,
                       current_version=run.accepted)
            return
        if result.finish_reason not in {"stop", "tool_calls"}:
            if result.finish_reason == "length":
                self._recover(run, ModelError("model_finish:length", kind="length"))
            else:
                self._finish(run, "failed", error=f"model_finish:{result.finish_reason}")
            return
        if result.wait and (result.calls or result.wait_for):
            self._recover(run, ModelError("decision_cannot_wait_and_call", kind="invalid_response"))
            return
        # Integers reference this decision's calls; strings reference existing pending IDs.
        references = tuple(range(len(result.calls))) if result.wait_for is None else result.wait_for
        if (not isinstance(references, (tuple, list)) or any(
                not ((type(ref) is int and 0 <= ref < len(result.calls)) or
                     (isinstance(ref, str) and ref in run.operations)) for ref in references)):
            self._recover(run, ModelError("invalid_wait_for", kind="invalid_response"))
            return
        if result.calls and run.tool_calls + len(result.calls) > self.limits.tool_calls:
            remaining = self.limits.tool_calls - run.tool_calls
            if (remaining <= 0 or run.model_calls >= self.limits.model_calls
                    or run.recoveries >= self.limits.model_retries):
                self._finish(run, "failed", error="tool_call_limit")
            else:
                self._recover(run, ModelError("tool_call_limit", kind="invalid_response"), feedback={
                    "code": "tool_call_limit", "requested": len(result.calls),
                    "remaining": remaining,
                    "message": "No calls in this batch were executed. Reduce the batch to fit "
                               "the remaining tool budget, or give a complete answer without tools.",
                })
            return
        run.feedback = ""
        if result.content:
            run.output = result.content
            run.context.append("assistant", result.content, run_id=run.id)
            self._emit(run, "output", content=result.content)
        operation_ids = [uuid4().hex for _ in result.calls]
        run.waiting_for.update(operation_ids[ref] if type(ref) is int else ref
                               for ref in references)
        if result.calls:
            for operation_id, call in zip(operation_ids, result.calls, strict=True):
                run.tool_calls += 1
                run.operations[operation_id] = {"id": operation_id, "name": call.name,
                                                "arguments": deepcopy(call.arguments)}
                self._emit(run, "tool.started", **run.operations[operation_id])
                self._spawn(run, operation_id, self._execute(run, operation_id, call))
        if result.calls or result.wait or run.waiting_for:
            run.state = "waiting"
            run.dirty = False
            if result.wait or run.waiting_for:
                self._emit(run, "run.waiting", pending=list(run.operations),
                           waiting_for=sorted(run.waiting_for))
        elif result.content:
            if run.operations:
                run.state = "waiting"
                run.dirty = False
            else:
                self._finish(run, "succeeded")
        else:
            self._recover(run, ModelError("empty_decision", kind="invalid_response"))

    async def _execute(self, run: _Run, operation_id: str, call: Call):
        try:
            async with asyncio.timeout(self.limits.tool_timeout):
                tool = self._tools.get(call.name)
                if tool is None:
                    raise ValueError(f"unknown tool: {call.name}")
                if not isinstance(call.arguments, dict):
                    raise ValueError("tool arguments must be an object")
                result = await tool.execute(call.arguments)
                serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
                if len(serialized) > self.limits.tool_result_chars:
                    raise ValueError("tool result exceeds character limit; narrow the request")
                result = json.loads(serialized)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}:{exc}"}
        if run.state in TERMINAL:
            self._emit(run, "tool.late", operation_id=operation_id, name=call.name)
            return
        self._admit(run, "tool", result, operation_id=operation_id, name=call.name,
                    arguments=deepcopy(call.arguments))

    def _finish(self, run: _Run, state: str, *, error: str | None = None):
        if run.state in TERMINAL:
            return
        run.state = state
        run.finished_at = time.monotonic()
        run.error = error
        if state != "succeeded":
            run.context.append("runtime", {"state": state, "error": error,
                               "unfinished_operations": list(run.operations.values())},
                               run_id=run.id)
        conversation = self._conversations[run.conversation_id]
        if conversation.active == run.id:
            conversation.active = None
        run.model_id = None
        run.phase = None
        run.dirty = False
        self._emit(run, f"run.{state}", output=run.output, error=error)
        for sub in tuple(run.subscribers):
            sub.close()
        # Cancellation is best effort; a remote call or a sync thread may still finish.
        for task in run.effects.values():
            task.cancel()
        run.operations.clear()
        run.summary_cache.clear()
        run.waiting_for.clear()
        run.finished.set()
