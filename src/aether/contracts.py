"""Public model, tool and observation contracts; independent of runtime state."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    content: str = ""
    calls: tuple[Call, ...] = ()
    wait: bool = False
    usage: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    wait_for: tuple[int | str, ...] | None = None


class Model(Protocol):
    async def decide(self, snapshot: dict, tools: list[dict]) -> Decision: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}

    async def execute(self, arguments: dict) -> Any:
        inspect.signature(self.handler).bind(**arguments)
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**arguments)
        result = await asyncio.to_thread(self.handler, **arguments)
        return await result if inspect.isawaitable(result) else result


@dataclass(frozen=True)
class Limits:
    model_calls: int = 10
    tool_calls: int = 20
    model_timeout: float = 120
    tool_timeout: float = 30
    tool_result_chars: int = 1000000
    model_retries: int = 3
    retry_delay: float = .1
    summary_calls: int = 12

    def __post_init__(self):
        for value in (self.model_calls, self.tool_calls, self.tool_result_chars, self.summary_calls):
            if type(value) is not int or value < 1:
                raise ValueError("count/size limits must be positive integers")
        for value in (self.model_timeout, self.tool_timeout):
            if value <= 0:
                raise ValueError("timeouts must be positive")
        if type(self.model_retries) is not int or self.model_retries < 0 or self.retry_delay < 0:
            raise ValueError("retry limits must be nonnegative")


@dataclass(frozen=True)
class Receipt:
    run_id: str
    message_id: str
    input_version: int
    duplicate: bool = False
    conversation_id: str | None = None


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int
    type: str
    data: dict
    at: float



@dataclass(frozen=True)
class Summary:
    content: str
    usage: dict = field(default_factory=dict)


class Summarizer(Protocol):
    async def __call__(self, previous: str, fragment: str, max_bytes: int) -> Summary: ...
