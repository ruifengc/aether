# Design

**English** · [简体中文](design.zh-CN.md)

[README](../README.md) · [API](api.md) · [Model integration](models.md)

## A running task can receive new information

An agent task may outlive its first request. A tool may take time, a user may add a requirement, or another service may report a result. Aether treats these arrivals as messages that update the same running task.

An application submits a message and receives a receipt. It can then continue doing other work, submit more information, query current state, or subscribe to events. Waiting for completion is a separate operation.

A conversation holds history across tasks. A run is one period of work that ends in success, failure or cancellation. A conversation has at most one active run; input submitted to that conversation joins its active run or starts the next one.

## State changes are ordered; slow work runs concurrently

Each run has a queue. It consumes incoming messages and returned results in order. The model receives a copy of current state and proposes output, tool calls or a wait. Network calls, tool execution and summarization happen outside the queue, so slow work does not prevent new input from being accepted.

```text
application message ─┐
external update ─────┼─> queue ─> state ─> model request
returned tool result ┘             ^             |
                                   |       check decision
                                   |             |
                                   └── tools / output / wait / finish

application <── query / subscription / completion
```

When a model result returns, the runtime checks which input version it used. A decision made before a newer accepted message is discarded before producing output or starting tools. The next request uses the updated state. This check is always enabled.

Different conversations and tool calls can execute concurrently. A run has only one model, compaction or retry operation in progress at a time.

## The model proposes; the kernel executes

`Model.decide(snapshot, tools)` returns a `Decision`. The model does not execute registered handlers. The runtime validates the decision, checks limits and starts the tools. Tool results return as messages, including errors and timeouts.

By default, a model waits for all tools in its new batch. It can select which results are needed for its next decision; the others continue running. New user input is still accepted while the selected tools finish. A decision that waits for nothing does not cause an empty inference loop: another message or result must arrive first.

Output and completion are separate. A run can publish a useful intermediate answer while tools are still running. The final state tells the application whether the work succeeded; an output event alone does not.

## Long-running work has explicit limits

Each run has limits for decisions, tools, summaries and recovery attempts. Remaining counts are included in model requests. The runtime enforces them before executing a decision; a model cannot extend them by asking again.

When history exceeds the configured input estimate, Aether summarizes an old prefix. It retains recent messages and, when space permits, complete older user messages. All summary pieces must succeed before the old prefix is replaced. New input received during compaction survives. Successful pieces can be reused during recovery within that run, without repeating model requests.

Model failures can be retried within the same limits. Tool failures are returned to the model as results; tools with external effects are not automatically repeated. A truncated model response does not publish partial text or execute incomplete tool arguments.

## Responsibilities

| Part | Responsibility |
| --- | --- |
| `contracts.py` | Public data types, tool definitions and model interfaces |
| `runtime.py` | Message acceptance, ordered state changes, tool execution, limits and events |
| `context.py` | Conversation history, input estimates and compaction |
| `adapters/openai.py` | OpenAI-compatible requests, connection reuse, response parsing and summaries |
| `tools.py` | Optional terminal, file and search handlers |
| Application | Credentials, model selection, task instructions, tool registration and presentation |

The runtime has no dependency on a provider SDK or a terminal application. Tool definitions passed to models contain names, descriptions and parameters, not provider-specific wrappers or executable handlers. An adapter performs the conversion required by its provider.

An Agent owns its model instance. Closing it stops outstanding work and then calls the model's optional `aclose()` method. The built-in adapter creates and reuses its own HTTP client. Applications do not manage that connection. Create a model instance for each Agent; the settings can be reused.

## Current boundaries

State and deduplication records live in one process. There is no restart recovery, distributed delivery or durable job queue. Applications release finished records explicitly through `prune` and `forget`.

Summaries lose detail. Input limits are UTF-8 byte estimates rather than provider token counts. Cancellation cannot undo external actions or forcibly stop a Python function already running in a thread. These boundaries are part of the API's behavior, not guarantees delegated to a prompt.
