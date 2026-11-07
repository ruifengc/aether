# API reference

**English** · [简体中文](api.zh-CN.md)

[Design](design.md) · [Model integration](models.md)

Public types are exported from `aether`. The HTTP adapter is imported separately from `aether.adapters`.

## Agent

```python
Agent(model, tools=(), *, limits=None, context=None,
      summarizer=None, system_prompt="")
```

`model` implements `async decide(snapshot, tools) -> Decision`. `tools` is a sequence of `Tool` objects. `limits` and `context` accept `Limits` and `ContextPolicy`. Without an explicit summarizer, the Agent uses `model.summarize` if available.

An Agent owns its model instance. Use `async with Agent(...) as agent` or call `await agent.close()`. Closing stops outstanding work, waits for its tasks to settle, then calls the model's optional asynchronous `aclose()`. Closing is idempotent. A closed Agent rejects new submissions and registrations; existing records remain available to query, prune or forget. Use a separate model instance for each Agent.

| Operation | Behavior |
| --- | --- |
| `await submit(content, *, run_id=None, conversation_id=None, message_id=None, source="user")` | Accept a message and return a `Receipt` |
| `query(run_id, *, after=None)` | Read a snapshot; include events after the given cursor when supplied |
| `subscribe(run_id, *, after=None, buffer=128)` | Subscribe to state changes and output |
| `await wait(run_id, *, timeout=None)` | Wait for a terminal state and return its snapshot |
| `await cancel(run_id)` | Request cancellation; use `wait` to observe completion |
| `register_tool(tool)` | Add a named tool for subsequent decisions |
| `register_prompt(prompt)` | Append instructions before the first successful submission |
| `prune(*, keep_runs=1, conversation_id=None)` | Remove older finished run records while retaining conversation history |
| `forget(conversation_id)` | Remove an inactive conversation and its associated records |

Instructions are fixed after the first successful submission. They count toward the input estimate and remain outside history compaction. Registering a tool does not trigger a model request; tool names must be unique. `agent.tools` is a read-only view.

## Messages, conversations and runs

Content must be JSON-serializable. The runtime copies it when accepted. `source` can be `user` or `external`; tool results and runtime messages are created by the kernel. External messages carry facts or notifications, not user instructions. They cannot impersonate a returned tool result.

- Without a run or conversation ID, submission creates a new conversation and run.
- With a conversation ID, input joins its active run or starts a new run using its history. An unknown conversation ID creates a conversation.
- With a run ID, input joins that unfinished run. If both IDs are supplied, they must match.
- Input to a run that is cancelling or already finished is rejected.

`Receipt` contains `run_id`, `conversation_id`, `message_id`, `input_version` and `duplicate`. It acknowledges acceptance into memory, not model processing or task completion. Repeating a message ID with identical content, source and compatible destination returns the original receipt with `duplicate=True`. Conflicting reuse is rejected. Deduplication lasts while the associated run records are retained.

A newly accepted message changes the input version. A model decision based on an older version is discarded before publishing output or starting tools. Already running tools continue, and their actual results return to the run.

## Observing work

`wait` returns when a run becomes `succeeded`, `failed` or `cancelled`. Its timeout or cancellation affects only that waiting caller; it does not cancel the run. To enforce a task deadline, explicitly call `cancel` after a wait timeout.

```python
subscription = agent.subscribe(receipt.run_id, after=0)
try:
    async for event in subscription:
        if event.type == "output":
            print(event.data["content"])
finally:
    subscription.close()
result = await agent.wait(receipt.run_id)
```

With no cursor, a subscription begins with a current `snapshot` event. With a cursor, it replays subsequent retained events before continuing live. `Event` contains `run_id`, `seq`, `type`, `data` and `at`. A slow reader whose buffer fills receives `SubscriptionOverflow`; it can query or resubscribe using its last consumed cursor. Observers do not block execution or trigger model calls.

An output event carries `{"content": ...}`. A run may emit more than one output. `query.output` is the latest output, including output emitted before a later failure. The final state, rather than the existence of output, determines success.

A snapshot includes:

| Fields | Meaning |
| --- | --- |
| `run_id`, `conversation_id` | Run and conversation identity |
| `state`, `phase`, `error`, `output` | Current state, activity, failure reason and latest output |
| `input_version`, `consumed_version` | Accepted and processed input versions |
| `inputs` | Original messages for this run |
| `context` | Summary and retained conversation messages for model input |
| `pending`, `completed`, `waiting_for` | Tool operations and the IDs needed before the next decision |
| `model_pending` | Whether model, compaction or retry work is in progress |
| `model_calls`, `tool_calls`, `summary_calls`, `recoveries` | Execution counts |
| `budget` | Each count's `max`, `used`, `remaining` and `unit` |
| `usage`, `usage_reports` | Reported provider usage and how many requests reported each field |
| `cursor`, `events` | Latest event position; `events` is included only when `after` is supplied |

`state="waiting"` can mean waiting for a model, tools, a retry or new input. `phase` is `deciding`, `compacting`, `retrying` or `None`. `run.waiting` is a specific model wait event, not an event for every use of the waiting state.

Event groups include `input.accepted/consumed`, `model.started/completed/retrying`, `decision.discarded`, `tool.started/late`, `context.compacting/summary_started/summary_reused/compacted`, `output` and `run.*`. A `model.completed` event includes the response finish reason, raw usage and whole-decision `elapsed_s`; that duration is not time to first token.

## Model and tool contracts

`Model.decide` receives a state snapshot and a list of provider-independent `ToolDefinition` dictionaries containing `name`, `description` and `parameters`. It returns:

```python
Decision(content="", calls=(), wait=False, usage={},
         finish_reason="stop", wait_for=None)
Call(name, arguments={})
```

| Decision | Behavior |
| --- | --- |
| Tool calls | Execute the batch, then normally wait for its results |
| `wait=True`, no calls | Publish any text, then wait for new input or results |
| Text, no calls or wait | Publish it; finish if no tools remain, otherwise continue waiting |
| Empty decision | Attempt bounded error recovery |
| Wait with calls or nonempty `wait_for` | Reject as inconsistent |

`wait_for=None` waits for all tools in the new batch. An empty tuple makes the batch nonblocking. A nonempty tuple selects zero-based call indexes or existing pending operation IDs. Errors and timeouts also satisfy a selected wait. Pending operations not selected continue running. New messages are accepted during a selected wait, but the next decision waits for those selected results.

```python
Decision(calls=(Call("fast"), Call("slow")), wait_for=(0,))
```

All references are checked before output or tool execution. A batch that exceeds the remaining tool allowance is rejected in full; when recovery is possible, the model receives feedback to correct it. No calls from a rejected batch execute.

`Tool(name, description, parameters, handler)` holds the public definition and callable. The runtime checks the Python signature, runs synchronous functions in a thread and awaits asynchronous functions. Results must be JSON-serializable and within `tool_result_chars`. The parameter schema is sent to the model; the runtime does not implement general JSON Schema validation. Unknown tools, invalid arguments, exceptions and timeouts become tool error results. Registration enables execution immediately.

## Context and limits

`context` contains `summary`, `through` and `messages`. `through` marks the summarized prefix; some complete user messages at or before that position may still be retained. Models must use the provided messages rather than filtering them again. The adapter sends this context, not a second copy of `inputs`.

| `ContextPolicy` field | Default | Meaning |
| --- | --- | --- |
| `input_bytes` | 24000 | Conservative UTF-8 input estimate |
| `summary_bytes` | 1200 | Summary size estimate, including JSON string encoding |
| `keep_recent` | 6 | Number of recent messages preferred for retention |
| `overhead_bytes` | 2048 | Allowance for protocol and request formatting |

The estimate includes tools, pending and completed operations, instructions and runtime budgets. Leave room outside it for model output. It does not discover provider context windows or count exact tokens. An actual context-limit response reduces the allowance before retrying.

Compaction processes complete messages in order, splitting a record only when necessary. A custom summarizer implements `async (previous, fragment, max_bytes) -> Summary`; fragments are JSON text containing records or labelled record fragments. It must return nonblank text within the size allowance. All pieces succeed before history changes. Recent messages and new arrivals survive; complete older user messages may be retained within the same budget. Successful identical summary pieces can be reused during recovery in that run.

| `Limits` field | Default |
| --- | --- |
| `model_calls` / `tool_calls` / `summary_calls` | 10 / 20 / 12 |
| `model_retries` / `retry_delay` | 3 / 0.1 seconds |
| `model_timeout` / `tool_timeout` | 120 / 30 seconds |
| `tool_result_chars` | 1000000 |

Model counts include decision attempts, including failed or discarded attempts. Summaries have a separate count. Recovery limits apply to the whole run and do not reset after a successful request. A finished run keeps its terminal state; continuing the conversation starts a new run with new allowances.

`ModelError` classifies failures. `transient`, `context`, `invalid_response` and `length` allow bounded recovery; other kinds terminate the run. Authentication failures are not retried. Truncated output is regenerated without executing incomplete calls. Tool errors are not automatically retried.

`usage` recursively sums reported nonnegative integer counters from decisions and summaries. `known` describes completeness of input/output totals. Missing details remain absent. `usage_reports.calls` counts metering reports, and `usage_reports.fields` counts valid reports for each dotted field path. A field reported by fewer requests is a partial total. Equivalent cache counters must not be added together. Calculate a hit rate only when both input and cache counts cover every metering report.

## Retaining records

`prune` removes finished runs in completion order and their message-ID deduplication records. It retains conversation memory and skips records whose background tasks have not stopped. `keep_runs=0` removes all eligible records. Removed runs can no longer be queried or subscribed to; existing copies and queued events remain readable.

`forget` requires the conversation to have no active run or unfinished background tasks. It removes conversation history, runs and deduplication records. Neither operation reverses external effects or guarantees physical erasure.

## Optional local tools

`aether.tools.local_tools(cwd=None, *, include=None)` constructs tools without registering them. `None` selects all; an empty sequence selects none. Names must be known and unique. `cwd` is a path base, not a sandbox; absolute paths and `..` are allowed.

| Tool | Parameters and behavior |
| --- | --- |
| `terminal` | `command`, `workdir="."`, `timeout_seconds=20` (up to 120). Bash process, no interactive stdin; returns stdout, stderr, exit code and timeout status. Retains at most 16000 bytes per stream while draining both. Cleans up the process group |
| `read_file` | `path`, `offset=0`, `limit=20000` (4–100000 bytes). Reads UTF-8; `next_offset` continues without splitting a character |
| `write_file` | `path`, `content`. Atomic UTF-8 replacement, creating parent directories and retaining existing permissions; new files use mode 0600 |
| `edit_file` | `path`, `old_text`, `new_text`. Replaces exactly one match, preserving line endings and permissions |
| `search_files` | `pattern`, `path="."`, optional `glob`. Literal ripgrep search; reports `no_matches` for exit code 1 |

Process tools require POSIX, Bash and ripgrep. Shell state does not persist between calls. Increase the outer tool timeout when allowing long commands, leaving time for cleanup. Cancellation cannot guarantee termination of a process that deliberately leaves its process group.
