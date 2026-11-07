# Model integration

**English** · [简体中文](models.zh-CN.md)

[Design](design.md) · [API](api.md)

## OpenAI-compatible services

Install the `live` extra and import `OpenAIModel` and `Settings` from `aether.adapters`. The adapter uses the Chat Completions request format supported by OpenAI-compatible services, including compatible DeepSeek and TokenHub endpoints.

```python
from aether import Agent
from aether.adapters import OpenAIModel, Settings

model = OpenAIModel(Settings(
    api_key=api_key,
    base_url=base_url,
    model=model_name,
))
async with Agent(model) as agent:
    receipt = await agent.submit("Your task")
    result = await agent.wait(receipt.run_id)
```

The application supplies configuration. The library does not search configuration files or read environment variables. `base_url` is the API root, such as `https://example.invalid/v1`, without `/chat/completions`.

The adapter creates its HTTP client on the first request and reuses the connection pool for decisions and summaries. Agent shutdown stops pending work before releasing the client. Application code does not supply or close an HTTP client. If you call a model directly without an Agent, call `await model.aclose()` when finished. A closed model cannot be reused.

### Settings

| Field | Default | Meaning |
| --- | --- | --- |
| `api_key` | Required | Provider credential |
| `base_url` | Required | API root |
| `model` | Required | Provider model name |
| `provider` | `"openai-compatible"` | `tencent` enables TokenHub routing; other values use ordinary compatible requests |
| `reasoning` | `None` | Value sent as `reasoning_effort`; `None` omits it |
| `thinking` | `None` | `True` enables, `False` disables, `None` leaves the provider default |

When thinking is explicitly disabled, the adapter omits `reasoning_effort`. Omitting effort does not disable thinking. The default request omits both extensions. Set them explicitly with providers that support them. `Settings` excludes credentials and the API root from its representation.

`OpenAIModel` accepts `max_tokens=1536`, `max_output_tokens=8192` and `timeout=110`. The first value is the initial completion allowance; recovery can increase it up to the second. Thinking tokens may consume this allowance. Summary requests use the maximum allowance. The timeout applies to an HTTP request, while `Limits.model_timeout` controls a runtime decision, so allow enough time at both levels.

### Request behavior

The adapter places application instructions in the system message alongside the runtime protocol. It serializes conversation context as a user message, followed by current operations, counts and recovery feedback. Tool definitions are converted to provider function definitions here, outside the runtime.

The provider returns ordinary function calls for tools. The adapter also exposes an `await_input` control function to express a wait. Calling it without arguments waits for new input. Its optional `wait_for` array selects call indexes or existing operation IDs; an empty array makes a new batch nonblocking. This name is reserved by the adapter. See [decision behavior](api.md#model-and-tool-contracts).

The conversation is represented by current runtime state rather than a replay of native assistant/tool messages. Native reasoning history, image inputs and token streaming are not implemented. Tools remain associated with the runtime's actual operations and results.

### Usage and caching

Stable conversation history is placed before frequently changing state to preserve a reusable request prefix. With `provider="tencent"`, the conversation ID sets `prompt_cache_key` and `X-Session-ID`; unsafe header values are hashed. Separate runs in the same conversation retain that ID. Summary requests do not inherit it.

The adapter forwards provider usage into the runtime. Cache counters can be named `prompt_cache_hit_tokens` or `prompt_tokens_details.cached_tokens`; availability depends on the service. An absent field means the provider did not report it. Context compaction, changes to instructions or tool definitions, and provider routing can affect reuse. Caching is a provider behavior and is not guaranteed by this library.

An optional `record_path` writes request/response JSON lines and elapsed time for diagnosis. It does not write authorization headers. The content can include application messages and model responses, so choose storage and retention appropriate to that data.

## Implementing a model

A model implements the `Model` protocol from `aether.contracts`; inheritance is not required:

```python
async def decide(snapshot: dict, tools: list[ToolDefinition]) -> Decision: ...
```

Use `snapshot["context"]` for conversation history, `pending` and `completed` for tool operations, and `budget` for remaining execution counts. Input sources distinguish user requirements, external facts, assistant output, tool results and runtime state. Preserve these distinctions in the provider request. The snapshot and tool definitions are copies; changing them does not update the Agent.

Return tool requests as `Call` objects. Execute registered functions only through the runtime; calling them from a model would bypass operation tracking, limits and cancellation. Usage must include any model requests made by the implementation, including requests that fail after consuming tokens. Raise `ModelError` with an appropriate kind for failures the runtime can handle, without including credentials or private provider error bodies.

A model that owns connections or other asynchronous resources implements `async aclose() -> None`. Agent invokes it after stopping outstanding work. Resource-free models can implement only `decide`.

For compaction, provide `async summarize(previous, fragment, max_bytes) -> Summary`, or pass a separate summarizer to Agent. Summary usage is accounted separately from decision attempts. If neither is available, an oversized context fails explicitly.

A native-message adapter must also account for accepted versus discarded decisions, concurrent tool IDs and results, compressed history and total input size. Private history must not silently bypass the runtime's context limit. These requirements apply equally when an application routes decisions between several models.
