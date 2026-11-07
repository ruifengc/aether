# 模型接入

[English](models.md) · **简体中文**

[设计说明](design.zh-CN.md) · [API 参考](api.zh-CN.md)

## OpenAI 兼容服务

安装 `live` 扩展，从 `aether.adapters` 导入 `OpenAIModel` 和 `Settings`。适配器使用 Chat Completions 请求格式，可连接支持该格式的服务，包括兼容的 DeepSeek 和 TokenHub 端点。

```python
from aether import Agent
from aether.adapters import OpenAIModel, Settings

model = OpenAIModel(Settings(
    api_key=api_key,
    base_url=base_url,
    model=model_name,
))
async with Agent(model) as agent:
    receipt = await agent.submit("你的任务")
    result = await agent.wait(receipt.run_id)
```

配置由应用提供，库不查找配置文件，也不读取环境变量。`base_url` 是 API 根地址，例如 `https://example.invalid/v1`，不含 `/chat/completions`。

适配器在首次请求时创建 HTTP 客户端，随后在决定和摘要之间复用连接池。Agent 关闭时，先停止在途工作，再释放客户端。应用代码无需提供或关闭 HTTP 客户端。若直接使用模型而不经过 Agent，应在结束后调用 `await model.aclose()`。已经关闭的模型不能复用。

### 配置

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `api_key` | 必填 | 供应商凭据 |
| `base_url` | 必填 | API 根地址 |
| `model` | 必填 | 供应商的模型名称 |
| `provider` | `"openai-compatible"` | `tencent` 启用 TokenHub 路由；其他值使用普通兼容请求 |
| `reasoning` | `None` | 作为 `reasoning_effort` 发送；`None` 表示省略 |
| `thinking` | `None` | `True` 开启、`False` 关闭、`None` 保留供应商默认值 |

明确关闭思考时，适配器不发送 `reasoning_effort`。省略强度参数不等于关闭思考。默认请求省略这两项扩展，仅在供应商支持时明确配置。`Settings` 的字符串表示不包含凭据和 API 根地址。

`OpenAIModel` 接受 `max_tokens=1536`、`max_output_tokens=8192` 和 `timeout=110`。第一个值是初始 completion 额度，恢复时可提高到第二个值；思考 token 也可能占用该额度。摘要请求使用最大额度。`timeout` 作用于 HTTP 请求，`Limits.model_timeout` 作用于运行中的一次模型决定，两处都应留出足够时间。

### 请求行为

适配器将应用指令与运行协议放入 system 消息，将会话上下文序列化为 user 消息，再附加当前操作、计数和恢复反馈。工具定义在这里转换为供应商的函数格式，运行内核不负责这种转换。

供应商通过普通函数调用返回工具请求。适配器还提供 `await_input` 控制函数表达等待：无参数时等待新输入；可选的 `wait_for` 数组选择本批调用序号或已有操作 ID；空数组使新批次不阻挡后续判断。该名称由适配器保留。详见[决定的行为](api.zh-CN.md#模型与工具接口)。

模型读取当前运行状态，而不是重放供应商原生的 assistant/tool 消息历史。目前没有实现原生推理历史、图片输入或逐 token 输出。工具与结果仍由内核关联到实际操作。

### 用量与缓存

稳定的会话历史放在频繁变化的状态之前，以保留可复用的请求前缀。`provider="tencent"` 时，会话 ID 用于 `prompt_cache_key` 和 `X-Session-ID`；不适合 HTTP 头的值会先计算哈希。同一会话的后续运行沿用该 ID，摘要请求不继承它。

适配器将供应商用量交给内核。缓存计数可能名为 `prompt_cache_hit_tokens` 或 `prompt_tokens_details.cached_tokens`，取决于服务；字段缺失表示供应商未报告。历史压缩、指令或工具定义变化，以及服务端路由都会影响复用。缓存由供应商实现，库不保证每次命中。

可选的 `record_path` 以 JSON 行记录请求、响应和耗时，用于诊断，不写入鉴权头。内容可能包含应用消息和模型响应，存储位置及保留时间应与这些数据相匹配。

## 实现模型接口

模型实现 `aether.contracts` 中的 `Model` 协议，无需继承基类：

```python
async def decide(snapshot: dict, tools: list[ToolDefinition]) -> Decision: ...
```

从 `snapshot["context"]` 读取会话历史，从 `pending`、`completed` 读取工具操作，从 `budget` 读取剩余执行次数。消息来源区分用户要求、外部事实、助手输出、工具结果和运行状态，请求中应保留这些区别。快照与工具定义是副本，修改它们不会更新 Agent。

工具请求以 `Call` 返回。注册函数应由内核执行；模型直接调用会绕过操作记录、次数限制和取消处理。用量应包含实现发出的模型请求，包括已经消耗 token 后失败的请求。对内核可以处理的故障，抛出相应类型的 `ModelError`，内容不包含凭据或供应商私有错误正文。

拥有连接等异步资源的模型实现 `async aclose() -> None`，Agent 在停止在途工作后调用它。不持有资源的模型只需实现 `decide`。

需要压缩时，实现 `async summarize(previous, fragment, max_bytes) -> Summary`，或向 Agent 提供独立摘要器。摘要用量与决定尝试分别计数。两者都没有时，超出限制的上下文会明确失败。

使用原生消息历史的适配器还须处理已接受和被丢弃的决定、并发工具 ID 与结果、已压缩历史以及完整输入大小。不能通过私有历史绕过内核的上下文限制。应用在多个模型之间分配决定时，也应遵守这些要求。
