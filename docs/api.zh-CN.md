# API 参考

[English](api.md) · **简体中文**

[设计说明](design.zh-CN.md) · [模型接入](models.zh-CN.md)

公共类型从 `aether` 导入，HTTP 适配器从 `aether.adapters` 单独导入。

## Agent

```python
Agent(model, tools=(), *, limits=None, context=None,
      summarizer=None, system_prompt="")
```

`model` 实现 `async decide(snapshot, tools) -> Decision`。`tools` 是 `Tool` 序列。`limits` 和 `context` 分别接受 `Limits`、`ContextPolicy`。没有显式摘要器时，Agent 使用模型提供的 `summarize` 方法。

Agent 管理传入的模型实例。使用 `async with Agent(...) as agent`，或调用 `await agent.close()`。关闭时先停止在途工作并等待相关异步任务结束，再调用模型可选的异步 `aclose()`。重复关闭不产生额外操作。关闭后不能提交新消息或注册内容，已有记录仍可查询、清理和遗忘。每个 Agent 使用独立的模型实例。

| 操作 | 行为 |
| --- | --- |
| `await submit(content, *, run_id=None, conversation_id=None, message_id=None, source="user")` | 接收消息，返回 `Receipt` |
| `query(run_id, *, after=None)` | 读取快照；指定游标时附带其后的事件 |
| `subscribe(run_id, *, after=None, buffer=128)` | 订阅状态变化和输出 |
| `await wait(run_id, *, timeout=None)` | 等待运行结束并返回快照 |
| `await cancel(run_id)` | 请求取消；通过 `wait` 确认结束 |
| `register_tool(tool)` | 为后续决定增加具名工具 |
| `register_prompt(prompt)` | 在首次成功提交前追加指令 |
| `prune(*, keep_runs=1, conversation_id=None)` | 删除较旧的已结束运行记录，保留会话历史 |
| `forget(conversation_id)` | 删除不活跃会话及其关联记录 |

指令在首次成功提交后固定，计入输入估计，保留在历史压缩之外。注册工具不会触发模型请求，工具名称必须唯一。`agent.tools` 是只读视图。

## 消息、会话与运行

消息内容必须可被 JSON 序列化，接收时会复制。`source` 可为 `user` 或 `external`，工具结果和运行消息由内核产生。外部消息提供事实或通知，不作为用户指令，也不能伪装成返回的工具结果。

- 未指定运行或会话 ID 时，创建新会话和运行。
- 指定会话 ID 时，消息加入该会话的活跃运行，或继承历史启动下一次运行；未知 ID 创建新会话。
- 指定运行 ID 时，消息加入该未结束运行；同时指定两个 ID 时必须匹配。
- 已在取消或已经结束的运行不再接收输入。

`Receipt` 包含 `run_id`、`conversation_id`、`message_id`、`input_version` 和 `duplicate`。回执确认消息进入内存，不代表模型已处理或任务已完成。同一消息 ID、相同内容与来源、兼容目标的重复提交返回原回执，标记 `duplicate=True`；冲突复用会被拒绝。去重只覆盖仍保留的运行记录。

新消息被接收时更新输入版本。旧版本的模型决定会在发布输出或执行工具前被丢弃。已经运行的工具继续执行，实际结果仍返回本次运行。

## 观察执行过程

`wait` 在运行成为 `succeeded`、`failed` 或 `cancelled` 后返回。等待方超时或取消只影响该等待方，不取消运行。应用若需要任务期限，应在等待超时后显式调用 `cancel`。

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

不指定游标时，订阅先收到当前 `snapshot`。指定游标时，先重放其后保留的事件，再继续接收新事件。`Event` 包含 `run_id`、`seq`、`type`、`data` 和 `at`。缓冲区被填满的慢订阅者会收到 `SubscriptionOverflow`，可以查询状态，或从最后消费的游标重新订阅。观察者不阻塞执行，也不触发模型请求。

输出事件的数据为 `{"content": ...}`。一次运行可以多次输出。`query.output` 是最近一次输出，后续失败不会抹去已经发出的内容；是否成功应看最终状态。

快照包含：

| 字段 | 含义 |
| --- | --- |
| `run_id`、`conversation_id` | 运行和会话标识 |
| `state`、`phase`、`error`、`output` | 当前状态、正在进行的活动、失败原因和最近输出 |
| `input_version`、`consumed_version` | 已接收和已处理的输入版本 |
| `inputs` | 本次运行的原始消息 |
| `context` | 模型使用的摘要和保留的会话消息 |
| `pending`、`completed`、`waiting_for` | 工具操作，以及下次决定必须等待的操作 ID |
| `model_pending` | 是否有模型、压缩或重试操作在途 |
| `model_calls`、`tool_calls`、`summary_calls`、`recoveries` | 各类执行次数 |
| `budget` | 每类计数的 `max`、`used`、`remaining`、`unit` |
| `usage`、`usage_reports` | 供应商已报告的用量，以及各字段被多少次请求报告 |
| `cursor`、`events` | 最新事件位置；仅指定 `after` 时包含 `events` |

`state="waiting"` 可能正在等待模型、工具、重试或新消息。`phase` 为 `deciding`、`compacting`、`retrying` 或 `None`。`run.waiting` 专指模型提出等待的事件，不覆盖所有等待状态。

事件包括 `input.accepted/consumed`、`model.started/completed/retrying`、`decision.discarded`、`tool.started/late`、`context.compacting/summary_started/summary_reused/compacted`、`output` 和 `run.*`。`model.completed` 包含结束原因、原始用量和整次决定的 `elapsed_s`，该耗时不是首 token 延迟。

## 模型与工具接口

`Model.decide` 接收状态快照和 `ToolDefinition` 字典列表。每个定义包含 `name`、`description`、`parameters`，与供应商协议无关。返回值为：

```python
Decision(content="", calls=(), wait=False, usage={},
         finish_reason="stop", wait_for=None)
Call(name, arguments={})
```

| 决定 | 行为 |
| --- | --- |
| 有工具调用 | 执行本批工具，通常等结果全部返回 |
| `wait=True`，无工具调用 | 发布已有文本，再等待新输入或结果 |
| 只有文本 | 发布文本；无在途工具时结束，否则继续等待 |
| 空决定 | 尝试有限次数的错误恢复 |
| 等待同时带工具调用或非空 `wait_for` | 作为冲突决定拒绝 |

`wait_for=None` 等待本批新工具全部返回。空元组表示本批工具不阻挡后续判断。非空元组选择本批调用的零起始序号，或已有在途操作的 ID。错误和超时也算返回；未选中的工具继续运行。等待选定结果期间仍接收新消息，但下次决定需要等这些结果返回。

```python
Decision(calls=(Call("fast"), Call("slow")), wait_for=(0,))
```

所有引用都在输出或执行工具前检查。超过剩余工具额度的批次整体拒绝；可以恢复时，将反馈交给模型纠正。被拒绝的批次没有任何调用会执行。

`Tool(name, description, parameters, handler)` 保存公共定义和函数。内核检查 Python 函数签名，同步函数在线程执行，异步函数直接等待。结果须可被 JSON 序列化，且不超过 `tool_result_chars`。参数 schema 发送给模型，内核没有实现通用 JSON Schema 校验。未知工具、参数错误、异常和超时作为工具错误结果返回。注册后即可执行。

## 上下文和执行限制

`context` 包含 `summary`、`through` 和 `messages`。`through` 标记已经摘要的前缀，但其中某些完整用户消息仍可能保留。模型应直接使用提供的消息，不能再自行按该位置过滤。适配器使用这份上下文，不再重复发送 `inputs`。

| `ContextPolicy` 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `input_bytes` | 24000 | 保守的 UTF-8 输入估计 |
| `summary_bytes` | 1200 | 摘要大小估计，包含 JSON 字符串编码开销 |
| `keep_recent` | 6 | 优先保留的近期消息数 |
| `overhead_bytes` | 2048 | 协议与请求格式的预留空间 |

估计包含工具、在途与已完成操作、指令及运行额度。需要在该额度外为模型输出留空间。它不自动发现供应商上下文窗口，也不精确计数 token。实际收到上下文超限响应时，会缩小额度后再试。

压缩按顺序处理完整消息，只有单条消息过大时才拆分。自定义摘要器实现 `async (previous, fragment, max_bytes) -> Summary`；`fragment` 为包含记录或带标记记录片段的 JSON 文本。返回文本须非空且在大小限制内。全部片段成功后才改变历史。近期消息和新输入会保留；较旧的完整用户消息可在同一预算内保留。同次运行的恢复可以复用成功且相同的摘要片段。

| `Limits` 字段 | 默认值 |
| --- | --- |
| `model_calls` / `tool_calls` / `summary_calls` | 10 / 20 / 12 |
| `model_retries` / `retry_delay` | 3 / 0.1 秒 |
| `model_timeout` / `tool_timeout` | 120 / 30 秒 |
| `tool_result_chars` | 1000000 |

模型计数包含失败或被丢弃的决定尝试，摘要独立计数。恢复次数限制作用于整个运行，不会在一次成功后重置。已结束的运行保持终态；继续会话将启动一次具有新额度的运行。

`ModelError` 区分故障类型。`transient`、`context`、`invalid_response` 和 `length` 可在限制内恢复，其他类型结束运行。鉴权失败不重试。被截断的输出重新生成，不执行残缺调用。工具错误不自动重试。

`usage` 递归累计决定和摘要已报告的非负整数计数。`known` 表示基础输入、输出总量是否完整，缺失明细不补零。`usage_reports.calls` 是计量报告次数，`usage_reports.fields` 按点分字段路径记录有效报告次数。报告次数不足的字段只代表已报告部分。同义缓存字段不能相加；仅在输入与缓存字段覆盖全部计量报告时计算命中率。

## 记录的保留与释放

`prune` 按结束顺序移除旧运行和对应的消息 ID 去重记录，保留会话历史，跳过后台任务尚未停止的记录。`keep_runs=0` 删除全部符合条件的记录。移除后不能再查询或订阅该运行，已有副本与已排队事件仍可读取。

`forget` 要求会话没有活跃运行或未结束的后台任务，随后移除会话历史、运行和去重记录。这两项操作都不能撤销外部影响，也不保证物理擦除。

## 可选本地工具

`aether.tools.local_tools(cwd=None, *, include=None)` 创建工具但不自动注册。`None` 选择全部，空序列不选择任何工具，名称须已知且唯一。`cwd` 是路径基准，不是沙箱；允许绝对路径和 `..`。

| 工具 | 参数和行为 |
| --- | --- |
| `terminal` | `command`、`workdir="."`、`timeout_seconds=20`（最多 120）。运行 Bash，无交互 stdin；返回 stdout、stderr、退出码和超时状态。每路最多保留 16000 字节，同时持续读取两路输出；结束时清理进程组 |
| `read_file` | `path`、`offset=0`、`limit=20000`（4–100000 字节）。读取 UTF-8；用 `next_offset` 续读，不切断字符 |
| `write_file` | `path`、`content`。原子替换 UTF-8 文件，创建父目录并保留原权限；新文件权限 0600 |
| `edit_file` | `path`、`old_text`、`new_text`。替换恰好一次匹配，保留换行和权限 |
| `search_files` | `pattern`、`path="."`、可选 `glob`。ripgrep 字面检索；退出码 1 时返回 `no_matches` |

进程工具要求 POSIX、Bash 和 ripgrep。Shell 状态不跨调用保留。允许长命令时，应用应提高外层工具超时，并留出清理时间。取消无法保证停止主动脱离进程组的进程。
