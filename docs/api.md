# Aether 接口与行为

本文是运行库接口的唯一完整说明；[快速上手](../README.md)、[架构边界](architecture.md) 和 [可运行示例](../examples/collaboration.py) 均位于本项目内。Python 3.11+，默认标准库依赖；ChatModel 的 HTTP 依赖通过 live 扩展安装。

## 对象与关联

| 名称 | 表示什么 | 不表示什么 |
| --- | --- | --- |
| Agent | 持有模型、工具与多份上下文的运行库实例 | 一个实例不只处理一个用户或一个运行 |
| conversation_id | 记忆连续性的归属，同一归属最多一个活跃运行 | 不要求用户发言一次就新建 |
| run_id | 一段可接收多条输入、产生多次输出并最终结束的运行过程；文档中的“工作”也指它 | 不等于一条用户消息，也不是一次模型请求 |
| message_id | 公开提交的输入去重标识，范围是当前 Agent 内仍保留的记录 | 不为每条输入预留独立回答 |
| operation_id | 内部关联一次操作的发起与完成；pending 工具项使用 id 字段保存它 | 不是提交接口接受的运行 ID |
| Event.seq / cursor | 同一 run 中观察事件的位置，用于重放 | 不等于输入版本或上下文消息 ID |

当前没有每条消息的回复 future、公开内部调度接口或自动参与方路由。调用方需要请求结果时组合 submit 与 wait；关心输出与通知时消费 subscribe。register_tool 注册执行能力，subscribe 注册观察者，二者用途不同。

## 使用

```python
import os

from aether import Agent, Tool
from aether.chat import ChatModel, Settings

async def lookup(sku: str):
    return {"sku": sku, "available": 7}

tool = Tool("lookup", "查询库存", {
    "type": "object", "properties": {"sku": {"type": "string"}},
    "required": ["sku"], "additionalProperties": False,
}, lookup)

settings = Settings(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"],
                    model=os.environ["LLM_MODEL"], provider="openai-compatible")
async with Agent(ChatModel(settings), [tool]) as agent:
    first = await agent.submit("查询 SKU-A 的库存")
    await agent.submit("只需要数量", conversation_id=first.conversation_id)
    result = await agent.wait(first.run_id, timeout=120)
    followup = await agent.submit("刚才查询的是哪个 SKU？", conversation_id=first.conversation_id)
    result = await agent.wait(followup.run_id, timeout=120)
```

## 接收与会话

| 接口 | 行为 |
| --- | --- |
| await agent.submit(content) | 新建会话与工作，接收后返回 Receipt |
| await agent.submit(content, conversation_id=...) | 该会话有活跃工作则补充，否则继承历史新建工作；新 ID 自动创建会话 |
| await agent.submit(content, run_id=...) | 只向指定非终态工作补充输入 |
| await agent.submit(content, source="external", ...) | 外部服务通知通过同一入口提交，归属规则相同 |
| agent.query(run_id, after=cursor) | 读取快照，可同时读取游标后的事件 |
| agent.subscribe(run_id, after=cursor) | 重放后持续接收事件 |
| await agent.wait(run_id, timeout=...) | 等待终态；调用方等待超时不取消工作 |
| await agent.cancel(run_id) | 提交取消；会话仍可开始后续工作 |
| agent.prune(keep_runs=1, conversation_id=None) | 释放旧的已结束工作记录，保留会话上下文，返回释放的 run_id 列表 |
| agent.forget(conversation_id) | 释放已结束会话的上下文、工作记录和关联去重项 |
| await agent.close() | 释放资源，通常使用 async with |

content 必须可由标准 JSON 序列化，接收时复制为独立数据。Receipt 含 run_id、conversation_id、message_id、input_version、duplicate；它只确认输入已进入内存队列，不代表已合并进上下文、模型已看过或任务已完成。同一 ID、来源、内容和兼容归属重复提交返回原输入对应的凭据并标记 duplicate=True；改变来源、内容，或指定不同归属复用 ID 会报错。

source 默认 user，另支持 external；tool 与 runtime 来源由内核产生。ChatModel 提示将 external 内容作为外部事实，区别于用户指令；自定义模型应自行遵守来源语义。公开 submit 不能伪装成 tool 结果，也不能靠在 external 内容中填 operation_id 完成某个 pending 工具；工具结果由已注册 handler 的返回关联。未指定归属的 external 和 user 一样会创建新上下文。显式 run_id 与 conversation_id 必须匹配。正在取消的运行在终态提交前暂不接收新输入。

conversation_id 表示上下文连续性，run_id 表示一段推进过程，均不要求用户输入对应一个轮次。旧运行的终态不改变，query 返回其结束时的上下文快照；后续运行拥有新的调用与恢复额度。未指定归属的 submit 始终创建独立上下文。

wait 直接等待终态提交的完成信号，不建立事件订阅，也不在第一条输出到达时返回。多个等待方互不影响，超时或取消某个等待方不取消运行；即使另一个等待方已清理该运行记录，已经开始的等待仍返回其结果。

输出通过订阅中的 `event.type == "output"` 接收，data 只含 `{"content": ...}`，run_id、事件序号由 Event 自身关联。回答和提问都走这个位置，可先于终态多次出现。query.output 是最近一次输出；即使随后失败或取消，已发出的输出仍保留，调用方应以 state 判断是否成功。

例如，在持有 agent 和 receipt 的异步函数内读取输出：

```python
subscription = agent.subscribe(receipt.run_id, after=0)
try:
    async for event in subscription:
        if event.type == "output":
            print(event.data["content"])
finally:
    subscription.close()
result = await agent.wait(receipt.run_id)  # 订阅结束后读取终态；也可由另一协程独立等待
```

after=0 从头重放，可取得订阅前已经发出的回答；after=None 默认先给当前 snapshot，再跟随后续事件，不重放过去每条 output。上例适合短运行，长记录的缓冲与溢出处理见下文。普通查询不会消耗事件，输出也不会自动作为新输入再次触发模型。

## 一条输入怎样参与处理

1. submit 分配输入版本、发出 input.accepted 事件并入队，立即返回凭据。工具返回也经过同一个内部接收步骤。
2. 运行队列逐条消费输入，追加到原始 inputs 与上下文，再发出 input.consumed。此时 pending 中对应工具才移入 completed。
3. 已接收的输入全部消费，没有当前模型、摘要或重试控制操作，且 waiting_for 为空时，捕获最新快照。需要压缩则先压缩，否则发起模型请求；未被选为关键项的在途工具不阻止此步。
4. 模型返回后校验输入版本；有新输入则丢弃旧决定，不发布其正文、不启动其工具，然后准备下一次判断。旧请求通常会完成才检查，不会因补充输入立即被取消。
5. 当前决定合法时，按下表输出、发起工具、暂停或结束。

上述版本保护按默认 freshness_guard=True 描述；False 仅供消融对照，会失去过时决定保护。合并是“把已排队消息都纳入一次状态快照”，没有固定批量大小或定时凑批窗口。先到先处理，快照包含累计摘要与保留原文，不是仅发送本批消息，也不保证压缩后仍逐字发送所有历史。推理中陆续到达的输入可能导致旧决定作废和额外请求。

同一队列还接收模型决定、压缩完成、重试到期和取消等内部控制消息；它们不都算作 inputs，也不都增加输入版本。模型正文进入上下文并产生 output，但不自动变成下一条待处理输入。

| 字段或事件 | 准确含义 |
| --- | --- |
| Receipt.input_version / query.input_version | 同一运行已接收输入的最大序号；用户、外部、工具输入都计入 |
| query.consumed_version | 已消费并合并进运行状态的最大输入序号 |
| input.accepted.data.seq / input.consumed.data.seq | 该条输入的序号；区别于外层 Event.seq |
| model.started.data.input_version | 本次决策捕获的输入版本；不表示所有输入已被语义正确理解 |
| context.messages 中的 id | 上下文记录序号，包含助手输出等记录，并在后续运行延续 |

## 决定、等待与终态

以下以合法且未过时、未声明 wait_for 的 Decision 为前提；截断、预算耗尽与取消优先按错误或停止规则处理。

| calls | wait | content | 行为 |
| --- | --- | --- | --- |
| 有 | False | 可有可无 | 有正文先发 output，再提交全部工具；默认等本批全部返回后统一判断 |
| 无 | True | 可有可无 | 有正文先发 output，然后暂停；无正文则静默等待，即使没有 pending 也不结束 |
| 无 | False | 有 | 发 output；仍有 pending 则保留运行，否则 succeeded |
| 无 | False | 无 | 空决定，进入有界格式恢复；有 pending 也不会自动视为等待 |
| 有 | True | 任意 | 决定冲突，进入有界格式恢复，不执行其中工具或正文 |

wait=True 是模型表达的暂停意图；agent.wait() 是调用方等待终态的方法，二者不同。暂停可被用户、external 或工具输入唤醒；查询、订阅、工具注册都不会唤醒。一次合法等待决定清除当前待判断标记，没有新消息时不重复推理。

| state | 应如何理解 |
| --- | --- |
| ready | 新建运行，尚未开始推进 |
| running | 内部正在准备一次操作，通常很短暂 |
| waiting | 非终态；可能在等模型、压缩、重试、工具或外部输入 |
| succeeded / failed / cancelled | 终态；原运行不再接收新的输入，可使用 conversation_id 开始后续运行 |

waiting 时结合其他字段判断：phase=deciding／compacting／retrying 表示当前控制操作；model_pending 为 True 表示这条控制路径有在途操作，包含摘要和重试等待，不只指 HTTP 模型请求。pending 列出全部在途工具，waiting_for 列出其中必须等的操作 ID。phase=None、model_pending=False 且 pending 为空时，才通常是在等外部输入。run.waiting 事件由普通等待或关键等待决定发出，不覆盖所有 state=waiting 的时刻。

同一运行最多一条模型／摘要／重试控制路径，多个工具可以并发；串行的是消息消费和决定提交，不是让全部工具挤在同一个执行线程。默认只等待本次新调用批，不自动等待此前遗留的非阻塞工具；模型可按下节显式覆盖。tool.started 表示操作已提交，不承诺 handler 已开始运行。同步函数通过线程池，协程工具通过事件循环执行；注册时没有业务同步等待选项。

cancel() 对非终态运行提交取消请求，已结束运行则无操作；要确认最终状态，可继续 wait()。即使观察到终态，也不保证远程动作或同步线程已停止，不能据此撤销已发生的副作用。

## 本次决定的关键工具等待

Decision.wait_for 默认 None，表示等待本次 calls 中的全部调用；显式空元组表示本批全部不等，非空元组指定等待子集。元素可以是本次 calls 的零起始序号，或已有 pending 项的 id。它表达下一步推理必须等的集合，不改变工具的并发执行方式，也不修改业务 handler 参数。

| wait_for | 行为 |
| --- | --- |
| None（省略） | 等待本次所有新调用；没有新调用则不增加等待项 |
| () | 本批全部不阻挡后续判断；新消息或工具结果到达时可继续 |
| (0,) / (0, 1) | 只等待本次指定序号；未选中的调用继续执行 |
| (existing_operation_id,) | 等待已有 pending 项，也可与本次序号混合 |

```python
from aether import Call, Decision

# 默认等齐本次两个调用，无需额外声明。
Decision(calls=(Call("fast"), Call("slow")))
# 只等 fast；它返回后，即使 slow 未完成也能继续判断。
Decision(calls=(Call("fast"), Call("slow")), wait_for=(0,))
# 两个调用都不阻挡新消息触发判断。
Decision(calls=(Call("fast"), Call("slow")), wait_for=())
# 下一次决定也可引用已经在途的 slow 操作，而不是重复发起它。
Decision("快结果已得到", wait_for=(slow_operation_id,))
```

内核先检查引用是否合法，再发布正文或执行工具；越界、负数、布尔值和未知操作 ID 均进入有界格式恢复，不产生该决定的输出或工具副作用。重复引用按集合合并。显式 wait_for 可以和 calls、content 同时出现；wait=True 仍不能与 calls 或非空 wait_for 混用。全部不等待也不立即空转调用模型，须有新的输入触发。

运行快照 waiting_for 始终是转换后的操作 ID 列表，便于与 pending 对照。它非空时，用户消息、external 通知和任意工具结果照常接收并消费，但不启动模型或上下文压缩。所选工具结果被消费时解除对应等待；成功、错误、超时都算返回，集合清空后合并累计信息再判断。未选中的 pending 不构成阻挡。取消与关闭不等待该集合；进入终态时清空它。

ChatModel 的普通原生工具调用自动采用默认等待，不必额外调用 await_input。只有覆盖默认时才在同一响应附加 `await_input({"wait_for":[0]})` 或 `await_input({"wait_for":[]})`。序号只计算业务工具，不计算 await_input 的位置；字符串引用已有 pending id。也可在没有新工具时声明等待已有操作。无参数 await_input() 仍表示普通 wait=True，暂停到下一条输入，不能和业务调用同时出现；显式空列表表示全部不等，二者不同。JSON null 不接受；默认通过省略控制调用表达。同一响应只能有一个 await_input。

普通补充的推理响应可能延后；文本中的“改目标”仍需等关键集合解除后才由模型理解，需要立即停止时使用 cancel。模型只在下一步不依赖全部结果时覆盖默认，不按工具耗时自动猜测关键性。工具返回句柄与后台业务实际完成是两件事；业务工具可自行维护后台任务，再提交 external 通知，内核不代管业务任务。

## 记录与会话生命周期

默认保留内存记录，释放操作由应用明确调用，不触发模型或工具。prune 按结束顺序保留最近 keep_runs 个终态工作，活跃工作不计入该数；conversation_id 限定范围，省略则面向整个 Agent。keep_runs=0 可释放全部可释放终态记录。后台协程未结束的旧记录跳过，因此保留数不是硬内存上限。keep_runs 必须是非负整数，未知会话范围报 KeyError。

prune 删除该工作的原始输入、事件、查询记录及所有关联 message_id 去重项，会话的摘要和近期消息继续保留。被删除工作不能再 query、subscribe 或新建 wait，均报 KeyError；已经返回的快照和已排队的订阅事件仍可读取。去重只覆盖尚保留的工作，已释放的 message_id 可以作为新消息重新提交。

forget 要求会话没有活跃工作，也没有未结束的后台协程，否则报 ValueError；未知会话报 KeyError。成功后可复用 conversation_id 开始全新会话。它与 prune 均可在 close 之后调用；不会撤销外部动作，也不能保证取消后的同步线程或脱离进程组的程序已停止。

```python
result = await agent.wait(receipt.run_id)
agent.prune(keep_runs=1, conversation_id=receipt.conversation_id)  # 继续交谈，释放旧工作记录
# 会话确认结束、相关后台协程也结束后：
agent.forget(receipt.conversation_id)
```

## 模型与工具契约

模型实现 async decide(snapshot, tools) -> Decision。Decision 的 content、calls、wait 组合按上文行为表处理，usage 和 finish_reason 用于统计与完整性检查。工具调用通过后台操作执行，结果经相同邮箱回流。工具注册不增加同步等待开关。

### 宿主提示注册

Agent(..., system_prompt="...") 配置宿主业务提示；agent.register_prompt(text) 按调用顺序追加片段，中间以两个换行分隔。默认无业务提示，不要求注册任何常用工具：

```python
async with Agent(model, system_prompt="你是新闻证据服务，依据工具资料回答。") as agent:
    agent.register_prompt("明确区分报道事实与推断，保留原文来源编号。")
    receipt = await agent.submit("查询最新进展。")
    result = await agent.wait(receipt.run_id)
```

注册为普通同步操作，不触发推理、不写入历史、不增加消息版本。构造参数允许默认空字符串；注册片段必须为非空文本，非文本报 TypeError，空白片段报 ValueError。首次成功 submit 后或 close 后注册报 RuntimeError；无效 submit 不冻结配置，prune/forget 不重新开放注册。工具注册仍按原规则可在运行中追加。

提示属于 Agent 实例，适用于其所有会话；不同业务使用不同 Agent，可以共享同一个模型适配器。没有注册时快照不含 system_prompt；有注册时 snapshot.system_prompt 为完整文本，自定义模型适配器负责把它放到供应商支持的指令位置。用户消息中同名字段仍只是普通输入，不改变注册配置。

默认 ChatModel 每次先发送简短的运行协议，再拼接宿主指令，两者位于同一 system 消息；业务提示不重复放进 user 快照。协议解释来源、摘要、在途操作和 await_input 控制，不预设业务角色、回答长短或新闻规则。工具描述与 schema 单独提供；摘要调用使用独立的压缩说明，不把宿主指令当摘要任务。库内默认协议、摘要说明、恢复反馈、控制工具及常用工具描述均为英文；宿主通过 system_prompt/register_prompt 提供的文本保留原样，不自动翻译。

宿主提示按现有 UTF-8 字节估计计入输入额度，不参与会话压缩；提示本身过大时明确报告预算错误，不自动摘要或截断。默认批等待、预算限制和版本检查仍由代码执行，不靠提示词保证。

snapshot 包含 inputs、pending、waiting_for、completed、output、context、feedback 和 recoveries，以及状态、版本、用量等观察字段；注册提示时还包含 system_prompt。context 包含 summary、through、messages，消息保留 source、run_id、独立递增 id，以及适用的工具关联字段。completed 独立保留当前运行的操作 ID、名称、参数、完成或错误状态及 context_message_id，不因压缩而删除；它不自动禁止显式刷新工具。query 还保留原始 inputs，模型适配器应使用 context 避免重复发送两份历史。

through 是已处理前缀的覆盖游标，不意味着所有 id<=through 的原文已删除：messages 可能保留这个范围内的短用户消息，按原始 ID 排序，与近期消息一起发给模型。适配器应直接使用 messages，不能再按 through 过滤；较早原文不能覆盖较新的有效更正。

ChatModel 使用 OpenAI 兼容协议，历史通过角色化状态快照表达，不回放原生 assistant tool_calls 或隐藏 reasoning 历史。它支持原生 function calling；await_input 是运行时等待控制，具体参数见上节，正文始终是唯一输出位置。正文可以和等待决定同时出现。完整 transcript 回放需另行遵守具体供应商协议。

请求将 context（旧式适配为 inputs）放在易变的 pending、completed 和 feedback 之前，尽量保留已有历史的可缓存前缀。TokenHub（provider=tencent）的决策请求以 conversation_id 设置 prompt_cache_key 和 X-Session-ID，跨 run 保持、换会话切换；中文、控制字符或过长的 ID 先转为稳定 SHA-256 值，避免非法 HTTP Header。没有会话 ID 时不添加，其他供应商不添加这两个路由字段。摘要接口没有会话 ID，不继承上次决策的标识。供应商返回的缓存用量保留在 model.completed 的原始 usage 中，query 的 usage 当前只汇总输入与输出 tokens，不汇总缓存明细。

缓存命中率按总命中输入 tokens / 总输入 tokens 计算，不是各请求比例的简单平均。首次请求、工具集合变化和摘要改写前缀时可能下降；服务端缓存不保证每次命中。

Tool 接收名称、说明、JSON schema 和普通函数。同步函数在线程执行，异步函数使用协程；签名错误、未知工具、超时和不可序列化结果作为带关联的工具错误交给模型。注册工具直接可用，无安全审批，也不自动重试外部写入。

除构造参数 `Agent(model, tools=[...])` 外，也可使用 `agent.register_tool(tool)` 追加注册，后续决策即可看到工具；注册本身不触发模型调用，不影响已经发出的请求。名称必须非空且唯一；tools 为只读映射，增添工具须使用 register_tool；仅 ChatModel 保留 await_input 名称，冲突在请求前以 fatal ModelError 报告，自定义模型不受这个名称限制；关闭后不允许注册。参数 schema 提供给模型，目前执行前检查 Python 函数签名，未引入通用 JSON Schema 校验器；本地工具自行检查必要的参数类型与范围。

## 压缩与预算

Agent(..., context=ContextPolicy(...), summarizer=...) 可配置压缩。默认优先使用模型的 summarize 方法；ChatModel 已提供。自定义摘要器为 async summarize(previous: str, fragment: str, max_bytes: int) -> Summary，返回文本与可选 usage。只有 decide 的简单模型仍可在预算内工作，超过预算却没有摘要器时会明确失败。

fragment 是 JSON 文本：通常为完整消息数组；单条记录过大时，使用 id、source、offset、fragment 对象，fragment 字段包含该记录的序列化文本片段，offset 按 Unicode 字符计数。片段按时间及偏移顺序处理，来源和消息 ID 不丢失；不能假设不同调用的 fragment 字符串直接拼接就是一个 JSON 数组。每次请求为 previous 预留摘要额度，并计入字符串转义开销。所有分片成功后才提交一次前缀替换，后续分片失败不提交半份摘要。

内置摘要按 Goal / Constraints / Done / Open / Facts 组织，空栏目省略；保留最新更正的适用范围、最终输出要求、完成的操作及精确结构化事实。使用紧凑英文和原始键值节省空间，仍按实际序列化字节数验收，不保证模型语义无损。

压缩同时在当前输入额度的四分之一内，按时间顺序尝试保留旧前缀的完整 user 消息；容纳不下的记录跳过原文保留，仍参与摘要。此保留额度计入同一输入预算，并非额外扩容；不保护工具或 external 的全部原文，也不为所有历史指令提供永久保留保证。近期消息按 keep_recent 优先保留，两部分合并检查预算；摘要失败时两部分都不提前提交。

| ContextPolicy 字段 | 默认值 | 含义 |
| --- | --- | --- |
| input_bytes | 24000 | 输入额度，使用 UTF-8 字节保守估计，不是精确供应商 tokenizer |
| summary_bytes | 1200 | 摘要的字节估计上限；内置摘要器据此约束文字长度 |
| keep_recent | 6 | 优先原样保留的近期消息数；单条超长输入可全部参与压缩 |
| overhead_bytes | 2048 | 系统提示、适配器包装等开销预留；工具与 pending 另计 |

输入额度之外须为模型输出预留空间。默认值较保守，不自动发现模型硬上限。实际收到上下文超限错误时，会进一步缩小额度并重新准备上下文；固定工具定义或 pending 过大、无法容纳时明确停止，不静默截断。

预算字段均以 bytes 命名，表示 UTF-8 字节保守估计；模型输出限制 max_tokens/max_output_tokens 和 usage 仍以真实 tokens 计数。模型窗口大小与应用何时压缩分别配置，库不自动推断供应商窗口。

ChatModel(..., max_tokens=1536, max_output_tokens=8192) 指定初始和恢复时最大输出额度。截断后重答而非拼接残缺工具 JSON；已有成功工具结果会保留。摘要请求使用最大输出额度以容纳思考和摘要，不把隐藏思考作为历史。

## 错误恢复与观察

ModelError(kind=...) 供适配器表达安全错误：transient、context、invalid_response、length 会内部恢复；其他类别结束当前工作。401 等鉴权错误不重试；429、408、常见 5xx、网络故障和超时进入退避。错误内容不得包含密钥、入口地址或网关正文。

Limits 默认：model_calls=10、tool_calls=20、summary_calls=12、model_retries=3、retry_delay=0.1、model_timeout=120、tool_timeout=30、tool_result_chars=1000000。model_calls 计决策请求尝试，summary_calls 单独计；恢复次数按整次工作累计，不因短暂成功而无限重置。工具结果原始接收上限与模型上下文额度独立，大结果先进入上下文再按需摘要。

快照的 phase 表示 deciding、compacting、retrying 或空闲，recoveries 和 summary_calls 分别计恢复和摘要请求。usage 汇总决策与摘要的已知用量，缺失时 known=False。事件包括 input.accepted、input.consumed、output、model.started/completed/retrying、decision.discarded、context.compacting/summary_started/compacted、tool.started 和 run.*。run.waiting 表示暂停，run.succeeded 等终态事件只标记生命周期，应用可据此避免重复展示最新输出。

input.consumed 只表示已合并进运行状态；model.started 的 input_version 标识该次请求捕获的输入。接收到新信息但未参与原请求时，过时决定会被丢弃。查询和订阅不触发额外模型调用。

订阅缓冲默认 128；慢订阅者收到 SubscriptionOverflow 后应重新查询或从有效游标重放。query 后 subscribe(after=cursor) 在同一事件循环中可衔接历史与后续事件。

## 常用工具

`aether.tools.local_tools(cwd=None, *, include=None)` 返回普通 Tool 列表，不执行命令、不自动注册，也不改变宿主工作目录。Agent 默认没有工具；CLI 和其他宿主都通过同一函数显式选择。

```python
from aether import Agent
from aether.tools import local_tools

tools = local_tools("/path/to/project", include=("read_file", "search_files"))
agent = Agent(my_model, tools=tools)
# 也可从返回列表中逐个选择，通过 agent.register_tool 注册。
```

include 为工具名序列，按指定顺序返回；None 返回下表全部工具，空序列返回空列表，未知或重复名称报 ValueError。cwd 指定固定路径基准，默认构造时的当前目录；绝对路径和 `..` 可用，该选项不是沙箱。进程工具目前面向 POSIX，终端依赖本机 Bash，检索依赖 rg；文件操作只用标准库。

| 工具 | 参数与行为 |
| --- | --- |
| terminal | command、workdir="."、timeout_seconds=20（最大 120）。单次 Bash，无交互 stdin，返回 stdout、stderr、exit_code、timed_out；每路最多保留 16000 字节并明确标记 truncated，同时持续排空输出防止死锁。超时、取消及调用结束会清理进程组 |
| read_file | path、offset=0、limit=20000（4–100000 字节）。读取 UTF-8 普通文件，截断返回 next_offset，用其续读不会切断多字节字符 |
| write_file | path、content。创建或完整覆盖 UTF-8 文件，自动建父目录，原子替换并保留原文件权限；新文件权限 0600 |
| edit_file | path、old_text、new_text。要求原文恰好匹配一次，否则报错且不修改文件；保留换行形式与权限 |
| search_files | pattern、path="."、可选 glob。rg 字面检索，遵循其忽略规则，输出路径、行号和匹配行；退出码 1 的 no_matches=True 表示没有匹配 |

终端每次启动新 shell，cd 和环境变量不跨调用保留，也不提供 PTY、常驻后台会话或进程轮询工具。Limits 的默认工具超时为 30 秒，需要长命令时由应用设置，并为进程清理留出时间。进程组清理不保证终止主动脱离该组的进程。

## 配置与边界

ChatModel 接收显式 Settings(api_key, base_url, model, provider, reasoning="high")，不读取环境变量或本机配置文件；provider="tencent" 启用 TokenHub 会话缓存路由，其余值使用普通兼容请求。base_url 应为 API 根地址，如 https://example.invalid/v1，不包含 /chat/completions，末尾斜杠会在请求时移除。Settings 的 repr 隐藏密钥和入口地址。原始模型正文只写入明确指定的 record_path，调用方负责该路径的访问与保留。

历史、原始输入、事件和去重表都在内存里，由 prune 和 forget 显式释放，尚无自动过期、持久化、跨进程交付或重启恢复。摘要有损；取消不能撤销已发生的外部动作，也无法强制停止正在执行的同步线程。

公开类型可从 aether 导入；跨模块契约位于 aether.contracts，包括 Model、Summarizer、Call、Decision、Tool、Limits、Receipt、Event 和 Summary。Subscription 与 Agent 属于运行实现，通过包入口导出。Model 使用结构化协议，不要求继承基类。
