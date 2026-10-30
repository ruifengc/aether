# 组合模型与交付前校核

Aether 的模型契约是 `async decide(snapshot, tools) -> Decision`。宿主可以根据当前快照，把每次决定委托给不同模型，让生成、校核和修订共用一个 run 的调用预算、用量和生命周期。

## 候选、校核与修订

[可运行示例](../examples/reviewed_research.py) 使用虚构库存数据演示流程：

1. 生成模型提出候选后，包装模型将它转为普通 stage_candidate 工具调用，正文暂不作为 output 发布。
2. 工具结果回到运行队列，下一次 decide 委托校核模型；校核模型可以接受候选，也可以请求已注册的业务工具补充事实。
3. 工具返回后继续生成，再次提出候选并校核；接受后的正文才作为最终输出。

模型路由只读取快照，不私存运行进度。每次生成或校核都经过 Agent 的正常决策路径，统一计入 model_calls 和 usage，遵守超时、取消及 freshness guard。需要调用真实模型时，每次 decide 委托一个模型请求；不在工具 handler 中隐藏额外模型请求。暂存候选本身计一次工具调用。

示例中的模型是固定 Python 决定，不调用真实模型。初始候选数量为 0，校核后查询到 SKU-A 的库存为 7，修订并再次校核后交付：

```bash
uv run --offline --no-sync python examples/reviewed_research.py
```

```json
{"state": "succeeded", "output": "7", "model_calls": 4, "tool_calls": 3}
```

把模型调用上限降为三次时，运行会以 model_call_limit 失败，不发布尚未通过校核的候选。校核期间收到新输入时，旧决定按同一版本规则丢弃，后续决定使用最新快照。

该配方面向默认整批等待、一次一个小候选的流程。候选正文进入普通工具参数和 completed 记录，需要预留相应输入额度；长文、并发候选、持久化和异步人工决定由宿主另行建模。替换脚本模型时，可按需要向 Agent 显式传入摘要器。工具由宿主注册，库不会自动开启校核。

## 原生消息适配

默认 ChatModel 从状态快照构造请求，不保留供应商的原生 assistant/tool 对话历史。自定义 Model 可以选择其他消息表达，但仍需遵守运行契约：

- 分离模型提出的决定与内核接受的决定；被 freshness guard 丢弃的结果不能进入后续有效历史。
- 正确关联并发工具调用与实际结果，处理重复参数、取消、重试及上下文压缩，避免孤立工具消息。
- 按会话隔离历史，把发送的原生消息及推理字段计入输入额度，不能通过适配器私有缓存绕过 ContextPolicy。

Decision/Call 当前不携带供应商原始工具 ID 或原生消息提交凭据。需要完整 transcript 的适配器应先明确这些生命周期与预算边界，再决定如何与宿主组合。快照模式提供的 context、pending、completed 和 budget 仍是自定义适配器可用的运行信息。

## 运行期限由宿主明确管理

wait(timeout) 只限制该等待方，不自动取消共享 run。拥有整个任务期限的宿主可以显式取消：

```python
try:
    result = await agent.wait(receipt.run_id, timeout=300)
except TimeoutError:
    await agent.cancel(receipt.run_id)
    result = await agent.wait(receipt.run_id)
```

模型看到的 budget 反映已配置的计数限制，不包含虚构的 remaining_time。已经结束的 run 保持终态；继续同一会话需要通过 conversation_id 提交新的 run。

完整接口与限制见 [API 说明](api.md)，状态提交和慢操作的职责见 [架构说明](architecture.md)。
