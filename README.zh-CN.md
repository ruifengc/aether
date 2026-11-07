# Aether

[English](README.md) · **简体中文**

Aether 是一个消息驱动的 LLM Agent 内核。它管理模型、工具和新消息之间的循环，让应用能够提交任务，在执行期间补充信息，并观察处理结果。

模型决定下一步做什么；Aether 负责执行，包括会话历史、并发工具调用、等待、取消、上下文压缩和有限次数的错误恢复。每次运行按顺序处理状态变化，模型和工具请求在队列外执行。如果模型请求期间收到新消息，旧决定会在发布输出或启动工具前被丢弃。

[设计说明](docs/design.zh-CN.md) · [API 参考](docs/api.zh-CN.md) · [模型接入](docs/models.zh-CN.md) · [使用示例](examples/follow_up.py)

## 开始使用

需要 Python 3.11+。运行内核只使用标准库；可选的 OpenAI 兼容连接使用 HTTPX。

```sh
git clone https://github.com/ruifengc/aether.git
cd aether
uv sync --extra live
```

按供应商设置 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`。由应用代码明确读取配置：

```python
import asyncio
import os

from aether import Agent
from aether.adapters import OpenAIModel, Settings


async def main():
    model = OpenAIModel(Settings(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
    ))
    async with Agent(model) as agent:
        receipt = await agent.submit("解释热泵如何工作。")
        await agent.submit("重点讲冬季住宅供暖。", run_id=receipt.run_id)
        result = await agent.wait(receipt.run_id)
        print(result["state"], result["output"])


asyncio.run(main())
```

`submit` 在输入被接收后返回；`wait` 等待本次运行结束。应用也可以通过查询和订阅观察执行过程。Aether 自行管理模型的 HTTP 连接，在 Agent 关闭时释放。

## 添加工具

工具由名称、说明、JSON 参数定义和普通 Python 函数组成。应用按需注册需要的能力：

```python
from aether import Tool

async def stock(sku: str):
    return {"sku": sku, "available": 7}

agent.register_tool(Tool("stock", "查询 SKU 库存。", {
    "type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"],
}, stock))
```

[补充要求示例](examples/follow_up.py) 使用真实模型选择设备，并在查询设备资料的工具执行期间增加重量限制。初始任务和后续要求通过同一个提交接口进入内核。示例需要模型凭据，会产生实际 API 调用费用。

## 适用范围

内核在单个进程中运行，状态保存在内存里。会话可以跨多次运行延续，每次运行有独立的执行额度。上下文压缩保留近期消息，并对旧历史生成有损摘要。取消会停止后续处理，但无法撤销已经完成的外部操作。

模型实现公共的 `Model` 接口。供应商请求格式由模型适配器处理；终端交互和应用配置由应用负责。独立的 [Aether CLI](https://github.com/ruifengc/aether-cli) 提供终端界面。

## 开发

```sh
bash scripts/check.sh
uv build
```

检查覆盖运行行为、模型协议处理、本地工具、代码风格和凭据扫描，不调用真实模型。公共接口定义在 `contracts.py`，具体实现位于 `runtime.py`、`context.py` 和 `adapters/`。
