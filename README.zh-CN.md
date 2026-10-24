# Aether

[English](README.md) · **简体中文**

简洁、消息驱动的 LLM 运行库，提供异步提交、内部流转、查询与订阅，包含连续上下文、自动压缩、工具注册及有界模型错误恢复。Python 3.11+，运行核心只依赖标准库；终端应用在独立仓库维护。

[完整接口](docs/api.md) · [架构边界](docs/architecture.md) · [协作示例](examples/collaboration.py)

## 使用真实 LLM

克隆仓库并安装可选 HTTP 适配器：

```bash
git clone https://github.com/ruifengc/aether.git
cd aether
uv sync --extra live --no-dev
```

模型凭据与配置由宿主应用提供，库不自动查找环境变量或配置文件。下面的示例由应用代码读取环境变量 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`；`LLM_BASE_URL` 是 API 根地址，例如 `https://example.invalid/v1`。

将以下代码保存为 `chat.py`，按供应商配置上述变量，再运行 `uv run --extra live --no-dev python chat.py`：

```python
import asyncio
import os

from aether import Agent
from aether.chat import ChatModel, Settings


async def main():
    settings = Settings(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        provider="openai-compatible",
        reasoning=None,  # 供应商支持时再设置推理级别。
    )
    async with Agent(ChatModel(settings), system_prompt="Keep exact IDs.") as agent:
        agent.register_prompt("Answer concisely.")
        first = await agent.submit("Remember the fictional project ID ORBIT-17.")
        await agent.wait(first.run_id)
        second = await agent.submit("What was the project ID?",
                                    conversation_id=first.conversation_id)
        result = await agent.wait(second.run_id)
        print(result["state"], result["output"])


asyncio.run(main())
```

这条路径会实际调用模型，需要可访问的模型服务及对应凭据。设置 `provider="tencent"` 可启用 TokenHub 专用缓存路由。业务提示在首次成功提交前注册，计入输入预算，保留在压缩之外。库默认提示使用英文；宿主提示、用户消息和工具结果保留原文。

## 固定脚本的运行机制示例

协作示例默认使用**普通 Python 代码返回固定决定，不执行 LLM 推理**。`ScriptedAssistant` 固定请求 `SKU-A` 的库存，确定性的库存服务返回数量 7。它检查提交、工具执行、另一参与方的 `submit`/`wait`、结果回流和最终输出这条路径，不验证自然语言理解或模型效果。

完成上面的环境安装后，可离线运行：

```bash
uv run --offline --no-sync python examples/collaboration.py
```

预期输出为 `7`。首次安装或同步依赖可能需要联网；固定脚本示例本身不发起网络请求。

如需将固定脚本替换为真实 TokenHub 模型，在应用环境中设置 `TOKENHUB_API_KEY` 后运行：

```bash
uv run --extra live --no-dev python examples/collaboration.py --live
```

这条路径实际调用 `deepseek/deepseek-flash`，可通过 `TOKENHUB_BASE_URL` 覆盖 API 根地址。示例不读取其他应用的配置文件。

## 自定义模型与工具

模型实现 `async decide(snapshot, tools) -> Decision` 即可，不必继承基类；`Model` 与 `Summarizer` 协议从 `aether` 导出。模型读取 `context` 并返回 `Call`，运行库执行工具并回流结果。压缩使用模型的 `summarize` 方法或显式摘要器，后者签名为 `async (previous, fragment, max_bytes) -> Summary`。

普通函数可以注册为工具：

```python
from aether import Agent, Tool
from aether.tools import local_tools


def multiply(a: int, b: int):
    return a * b


custom = Tool("multiply", "Multiply two integers.", {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"], "additionalProperties": False,
}, multiply)
agent = Agent(my_model, tools=[custom])
for tool in local_tools(".", include=("read_file", "search_files")):
    agent.register_tool(tool)
```

库默认无工具，`agent.tools` 为只读映射，追加通过 `register_tool`。同步 handler 在线程池执行，异步 handler 直接 await。一次决定默认等待本批新工具全部返回；`Decision.wait_for` 可选择等待子集，`()` 表示本批全部不阻挡后续判断。

`local_tools` 省略 `include` 时返回 `terminal`、`read_file`、`write_file`、`edit_file`、`search_files` 五种工具；空序列不选择任何工具。构造不执行命令、不改变工作目录、不自动注册。路径基准不是沙箱；进程工具目前面向 POSIX，命令依赖 Bash，检索依赖 `rg`。

## 目录与开发

可导入代码位于 `src/aether`，测试、示例和文档位于安装包之外。这是常用的 [src 布局](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)，可以帮助发现打包错误，避免测试意外导入仓库根目录中的源码。`uv` 会先安装项目再运行。

```bash
bash scripts/check.sh
uv build
```

检查包含工作区和实际 index 的凭据扫描、Ruff、105 项机制与工具测试，以及固定脚本协作示例，不需要 CLI 或原实验工作区。初次安装依赖可能需要联网，检查本身不调用模型。源码发行包包含两份 README、文档、示例、测试与检查脚本。

上下文预算使用 `ContextPolicy(input_bytes=24000, summary_bytes=1200, overhead_bytes=2048)`，按 UTF-8 字节保守估计；模型输出限制 `max_tokens`/`max_output_tokens` 和实际 `usage` 仍以 tokens 计数。

当前为单进程、内存运行库。摘要有损，取消不能撤销外部动作；跨进程交付与重启恢复留待具体需求驱动。
