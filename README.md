# Aether

消息驱动的 LLM 运行库：异步提交、内部流转、查询与订阅，包含连续上下文、自动压缩、工具注册及有界错误恢复。Python 3.11+，运行核心只依赖标准库，不附带 CLI 入口。

[完整接口](docs/api.md) · [架构边界](docs/architecture.md) · [可运行示例](examples/collaboration.py)

## 安装与快速运行

在本项目目录安装源码；移到独立仓库后命令相同：

```bash
uv venv
uv pip install .
# 使用默认 HTTP 适配器时：
uv pip install '.[live]'
```

无需密钥或网络即可运行协作示例：

```bash
uv run python examples/collaboration.py
```

例子注册普通工具，将另一个参与方的 submit/wait 组合成库存查询，输出 7。--live 切换到真实模型，需要宿主设置 TOKENHUB_API_KEY；不会读取其他应用的配置文件。

下面是一个完整的连续对话示例：

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
        reasoning="high",  # 供应商不支持时设为 None
    )
    async with Agent(ChatModel(settings), system_prompt="Use evidence; keep exact IDs.") as agent:
        agent.register_prompt("Answer concisely.")
        first = await agent.submit("记住虚构项目编号 ORBIT-17。")
        await agent.wait(first.run_id)
        second = await agent.submit("刚才的编号是什么？", conversation_id=first.conversation_id)
        result = await agent.wait(second.run_id)
        print(result["state"], result["output"])


asyncio.run(main())
```

Settings 由宿主显式提供，库不读取环境变量或文件。上例的环境变量读取属于宿主代码。provider="tencent" 启用 TokenHub 专用缓存路由，其余值走普通兼容协议。业务提示在首次成功提交前注册，计入输入预算，保留在压缩之外。库默认提示使用英文，宿主提示、输入和工具结果不翻译。

## 自定义模型与工具

模型实现 async decide(snapshot, tools) -> Decision 即可，不必继承基类；可从 aether 导入 Model 和 Summarizer 协议。模型从 context 读取摘要与原文，返回 Call 发起工具；工具结果由库回流。超预算时使用模型的 summarize 或显式 summarizer，签名为 async (previous, fragment, max_bytes) -> Summary。

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

Agent 默认无工具，agent.tools 为只读映射，所有追加注册走 register_tool。同步函数在线程池执行，协程工具直接 await。普通调用默认等本批，Decision.wait_for 可指定等待子集或用 () 表示全部不等，详见接口说明。

local_tools 省略 include 返回 terminal、read_file、write_file、edit_file、search_files 五种工具，空序列返回空列表。构造不执行命令、不改变工作目录、不自动注册。路径基准不是沙箱；进程工具目前面向 POSIX，终端依赖 Bash，检索依赖 rg。

## 开发检查与构建

```bash
bash scripts/check.sh
uv build
```

检查入口安装开发依赖与 live 扩展，执行 Ruff、机制及工具测试和无网络协作示例，不需要 CLI、工作区根目录或实验资料。源码发行包包含文档、示例、测试与检查脚本，可独立重建。

输入、摘要与包装预算使用 ContextPolicy(input_bytes=24000, summary_bytes=1200, overhead_bytes=2048)。这些单位是保守 UTF-8 字节；HTTP 输出限制 max_tokens/max_output_tokens 和实际 usage 是 tokens。旧 *_tokens 预算字段已改名，不保留同义参数。

目前是单进程内存运行库。摘要有损，取消不能撤销外部动作；跨进程交付与重启恢复仍由后续具体需求决定。
