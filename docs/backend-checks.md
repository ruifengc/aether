# 后端基础检查

默认适配器使用状态快照，不回放供应商原生 assistant/tool 历史。它能发送文本与 function calling，但没有图片内容块、推理历史保留或流式输出。缺少这些能力不等于缓存必然失效，也不能仅凭接口兼容就声称性能等价。

## 客户端与用量观察

- 可借用宿主管理的 HTTP 客户端，供决策和摘要复用连接；宿主负责关闭。
- 思考开关独立于 reasoning_effort，明确关闭时不再发送强度字段。
- 汇总已报告的缓存、推理及其他整数用量明细，保留字段覆盖次数；不把缺失字段当成零。
- model.completed 记录整次决定耗时、finish_reason 和原始 usage，可定位截断与恢复；整次耗时不是 TTFT。
- CLI 修复命令行默认值覆盖环境推理强度的问题，提供思考开关和初始/恢复输出 token 额度参数。

这些改动的协议测试使用可控 HTTP 返回，验证请求、所有权与计量行为，不证明在线加速、实际命中率或模型任务能力。默认输出额度暂不调整；先观察 length 和 reasoning_tokens，再决定具体后端的默认值。

## 真实请求对照

明确配置 TOKENHUB_API_KEY 后，在源码目录运行（会产生真实模型费用）：

```bash
uv run --extra live python scripts/probe_backend.py --output /tmp/aether-audit.json
```

默认使用请求名 deepseek/deepseek-flash、high effort、每次初始 4096 token、两次工具任务对照。每次任务最多五次决策；截断恢复最高 8192 token。整次检查最多 33 次请求，不对外部工具执行写入。--trials 可设 1 至 3，--max-tokens 可设 1 至 16384。

检查包含：

1. 同一会话内，追加尾部消息的三次长快照请求，观察实际缓存字段。
2. 通过 Aether 执行先查库存、再核验回执的任务，检查调用顺序与精确 JSON 结果。
3. 使用原生顺序 assistant/tool 消息完成相同任务，分别保留与省略 reasoning_content，观察接受情况、用量和耗时。

报告只保存请求模型名、供应商返回模型名、用量、耗时、结束原因、推理文本长度及任务检查结果，不保存凭据、HTTP 头、供应商错误正文或推理文本。返回模型名只作为标签，不能证明后台版本。

原生顺序对照没有 Aether 的异步状态、版本拒绝和压缩机制，系统提示与控制工具集合也不同；它用于发现适配差异，不能把差值全部归因于消息格式。省略推理历史的分组可能预期返回 400；报告保留这一结果，命令非零退出不自动代表回归。供应商没有报告缓存字段时记为未知。样本很小，不能据此声称长任务或普遍性能更优。

如需直连 DeepSeek，使用 DEEPSEEK_API_KEY 并明确指定 --provider deepseek --model deepseek-flash。TokenHub 结果不能代替 DeepSeek 直连验证。

## 适配边界

原生推理历史、流式交付和图片输入尚未实现。原生历史若进入正式适配器，必须同时处理被拒绝的决定、并发工具关联、取消、压缩和输入预算，不能私存一份绕过这些机制的 transcript。仅有顺序任务对照不构成替换默认消息协议的证据。

## 官方参考

- [TokenHub DeepSeek 调用指南](https://cloud.tencent.com/document/product/1823/132248)
- [TokenHub 缓存命中指南](https://cloud.tencent.com/document/product/1823/131410)
- [DeepSeek 思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)
- [DeepSeek 缓存](https://api-docs.deepseek.com/zh-cn/guides/kv_cache/)
- [DeepSeek 图像理解](https://api-docs.deepseek.com/zh-cn/guides/vision/)
