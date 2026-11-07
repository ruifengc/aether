# Aether

Read README.md and the relevant documents under docs/ before changing code. Keep the design small, explicit and easy to follow.

- Use uv for Python dependencies and execution. Work on a feature branch.
- Push changes to Gitea for review. Publish or synchronize GitHub only after review approval; do not open GitHub PRs as the review path.
- Run bash scripts/check.sh and uv build. Before committing, inspect and scan the actual Git index.
- Credentials are supplied at runtime. Never put them in source, commits, remote URLs or logs.
- Keep public data and interfaces in contracts.py; execution in runtime.py; provider protocols and connections in adapters/.
- Accept new messages while work is running. Keep state changes ordered, with model and tool calls outside the run queue.
- Never allow an old model decision to publish output or start tools after newer input is accepted.
- Agent manages model shutdown. Applications configure models and tools, but do not manage HTTP clients.
- Tools are enabled by registration. Do not introduce an approval mechanism.
- Keep the kernel independent of CLI, local configuration discovery and provider SDKs.
- Update English and Chinese documentation together. Use ordinary words and actual API names.
- Keep examples useful with real models. Put deterministic scheduling fixtures in tests; keep development reports and probes outside user documentation.
- Keep tests for observable behavior and important failures. Remove duplicate checks and experiment-only switches.
- Change interfaces and their users together. Do not retain forwarding modules without an actual compatibility requirement.
