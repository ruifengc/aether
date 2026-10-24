#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --extra live python scripts/check_sensitive.py
uv run --extra live python scripts/check_sensitive.py --staged
uv run --extra live ruff check .
uv run --extra live python -m unittest discover -s tests -v
uv run --extra live python examples/collaboration.py
