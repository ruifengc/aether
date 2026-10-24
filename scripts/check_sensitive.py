"""Check worktree and actual index without displaying credential values."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    rb"sk-[A-Za-z0-9_-]{20,}",
    rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    rb"https?://[^\s/@:]+:[^\s/@]+@",
)
EXCLUDED = {".git", ".venv", ".ruff_cache", "__pycache__", "build", "dist", "data"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    if (ROOT / ".git").exists():
        command = ["git", "ls-files", "-z", "--cached"]
        if not args.staged:
            command += ["--others", "--exclude-standard"]
        names = {os.fsdecode(p) for p in subprocess.check_output(command, cwd=ROOT).split(b"\0")
                 if p}
    elif args.staged:
        print("Sensitive check: no Git index in source archive")
        return
    else:
        names = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and
                 not any(part in EXCLUDED or part.endswith(".egg-info")
                         for part in p.relative_to(ROOT).parts)}
    keys = [value.encode() for name, value in os.environ.items() if
            (name.endswith(("_API_KEY", "_TOKEN")) or name == "GH_TOKEN") and len(value) >= 12]
    failed = []
    count = 0
    for name in sorted(names):
        if args.staged:
            content = subprocess.check_output(["git", "show", f":{name}"], cwd=ROOT)
        else:
            path = ROOT / name
            if not path.is_file():
                continue
            content = path.read_bytes()
        count += 1
        private_path = any(part == "data" or part == ".env" or part.startswith(".env.")
                           for part in Path(name).parts)
        if private_path or any(key in content for key in keys) or any(
                re.search(pattern, content) for pattern in PATTERNS):
            failed.append(name)
    for name in failed:
        print("Potential sensitive value:", name)
    print(f"Sensitive check: {count} files, {len(failed)} findings")
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
