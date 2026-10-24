"""Optional local tools. Importing or constructing them never registers tools."""

from __future__ import annotations

import asyncio
import math
import os
import signal
import tempfile
import threading
from pathlib import Path

from .contracts import Tool

OUTPUT_BYTES = 16000
MAX_TIMEOUT = 120


async def _process(argv, cwd, seconds):
    """Drain both streams with bounded retention; own and clean up the process group."""
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or not 0 < seconds <= MAX_TIMEOUT):
        raise ValueError(f"timeout must be between 0 and {MAX_TIMEOUT} seconds")
    loop = asyncio.get_running_loop()
    exited, finished = loop.create_future(), loop.create_future()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    sizes = {"stdout": 0, "stderr": 0}

    class Capture(asyncio.SubprocessProtocol):
        def pipe_data_received(self, fd, chunk):
            name = {1: "stdout", 2: "stderr"}[fd]
            sizes[name] += len(chunk)
            buffers[name].extend(chunk[:max(0, OUTPUT_BYTES - len(buffers[name]))])

        def process_exited(self):
            exited.set_result(None)

        def connection_lost(self, exc):
            finished.set_result(None)

    # The public transport API lets cleanup close pipes independently of process exit.
    transport, _ = await loop.subprocess_exec(
        Capture, *argv, cwd=cwd, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
    )
    timed_out = False
    try:
        async with asyncio.timeout(seconds):
            await asyncio.shield(finished)
    except TimeoutError:
        timed_out = True
    finally:
        # Kill remaining descendants even when the shell exited before its children.
        try:
            os.killpg(transport.get_pid(), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await asyncio.shield(exited)
        transport.close()
    return {
        "cwd": str(cwd), "exit_code": transport.get_returncode(), "timed_out": timed_out,
        **{name: value.decode("utf-8", errors="replace") for name, value in buffers.items()},
        "truncated": any(sizes[name] > OUTPUT_BYTES for name in sizes),
        "output_bytes": sizes,
    }


def _integer(value, name, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")


def _write(path, content):
    """Replace atomically and preserve the existing file's permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode("utf-8"))
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def local_tools(cwd: str | Path | None = None, *,
                include: tuple[str, ...] | list[str] | None = None) -> list[Tool]:
    """Build tools with a fixed path base, without changing the host process cwd."""
    root = Path(cwd or Path.cwd()).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("tool working directory must be a directory")
    writes = threading.Lock()

    def resolve(path):
        if not isinstance(path, str) or not path:
            raise ValueError("path must be nonempty text")
        expanded = Path(path).expanduser()
        return (root / expanded).resolve()

    async def terminal(command: str, workdir: str = ".", timeout_seconds: float = 20):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be nonempty text")
        return await _process(["bash", "-c", command], resolve(workdir), timeout_seconds)

    def read_file(path: str, offset: int = 0, limit: int = 20000):
        _integer(offset, "offset", 0)
        _integer(limit, "limit", 4, 100000)
        target = resolve(path)
        if not target.is_file():
            raise ValueError("read_file requires a regular file")
        with target.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(limit + 1)
        more = len(data) > limit
        data = data[:limit]
        if b"\0" in data:
            raise ValueError("read_file supports UTF-8 text, not binary files")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            if not more or exc.reason != "unexpected end of data":
                raise ValueError("file must be UTF-8; offset must be a character boundary") from None
            data = data[:exc.start]
            content = data.decode("utf-8")
        return {"path": str(target), "content": content, "offset": offset,
                "next_offset": offset + len(data) if more else None, "truncated": more}

    def write_file(path: str, content: str):
        if not isinstance(content, str):
            raise ValueError("content must be text")
        target = resolve(path)
        with writes:
            _write(target, content)
        return {"path": str(target), "bytes_written": len(content.encode("utf-8"))}

    def edit_file(path: str, old_text: str, new_text: str):
        if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str):
            raise ValueError("old_text must be nonempty text and new_text must be text")
        target = resolve(path)
        with writes:
            content = target.read_bytes().decode("utf-8")
            count = content.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must match exactly once; found {count}")
            _write(target, content.replace(old_text, new_text, 1))
        return {"path": str(target), "replacements": 1}

    async def search_files(pattern: str, path: str = ".", glob: str | None = None):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be nonempty text")
        argv = ["rg", "--fixed-strings", "--line-number", "--no-heading", "--color", "never"]
        if glob is not None:
            if not isinstance(glob, str):
                raise ValueError("glob must be text")
            argv += ["--glob", glob]
        result = await _process([*argv, "--", pattern, str(resolve(path))], root, 20)
        result["no_matches"] = result["exit_code"] == 1 and not result["timed_out"]
        return result

    def tool(name, description, properties, required, handler):
        return Tool(name, f"{description} Relative paths resolve from {root}.", {
            "type": "object", "properties": properties, "required": required,
            "additionalProperties": False,
        }, handler)

    text = {"type": "string"}
    available = [
        tool("terminal", "Run one Bash command; return stdout, stderr, exit code, and timeout status. "
             "Each call starts a fresh shell; cd and environment changes do not persist. Set workdir explicitly. "
             "No interactive stdin or persistent services. Retain at most 16000 bytes per stream; "
             "clean up the process group on timeout, cancellation, or completion.", {
                 "command": text, "workdir": text,
                 "timeout_seconds": {"type": "number", "exclusiveMinimum": 0,
                                     "maximum": MAX_TIMEOUT, "default": 20},
             }, ["command"], terminal),
        tool("read_file", "Read a UTF-8 file; offset and limit are byte counts. "
             "Continue truncated reads with next_offset, which preserves UTF-8 character boundaries.", {
                 "path": text, "offset": {"type": "integer", "minimum": 0, "default": 0},
                 "limit": {"type": "integer", "minimum": 4, "maximum": 100000, "default": 20000},
             }, ["path"], read_file),
        tool("write_file", "Create or fully overwrite a UTF-8 file, creating parent directories as needed.", {
            "path": text, "content": text,
        }, ["path", "content"], write_file),
        tool("edit_file", "Replace the unique exact match of old_text with new_text. "
             "Zero or multiple matches fail; read the file first to choose sufficiently specific text.", {
                 "path": text, "old_text": text, "new_text": text,
             }, ["path", "old_text", "new_text"], edit_file),
        tool("search_files", "Search literal text with rg; return paths, line numbers, and matching lines. "
             "Optionally filter files with glob. Respect rg ignore rules; exit code 1 means no matches.", {
                 "pattern": text, "path": text, "glob": text,
             }, ["pattern"], search_files),
    ]
    if include is None:
        return available
    if isinstance(include, str):
        raise ValueError("include must be a sequence of tool names")
    by_name = {tool.name: tool for tool in available}
    selected = []
    for name in include:
        if not isinstance(name, str) or name not in by_name:
            raise ValueError(f"unknown local tool: {name}")
        if name in selected:
            raise ValueError(f"duplicate local tool: {name}")
        selected.append(name)
    return [by_name[name] for name in selected]
