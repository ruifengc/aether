from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
import tempfile
import unittest
from pathlib import Path

from aether import Agent, Call, Decision, Tool
from aether.tools import OUTPUT_BYTES, local_tools


class LocalToolsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tools = {t.name: t for t in local_tools(self.root)}

    async def execute(self, name, **arguments):
        return await self.tools[name].execute(arguments)

    async def test_local_tools_are_explicit_and_can_be_selected_without_registering(self):
        selected = local_tools(self.root, include=("read_file", "terminal"))
        self.assertEqual([tool.name for tool in selected], ["read_file", "terminal"])
        self.assertEqual(local_tools(self.root, include=()), [])
        async with Agent(object()) as agent:
            self.assertEqual(agent.tools, {})
            agent.register_tool(selected[0])
            self.assertEqual(list(agent.tools), ["read_file"])
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_invalid_tool_selection_is_rejected(self):
        for include in ("terminal", ("missing",), ("read_file", "read_file"), (123,)):
            with self.subTest(include=include), self.assertRaises(ValueError):
                local_tools(self.root, include=include)

    async def test_terminal_uses_workdir_reports_exit_and_does_not_change_host_cwd(self):
        previous = Path.cwd()
        (self.root / "subdir").mkdir()
        result = await self.execute("terminal", command="pwd; printf failure >&2; exit 7",
                                    workdir="subdir")
        self.assertEqual(result["stdout"].strip(), str(self.root / "subdir"))
        self.assertEqual(result["stderr"], "failure")
        self.assertEqual(result["exit_code"], 7)
        self.assertFalse(result["timed_out"])
        self.assertEqual(Path.cwd(), previous)
        eof = await self.execute("terminal", command="read value; printf '%s' $?")
        self.assertEqual(eof["stdout"], "1")  # It cannot consume the CLI's stdin.

    async def test_terminal_drains_large_stdout_and_stderr_with_bounded_retention(self):
        program = "import os; os.write(1, b'x'*200000); os.write(2, b'y'*200000)"
        result = await self.execute("terminal", command=shlex.join([sys.executable, "-c", program]))
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "x" * OUTPUT_BYTES)
        self.assertEqual(result["stderr"], "y" * OUTPUT_BYTES)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["output_bytes"], {"stdout": 200000, "stderr": 200000})

    async def test_timeout_and_cancellation_stop_descendants(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                pidfile = self.root / f"pid-{cancel}"
                marker = self.root / f"late-{cancel}"
                program = ("import os,time,pathlib; "
                           f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                           f"time.sleep(.6); pathlib.Path({str(marker)!r}).write_text('late')")
                task = asyncio.create_task(self.execute(
                    "terminal", command=shlex.join([sys.executable, "-c", program]) + " & wait",
                    timeout_seconds=.3 if not cancel else 10))
                async with asyncio.timeout(2):
                    while not pidfile.exists():  # noqa: ASYNC110 - external process PID file
                        await asyncio.sleep(.005)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 2)
                else:
                    result = await asyncio.wait_for(task, 2)
                    self.assertTrue(result["timed_out"])
                    self.assertNotEqual(result["exit_code"], 0)
                # A killed orphan may be a zombie until init reaps it.
                status = Path(f"/proc/{pidfile.read_text()}/stat")
                if await asyncio.to_thread(status.exists):
                    self.assertEqual((await asyncio.to_thread(status.read_text)).split()[2], "Z")
                self.assertFalse(marker.exists())

    async def test_utf8_pagination_reconstructs_long_lines_and_preserves_crlf(self):
        content = "first\r\n" + "中文🙂" * 40 + "\r\nlast"
        await self.execute("write_file", path="nested/note.txt", content=content)
        pieces, offset = [], 0
        while True:
            result = await self.execute("read_file", path="nested/note.txt", offset=offset, limit=13)
            pieces.append(result["content"])
            if result["next_offset"] is None:
                break
            self.assertGreater(result["next_offset"], offset)
            offset = result["next_offset"]
        self.assertEqual("".join(pieces), content)

    async def test_detached_child_holding_pipes_cannot_hold_timeout_open(self):
        pidfile = self.root / "detached.pid"
        program = ("import subprocess,pathlib,time; "
                   "p=subprocess.Popen(['sleep','10'],start_new_session=True); "
                   f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(10)")
        try:
            result = await asyncio.wait_for(self.execute(
                "terminal", command=shlex.join([sys.executable, "-c", program]),
                timeout_seconds=.3), 2)
            self.assertTrue(result["timed_out"])
            self.assertTrue(pidfile.exists())
        finally:
            # Deliberately detached processes are outside the tool's process group.
            if pidfile.exists():
                try:
                    os.killpg(int(pidfile.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    async def test_exact_edit_rejects_missing_or_ambiguous_text_without_modifying_file(self):
        path = self.root / "note.txt"
        path.write_bytes(b"alpha\r\nalpha\r\nbeta\r\n")
        path.chmod(0o640)
        for old in ("missing", "alpha"):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                await self.execute("edit_file", path="note.txt", old_text=old, new_text="changed")
            self.assertEqual(path.read_bytes(), b"alpha\r\nalpha\r\nbeta\r\n")
        await self.execute("edit_file", path="note.txt", old_text="beta", new_text="gamma")
        self.assertEqual(path.read_bytes(), b"alpha\r\nalpha\r\ngamma\r\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o640)
        self.assertEqual(list(self.root.iterdir()), [path])

    async def test_invalid_text_and_arguments_fail_before_writing_or_spawning(self):
        (self.root / "binary").write_bytes(b"a\x00b")
        with self.assertRaisesRegex(ValueError, "binary"):
            await self.execute("read_file", path="binary")
        for value in (True, -1, float("nan"), 121):
            with self.assertRaisesRegex(ValueError, "timeout"):
                await self.execute("terminal", command="touch unexpected", timeout_seconds=value)
        with self.assertRaisesRegex(ValueError, "content"):
            await self.execute("write_file", path="unexpected", content=123)
        self.assertFalse((self.root / "unexpected").exists())

    async def test_literal_search_glob_no_match_and_option_like_pattern(self):
        (self.root / "one.txt").write_text("prefix a.b\na-b\n--help\n", encoding="utf-8")
        (self.root / "two.py").write_text("a.b\n", encoding="utf-8")
        result = await self.execute("search_files", pattern="a.b", glob="*.txt")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("one.txt:1:prefix a.b", result["stdout"])
        self.assertNotIn("two.py", result["stdout"])
        self.assertNotIn("a-b", result["stdout"])
        result = await self.execute("search_files", pattern="--help")
        self.assertIn("one.txt:3:--help", result["stdout"])
        result = await self.execute("search_files", pattern="not-present")
        self.assertTrue(result["no_matches"])
        self.assertEqual(result["stderr"], "")

    async def test_real_tools_run_through_kernel_and_errors_are_returned_to_model(self):
        observed = []

        class Model:
            async def decide(_, snapshot, tools):
                results = [item for item in snapshot["inputs"] if item["source"] == "tool"]
                if not results:
                    return Decision(calls=(Call("read_file", {"path": "missing.txt"}),))
                observed.extend(results)
                if len(results) == 1:
                    return Decision(calls=(Call("terminal", {"command": "printf recovered"}),))
                return Decision(results[-1]["content"]["stdout"])

        async with Agent(Model(), list(self.tools.values())) as agent:
            receipt = await agent.submit("read and recover")
            result = await agent.wait(receipt.run_id, timeout=3)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["output"], "recovered")
        self.assertIn("error", observed[0]["content"])


class RegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_registration_is_available_to_next_decision_without_triggering_inference(self):
        schemas = []

        class Model:
            async def decide(self, snapshot, tools):
                schemas.append(tools)
                if not snapshot["completed"]:
                    return Decision(calls=(Call("added"),))
                return Decision("done")

        async with Agent(Model()) as agent:
            tool = Tool("added", "user tool", {"type": "object"}, lambda: 42)
            agent.register_tool(tool)
            self.assertEqual(schemas, [])
            with self.assertRaisesRegex(ValueError, "unique"):
                agent.register_tool(tool)
            receipt = await agent.submit("use it")
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["completed"][0]["name"], "added")
            self.assertEqual(schemas[0][0]["function"]["name"], "added")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            agent.register_tool(Tool("after_close", "", {}, lambda: None))

    async def test_tool_registry_is_read_only_and_view_tracks_registration(self):
        async with Agent(object()) as agent:
            view = agent.tools
            tool = Tool("lookup", "", {}, lambda: 1)
            with self.assertRaises(TypeError):
                view["lookup"] = tool
            agent.register_tool(tool)
            self.assertIs(view["lookup"], tool)
            with self.assertRaises(TypeError):
                del view["lookup"]
            with self.assertRaises(AttributeError):
                agent.tools = {}

    async def test_custom_model_can_execute_a_tool_named_await_input(self):
        class Model:
            async def decide(self, snapshot, tools):
                if not snapshot["completed"]:
                    return Decision(calls=(Call("await_input"),))
                return Decision(str(snapshot["inputs"][-1]["content"]))

        async with Agent(Model(), [Tool("await_input", "", {}, lambda: 42)]) as agent:
            receipt = await agent.submit("run")
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "succeeded")
            self.assertEqual(result["output"], "42")
