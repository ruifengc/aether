from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from tests_helper import ControlledModel

from aether import Agent, Call, Decision, Limits, SubscriptionOverflow, Tool


def tool(handler, name="lookup"):
    return Tool(name, "test tool", {"type": "object", "properties": {}}, handler)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = ControlledModel()
        self.agent = Agent(self.model)

    async def asyncTearDown(self):
        await self.agent.close()

    async def test_wait_does_not_subscribe_or_query_during_progress(self):
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        with (patch.object(self.agent, "query", wraps=self.agent.query) as query,
              patch.object(self.agent, "subscribe", wraps=self.agent.subscribe) as subscribe):
            waiting = asyncio.create_task(self.agent.wait(receipt.run_id, timeout=1))
            try:
                await asyncio.sleep(0)
                for index in range(8):
                    await self.agent.submit(str(index), run_id=receipt.run_id)
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                self.assertEqual(query.call_count, 0)
                first.set_result(Decision("stale"))
                _, second = await self.model.next()
                second.set_result(Decision("fresh"))
                self.assertEqual((await waiting)["output"], "fresh")
                self.assertEqual(query.call_count, 0)
                self.assertEqual(subscribe.call_count, 0)
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)

    async def test_receipt_does_not_wait_for_model_and_query_is_read_only(self):
        receipt = await asyncio.wait_for(self.agent.submit("start"), .1)
        snapshot, future = await self.model.next()
        for _ in range(10):
            self.assertEqual(self.agent.query(receipt.run_id)["model_calls"], 1)
        self.assertEqual(snapshot["inputs"][0]["source"], "user")
        future.set_result(Decision("done"))
        self.assertEqual((await self.agent.wait(receipt.run_id, timeout=1))["output"], "done")

    async def test_input_during_model_invalidates_old_final(self):
        receipt = await self.agent.submit("old")
        _, first = await self.model.next()
        await self.agent.submit("new", run_id=receipt.run_id)
        first.set_result(Decision("old answer"))
        view, second = await self.model.next()
        self.assertEqual([i["content"] for i in view["inputs"]], ["old", "new"])
        second.set_result(Decision("new answer"))
        result = await self.agent.wait(receipt.run_id, timeout=1)
        self.assertEqual(result["output"], "new answer")
        self.assertEqual(result["discarded_decisions"], 1)
        self.assertEqual([e.data["content"] for e in
                          self.agent.query(receipt.run_id, after=0)["events"] if e.type == "output"],
                         ["new answer"])

    async def test_input_invalidates_old_tool_plan_before_execution(self):
        executed = []
        self.agent.register_tool(tool(lambda: executed.append(True)))
        receipt = await self.agent.submit("old")
        _, first = await self.model.next()
        await self.agent.submit("do something else", run_id=receipt.run_id)
        first.set_result(Decision(calls=(Call("lookup"),), wait_for=(0,)))
        view, second = await self.model.next()
        self.assertEqual(view["waiting_for"], [])
        second.set_result(Decision("done"))
        await self.agent.wait(receipt.run_id, timeout=1)
        self.assertEqual(executed, [])

    async def test_user_input_and_tool_result_share_mailbox(self):
        gate = asyncio.Event()

        async def lookup():
            await gate.wait()
            return {"value": 7}

        self.agent.register_tool(tool(lookup))
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision(calls=(Call("lookup"),), wait_for=()))
        await self._event(receipt.run_id, "tool.started")
        await self.agent.submit("additional constraint", run_id=receipt.run_id)
        view, second = await self.model.next()
        self.assertEqual(len(view["pending"]), 1)
        self.assertEqual(view["inputs"][-1]["content"], "additional constraint")
        second.set_result(Decision(wait=True))
        await self._event(receipt.run_id, "run.waiting")
        gate.set()
        view, third = await self.model.next()
        self.assertEqual(view["inputs"][-1]["source"], "tool")
        self.assertEqual(view["inputs"][-1]["content"], {"value": 7})
        third.set_result(Decision("7"))
        self.assertEqual((await self.agent.wait(receipt.run_id, timeout=1))["output"], "7")

    async def test_wait_for_user_does_not_need_resume_api(self):
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision("which city?", wait=True))
        await self._event(receipt.run_id, "run.waiting")
        self.assertEqual(self.agent.query(receipt.run_id)["output"], "which city?")
        await self.agent.submit("Hangzhou", run_id=receipt.run_id)
        _, second = await self.model.next()
        second.set_result(Decision("Hangzhou"))
        await self.agent.wait(receipt.run_id, timeout=1)

    async def test_partial_results_can_drive_work_without_all_tools(self):
        gate = asyncio.Event()

        async def slow():
            await gate.wait()
            return "slow"

        self.agent.register_tool(tool(lambda: "fast", "fast"))
        self.agent.register_tool(tool(slow, "slow"))
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision(calls=(Call("fast"), Call("slow")), wait_for=(0,)))
        view, second = await self.model.next()
        self.assertEqual(view["inputs"][-1]["content"], "fast")
        self.assertEqual(view["pending"][0]["name"], "slow")
        second.set_result(Decision(wait=True))
        await self._event(receipt.run_id, "run.waiting")
        gate.set()
        _, third = await self.model.next()
        third.set_result(Decision("both"))
        await self.agent.wait(receipt.run_id, timeout=1)

    async def test_repeated_message_id_is_idempotent_and_reuse_is_checked(self):
        first = await self.agent.submit("start", message_id="one")
        second = await self.agent.submit("start", message_id="one")
        self.assertTrue(second.duplicate)
        self.assertEqual(first.run_id, second.run_id)
        with self.assertRaises(ValueError):
            await self.agent.submit("different", message_id="one")
        _, reply = await self.model.next()
        reply.set_result(Decision("done"))
        result = await self.agent.wait(first.run_id, timeout=1)
        self.assertEqual(len(result["inputs"]), 1)

    async def test_snapshot_and_subscriber_payloads_cannot_mutate_run(self):
        receipt = await self.agent.submit({"goal": "start"})
        _, reply = await self.model.next()
        sub = self.agent.subscribe(receipt.run_id)
        snapshot = (await anext(sub)).data
        snapshot["inputs"][0]["content"]["goal"] = "tampered"
        snapshot["context"]["messages"][0]["content"]["goal"] = "tampered context"
        self.assertEqual(self.agent.query(receipt.run_id)["inputs"][0]["content"], {"goal": "start"})
        self.assertEqual(self.agent.query(receipt.run_id)["context"]["messages"][0]["content"],
                         {"goal": "start"})
        sub.close()
        reply.set_result(Decision("done"))
        await self.agent.wait(receipt.run_id, timeout=1)

    async def test_query_cursor_then_subscribe_replays_without_gap(self):
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        cursor = self.agent.query(receipt.run_id)["cursor"]
        await self.agent.submit("new", run_id=receipt.run_id)
        sub = self.agent.subscribe(receipt.run_id, after=cursor)
        first.set_result(Decision("stale"))
        _, second = await self.model.next()
        second.set_result(Decision("done"))
        await self.agent.wait(receipt.run_id, timeout=1)
        events = [event async for event in sub]
        self.assertEqual([e.seq for e in events], list(range(cursor + 1, events[-1].seq + 1)))
        self.assertEqual(events[-1].type, "run.succeeded")

    async def test_slow_subscriber_reports_overflow_without_blocking_run(self):
        receipt = await self.agent.submit("start")
        sub = self.agent.subscribe(receipt.run_id, buffer=1)
        _, first = await self.model.next()
        first.set_result(Decision("done"))
        result = await self.agent.wait(receipt.run_id, timeout=1)
        self.assertEqual(result["state"], "succeeded")
        with self.assertRaises(SubscriptionOverflow):
            await anext(sub)

    async def test_two_runs_do_not_mix_inputs(self):
        one = await self.agent.submit("one")
        two = await self.agent.submit("two")
        for _ in range(2):
            snapshot, reply = await self.model.next()
            reply.set_result(Decision(snapshot["inputs"][0]["content"]))
        self.assertEqual((await self.agent.wait(one.run_id, timeout=1))["output"], "one")
        self.assertEqual((await self.agent.wait(two.run_id, timeout=1))["output"], "two")

    async def test_cancel_keeps_terminal_state_when_tool_ignores_cancellation(self):
        gate = asyncio.Event()

        async def stubborn():
            try:
                await gate.wait()
            except asyncio.CancelledError:
                pass
            return "late result"

        self.agent.register_tool(tool(stubborn))
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision(calls=(Call("lookup"),)))
        await self._event(receipt.run_id, "tool.started")
        # Let the tool coroutine actually start before cancelling it.
        await asyncio.sleep(0)
        await self.agent.cancel(receipt.run_id)
        self.assertEqual((await self.agent.wait(receipt.run_id, timeout=1))["state"], "cancelled")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(self.agent.query(receipt.run_id)["state"], "cancelled")
        self.assertEqual(self.agent.query(receipt.run_id)["model_calls"], 1)

    async def test_model_timeout_fails_instead_of_waiting_forever(self):
        self.agent.limits = Limits(model_timeout=.02, retry_delay=0)
        receipt = await self.agent.submit("start")
        result = await self.agent.wait(receipt.run_id, timeout=1)
        self.assertEqual(result["state"], "failed")
        self.assertIn("TimeoutError", result["error"])
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(result["recoveries"], 3)

    async def test_tool_timeout_becomes_input_for_next_decision(self):
        self.agent.limits = Limits(tool_timeout=.02)

        async def slow():
            await asyncio.Event().wait()

        self.agent.register_tool(tool(slow))
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision(calls=(Call("lookup"),)))
        view, second = await self.model.next()
        self.assertIn("TimeoutError", view["inputs"][-1]["content"]["error"])
        second.set_result(Decision("tool timed out"))
        await self.agent.wait(receipt.run_id, timeout=1)

    async def test_sync_tool_does_not_block_mailbox(self):
        started, release = threading.Event(), threading.Event()

        def blocking():
            started.set()
            release.wait(1)
            return "done"

        self.agent.register_tool(tool(blocking))
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision(calls=(Call("lookup"),), wait_for=()))
        await asyncio.to_thread(started.wait, 1)
        try:
            await asyncio.wait_for(self.agent.submit("extra", run_id=receipt.run_id), .2)
            view, second = await self.model.next()
            self.assertTrue(view["pending"])
            self.assertEqual(view["inputs"][-1]["content"], "extra")
            second.set_result(Decision(wait=True))
            await self._event(receipt.run_id, "run.waiting")
        finally:
            release.set()
        _, third = await self.model.next()
        third.set_result(Decision("done"))
        await self.agent.wait(receipt.run_id, timeout=1)

    async def test_queued_inputs_coalesce_while_tools_keep_running(self):
        gate = asyncio.Event()

        async def lookup():
            await gate.wait()
            return "fact"

        self.agent.register_tool(tool(lookup))
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision(calls=(Call("lookup"),), wait_for=()))
        await self._event(receipt.run_id, "tool.started")
        for index in range(5):
            await self.agent.submit(f"update-{index}", run_id=receipt.run_id)
        view, second = await self.model.next()
        self.assertEqual(len(view["inputs"]), 6)
        self.assertEqual(self.agent.query(receipt.run_id)["model_calls"], 2)
        self.assertTrue(view["pending"])
        second.set_result(Decision(wait=True))
        await self._event(receipt.run_id, "run.waiting")
        await asyncio.sleep(0)
        self.assertTrue(self.model.requests.empty())
        gate.set()
        view, third = await self.model.next()
        self.assertEqual(len(view["inputs"]), 7)
        third.set_result(Decision("done"))
        self.assertEqual((await self.agent.wait(receipt.run_id, timeout=1))["model_calls"], 3)

    async def test_cancel_admission_prevents_queued_tool_plan_from_starting(self):
        executed = []
        self.agent.register_tool(tool(lambda: executed.append(True)))
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision(calls=(Call("lookup"),)))
        await self.agent.cancel(receipt.run_id)
        with self.assertRaises(ValueError):
            await self.agent.submit("too late", run_id=receipt.run_id)
        result = await self.agent.wait(receipt.run_id, timeout=1)
        self.assertEqual(result["state"], "cancelled")
        self.assertEqual(executed, [])

    async def test_many_inputs_during_one_model_call_are_merged_in_one_next_decision(self):
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        for index in range(8):
            await self.agent.submit(str(index), run_id=receipt.run_id)
        first.set_result(Decision("old"))
        view, second = await self.model.next()
        self.assertEqual(len(view["inputs"]), 9)
        second.set_result(Decision("done"))
        self.assertEqual((await self.agent.wait(receipt.run_id, timeout=1))["model_calls"], 2)

    async def test_unknown_tool_and_bad_arguments_are_reported_as_inputs(self):
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        first.set_result(Decision(calls=(Call("missing"),)))
        view, second = await self.model.next()
        self.assertIn("unknown tool", view["inputs"][-1]["content"]["error"])
        second.set_result(Decision("cannot use missing tool"))
        await self.agent.wait(receipt.run_id, timeout=1)

    async def test_model_call_limit_and_truncation_are_explicit_failures(self):
        self.agent.limits = Limits(model_calls=1)
        receipt = await self.agent.submit("start")
        _, first = await self.model.next()
        await self.agent.submit("new", run_id=receipt.run_id)
        first.set_result(Decision("old"))
        result = await self.agent.wait(receipt.run_id, timeout=1)
        self.assertEqual(result["error"], "model_call_limit")
        another = await self.agent.submit("start")
        _, reply = await self.model.next()
        reply.set_result(Decision("partial", finish_reason="length"))
        self.assertEqual((await self.agent.wait(another.run_id, timeout=1))["state"], "failed")

    async def test_ablation_without_freshness_guard_can_publish_old_answer(self):
        self.agent.freshness_guard = False
        receipt = await self.agent.submit("old")
        _, first = await self.model.next()
        await self.agent.submit("new", run_id=receipt.run_id)
        first.set_result(Decision("old answer"))
        self.assertEqual((await self.agent.wait(receipt.run_id, timeout=1))["output"], "old answer")

    async def test_wait_timeout_does_not_cancel_business_work(self):
        receipt = await self.agent.submit("start")
        _, reply = await self.model.next()
        with self.assertRaises(TimeoutError):
            await self.agent.wait(receipt.run_id, timeout=.01)
        reply.set_result(Decision("done"))
        self.assertEqual((await self.agent.wait(receipt.run_id, timeout=1))["state"], "succeeded")

    async def _event(self, run_id, kind):
        sub = self.agent.subscribe(run_id, after=0)
        try:
            async with asyncio.timeout(1):
                async for event in sub:
                    if event.type == kind:
                        return event
        finally:
            sub.close()
        self.fail(f"missing event: {kind}")
