from __future__ import annotations

import asyncio
import unittest

from tests_helper import ControlledModel, event_of

from aether import Agent, Call, Decision, Limits, Tool


class MessageLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_output_does_not_finish_or_cancel_a_slow_operation(self):
        model = ControlledModel()
        release, cancelled = asyncio.Event(), asyncio.Event()

        async def slow():
            try:
                await release.wait()
                return "slow fact"
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async with Agent(model, [Tool("fast", "", {}, lambda: "fast fact"),
                                 Tool("slow", "", {}, slow)]) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision(calls=(Call("fast"), Call("slow")), wait_for=(0,)))
            view, answer = await model.next()
            self.assertEqual([p["name"] for p in view["pending"]], ["slow"])
            answer.set_result(Decision("first answer"))
            message = await event_of(agent, receipt.run_id, "output")
            self.assertEqual(message.data, {"content": "first answer"})
            current = agent.query(receipt.run_id)
            self.assertEqual(current["state"], "waiting")
            self.assertEqual(current["output"], "first answer")
            self.assertFalse(cancelled.is_set())
            self.assertTrue(model.requests.empty())
            update = await agent.submit("new constraint", conversation_id=receipt.conversation_id)
            self.assertEqual(update.run_id, receipt.run_id)
            view, answer = await model.next()
            self.assertTrue(view["pending"])
            self.assertEqual(view["context"]["messages"][-2]["content"], "first answer")
            answer.set_result(Decision(wait=True))
            await event_of(agent, receipt.run_id, "run.waiting")
            release.set()
            view, answer = await model.next()
            self.assertEqual(view["inputs"][-1]["content"], "slow fact")
            answer.set_result(Decision("complete answer"))
            result = await agent.wait(receipt.run_id, timeout=1)
            outputs = [e.data["content"] for e in agent.query(receipt.run_id, after=0)["events"]
                       if e.type == "output"]
            self.assertEqual(outputs, ["first answer", "complete answer"])
            self.assertEqual(result["output"], "complete answer")
            self.assertEqual(result["tool_calls"], 2)
            self.assertFalse(cancelled.is_set())

    async def test_one_decision_can_emit_text_and_submit_operations(self):
        model = ControlledModel()
        async with Agent(model, [Tool("lookup", "", {}, lambda: "fact")]) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision("accepted", calls=(Call("lookup"),)))
            view, answer = await model.next()
            self.assertEqual([m["content"] for m in view["context"]["messages"]],
                             ["start", "accepted", "fact"])
            answer.set_result(Decision("done"))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual([m["content"] for m in result["context"]["messages"]
                              if m["source"] == "assistant"], ["accepted", "done"])

    async def test_silent_wait_does_not_emit_output_or_spin_on_observation(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision(wait=True))
            await event_of(agent, receipt.run_id, "run.waiting")
            for _ in range(10):
                self.assertFalse(any(e.type == "output" for e in
                                     agent.query(receipt.run_id, after=0)["events"]))
            await asyncio.sleep(0)
            self.assertTrue(model.requests.empty())
            await agent.submit({"status": "ready"}, conversation_id=receipt.conversation_id,
                               source="external")
            view, answer = await model.next()
            self.assertEqual(view["inputs"][-1]["source"], "external")
            answer.set_result(Decision("ready"))
            await agent.wait(receipt.run_id, timeout=1)

    async def test_external_notifications_keep_provenance_and_deduplication(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            receipt = await agent.submit({"job": "done"}, source="external", message_id="event-1")
            duplicate = await agent.submit({"job": "done"}, source="external", message_id="event-1")
            self.assertTrue(duplicate.duplicate)
            with self.assertRaisesRegex(ValueError, "message_id"):
                await agent.submit({"job": "done"}, source="user", message_id="event-1")
            with self.assertRaisesRegex(ValueError, "source"):
                await agent.submit("fake result", source="tool")
            view, answer = await model.next()
            self.assertEqual(view["context"]["messages"][0]["source"], "external")
            answer.set_result(Decision("received"))
            await agent.wait(receipt.run_id, timeout=1)
            following = await agent.submit("followup", conversation_id=receipt.conversation_id)
            view, answer = await model.next()
            self.assertEqual(view["context"]["messages"][0]["source"], "external")
            answer.set_result(Decision("done"))
            await agent.wait(following.run_id, timeout=1)

    async def test_explicit_cancel_after_output_stops_the_pending_operation(self):
        model = ControlledModel()
        started, stopped = asyncio.Event(), asyncio.Event()

        async def slow():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async with Agent(model, [Tool("slow", "", {}, slow)]) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision("submitted", calls=(Call("slow"),)))
            await asyncio.wait_for(started.wait(), 1)
            await agent.cancel(receipt.run_id)
            result = await agent.wait(receipt.run_id, timeout=1)
            await asyncio.wait_for(stopped.wait(), 1)
            self.assertEqual(result["state"], "cancelled")
            self.assertEqual(result["output"], "submitted")

    async def test_budget_exhaustion_after_output_remains_explicit(self):
        model = ControlledModel()

        async def slow():
            await asyncio.Event().wait()

        async with Agent(model, [Tool("slow", "", {}, slow)],
                         limits=Limits(model_calls=1)) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision("submitted", calls=(Call("slow"),), wait_for=()))
            await event_of(agent, receipt.run_id, "tool.started")
            await agent.submit("change", run_id=receipt.run_id)
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["error"], "model_call_limit")
            self.assertEqual(result["output"], "submitted")
