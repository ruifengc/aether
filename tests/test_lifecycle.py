from __future__ import annotations

import asyncio
import gc
import unittest
import weakref

from tests_helper import ControlledModel

from aether import Agent, Call, ContextPolicy, Decision, Tool


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def turn(self, agent, model, text, conversation="chat", message_id=None):
        receipt = await agent.submit(text, conversation_id=conversation, message_id=message_id)
        view, answer = await model.next()
        answer.set_result(Decision(text))
        result = await agent.wait(receipt.run_id, timeout=1)
        return receipt, view, result

    async def test_many_events_and_one_timed_out_waiter_do_not_break_result_waiting(self):
        model = ControlledModel()
        async with Agent(model, context=ContextPolicy(input_bytes=1000000)) as agent:
            receipt = await agent.submit("start")
            _, first = await model.next()
            impatient = asyncio.create_task(agent.wait(receipt.run_id, timeout=.01))
            patient = asyncio.create_task(agent.wait(receipt.run_id, timeout=1))
            try:
                await asyncio.sleep(0)
                for index in range(200):
                    await agent.submit(str(index), run_id=receipt.run_id)
                with self.assertRaises(TimeoutError):
                    await impatient
                self.assertFalse(patient.done())
                first.set_result(Decision("stale"))
                _, answer = await model.next()
                answer.set_result(Decision("fresh"))
                self.assertEqual((await patient)["output"], "fresh")
            finally:
                impatient.cancel()
                patient.cancel()
                await asyncio.gather(impatient, patient, return_exceptions=True)

    async def test_prune_preserves_context_and_retained_message_deduplication(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            first, _, old = await self.turn(agent, model, "first", message_id="m1")
            second, _, current = await self.turn(agent, model, "second", message_id="m2")
            other, _, _ = await self.turn(agent, model, "other", conversation="other")
            self.assertEqual(agent.prune(keep_runs=1, conversation_id="chat"), [first.run_id])
            self.assertEqual(agent.query(second.run_id), current)
            self.assertEqual(old["output"], "first")
            with self.assertRaises(KeyError):
                agent.query(first.run_id)
            with self.assertRaises(KeyError):
                agent.subscribe(first.run_id)
            with self.assertRaises(KeyError):
                await agent.wait(first.run_id)
            duplicate = await agent.submit("second", conversation_id="chat", message_id="m2")
            self.assertTrue(duplicate.duplicate)
            third = await agent.submit("reuse", conversation_id="chat", message_id="m1")
            self.assertFalse(third.duplicate)
            view, answer = await model.next()
            self.assertEqual([m["content"] for m in view["context"]["messages"]],
                             ["first", "first", "second", "second", "reuse"])
            answer.set_result(Decision("done"))
            await agent.wait(third.run_id, timeout=1)
            self.assertEqual(agent.query(other.run_id)["output"], "other")

    async def test_prune_all_records_allows_continuing_the_conversation(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            first, _, _ = await self.turn(agent, model, "remember")
            self.assertEqual(agent.prune(keep_runs=0), [first.run_id])
            second, view, _ = await self.turn(agent, model, "continue")
            self.assertEqual([m["content"] for m in view["context"]["messages"]],
                             ["remember", "remember", "continue"])
            self.assertNotEqual(second.run_id, first.run_id)

    async def test_prune_keeps_most_recent_completion_instead_of_creation(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            slow = await agent.submit("slow")
            _, slow_answer = await model.next()
            fast = await agent.submit("fast")
            _, fast_answer = await model.next()
            fast_answer.set_result(Decision("fast"))
            await agent.wait(fast.run_id, timeout=1)
            slow_answer.set_result(Decision("slow"))
            await agent.wait(slow.run_id, timeout=1)
            self.assertEqual(agent.prune(keep_runs=1), [fast.run_id])
            self.assertEqual(agent.query(slow.run_id)["output"], "slow")

    async def test_forget_rejects_active_work_and_does_not_touch_another_conversation(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            first = await agent.submit("first", conversation_id="chat", message_id="m1")
            _, pending = await model.next()
            other, _, _ = await self.turn(agent, model, "other", conversation="other")
            with self.assertRaisesRegex(ValueError, "active work"):
                agent.forget("chat")
            self.assertEqual(agent.prune(keep_runs=0, conversation_id="chat"), [])
            pending.set_result(Decision("first answer"))
            await agent.wait(first.run_id, timeout=1)
            agent.forget("chat")
            self.assertEqual(agent.query(other.run_id)["output"], "other")
            with self.assertRaises(KeyError):
                agent.query(first.run_id)
            fresh, view, _ = await self.turn(agent, model, "fresh", message_id="m1")
            self.assertFalse(fresh.duplicate)
            self.assertEqual(len(view["context"]["messages"]), 1)

    async def test_close_and_prune_release_receipts_that_never_reached_the_pump(self):
        agent = Agent(ControlledModel())
        first = await agent.submit("first", message_id="m1")
        await agent.submit("second", run_id=first.run_id, message_id="m2")
        await agent.close()
        self.assertEqual(agent.query(first.run_id)["inputs"], [])
        self.assertEqual(agent.prune(keep_runs=0), [first.run_id])
        self.assertEqual(agent._receipts, {})
        agent.forget(first.conversation_id)
        self.assertEqual(agent._conversations, {})

    async def test_unfinished_background_task_prevents_releasing_its_terminal_run(self):
        model = ControlledModel()
        started, release = asyncio.Event(), asyncio.Event()

        async def stubborn():
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return "late"

        async with Agent(model, [Tool("slow", "", {}, stubborn)]) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision(calls=(Call("slow"),)))
            await asyncio.wait_for(started.wait(), 1)
            await agent.cancel(receipt.run_id)
            self.assertEqual((await agent.wait(receipt.run_id, timeout=1))["state"], "cancelled")
            self.assertEqual(agent.prune(keep_runs=0), [])
            with self.assertRaisesRegex(ValueError, "background tasks"):
                agent.forget(receipt.conversation_id)
            release.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(agent.prune(keep_runs=0), [receipt.run_id])

    async def test_closed_subscription_does_not_retain_a_pruned_run(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            sub = agent.subscribe(receipt.run_id, after=0)
            reference = weakref.ref(agent._runs[receipt.run_id])
            answer.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)
            agent.prune(keep_runs=0)
            await asyncio.sleep(0)
            gc.collect()
            self.assertIsNone(reference())
            events = [event async for event in sub]
            self.assertEqual(events[-1].type, "run.succeeded")

    async def test_one_waiter_can_prune_without_breaking_another_waiters_result(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()

            async def wait_and_prune():
                result = await agent.wait(receipt.run_id, timeout=1)
                self.assertEqual(agent.prune(keep_runs=0), [receipt.run_id])
                return result

            first = asyncio.create_task(wait_and_prune())
            second = asyncio.create_task(agent.wait(receipt.run_id, timeout=1))
            await asyncio.sleep(0)
            answer.set_result(Decision("done"))
            results = await asyncio.gather(first, second)
            self.assertEqual([result["output"] for result in results], ["done", "done"])

    async def test_prune_rejects_invalid_limits_and_unknown_scope(self):
        async with Agent(ControlledModel()) as agent:
            for keep in (-1, True, 1.5):
                with self.assertRaisesRegex(ValueError, "keep_runs"):
                    agent.prune(keep_runs=keep)
            with self.assertRaises(KeyError):
                agent.prune(conversation_id="missing")
            with self.assertRaises(KeyError):
                agent.forget("missing")
