import asyncio
import unittest

from tests_helper import ControlledModel

from aether import Agent, Call, Decision, Limits, Tool


async def until(agent, run_id, predicate):
    subscription = agent.subscribe(run_id, after=0, buffer=512)
    try:
        async with asyncio.timeout(1):
            async for event in subscription:
                if predicate(event):
                    return event
    finally:
        subscription.close()
    raise AssertionError("expected event did not arrive")


async def consumed(agent, receipt):
    await until(agent, receipt.run_id, lambda e: e.type == "input.consumed"
                and e.data["seq"] >= receipt.input_version)


class CriticalWaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_batch_coalesces_inputs_without_intermediate_inference(self):
        model = ControlledModel()
        gates = [asyncio.Event(), asyncio.Event()]

        async def lookup(index):
            await gates[index].wait()
            return index

        async with Agent(model, [Tool("lookup", "", {}, lookup)]) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision(calls=(Call("lookup", {"index": 0}),
                                             Call("lookup", {"index": 1}))))
            await until(agent, receipt.run_id, lambda e: e.type == "run.waiting")
            gates[0].set()
            for index in range(10):
                update = await agent.submit(f"amendment {index}", run_id=receipt.run_id)
            update = await agent.submit("notice", run_id=receipt.run_id, source="external")
            await consumed(agent, update)
            await until(agent, receipt.run_id, lambda e: e.type == "input.consumed"
                        and e.data["source"] == "tool")
            view = agent.query(receipt.run_id)
            self.assertEqual(view["model_calls"], 1)
            self.assertEqual(view["consumed_version"], view["input_version"])
            self.assertTrue(model.requests.empty())
            subscription = agent.subscribe(receipt.run_id)
            self.assertEqual((await anext(subscription)).type, "snapshot")
            subscription.close()
            gates[1].set()
            view, answer = await model.next()
            self.assertEqual(view["waiting_for"], [])
            self.assertEqual(len(view["inputs"]), 14)
            self.assertEqual([i["content"] for i in view["inputs"] if i["source"] == "tool"],
                             [0, 1])
            answer.set_result(Decision("done"))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["model_calls"], 2)

    async def test_subset_releases_while_noncritical_operation_remains_then_waits_by_id(self):
        model = ControlledModel()
        gates = [asyncio.Event(), asyncio.Event()]

        async def lookup(index):
            await gates[index].wait()
            return index

        async with Agent(model, [Tool("lookup", "", {}, lookup)]) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision(calls=(Call("lookup", {"index": 0}),
                                             Call("lookup", {"index": 1})), wait_for=(0,)))
            gates[0].set()
            view, answer = await model.next()
            self.assertEqual(len(view["pending"]), 1)
            operation = view["pending"][0]["id"]
            answer.set_result(Decision("partial", wait_for=(operation,)))
            await until(agent, receipt.run_id, lambda e: e.type == "output")
            update = await agent.submit("changed", run_id=receipt.run_id)
            await consumed(agent, update)
            self.assertEqual(agent.query(receipt.run_id)["model_calls"], 2)
            gates[1].set()
            view, answer = await model.next()
            self.assertEqual(view["pending"], [])
            self.assertEqual(view["output"], "partial")
            self.assertIn("changed", [i["content"] for i in view["inputs"]])
            answer.set_result(Decision("final"))
            await agent.wait(receipt.run_id, timeout=1)

    async def test_errors_and_timeouts_release_waiting_set(self):
        model = ControlledModel()

        async def never():
            await asyncio.Event().wait()

        async with Agent(model, [Tool("never", "", {}, never)],
                         limits=Limits(tool_timeout=.03)) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision(calls=(Call("unknown"), Call("never")), wait_for=(0, 1)))
            view, answer = await model.next()
            self.assertEqual(view["waiting_for"], [])
            self.assertEqual(len(view["completed"]), 2)
            self.assertTrue(all(i["status"] == "error" for i in view["completed"]))
            answer.set_result(Decision("errors handled"))
            await agent.wait(receipt.run_id, timeout=1)

    async def test_cancel_bypasses_critical_wait_and_clears_it(self):
        model = ControlledModel()
        stopped = asyncio.Event()
        started = asyncio.Event()

        async def never():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async with Agent(model, [Tool("never", "", {}, never)]) as agent:
            receipt = await agent.submit("start")
            _, answer = await model.next()
            answer.set_result(Decision(calls=(Call("never"),), wait_for=(0,)))
            await asyncio.wait_for(started.wait(), 1)
            await agent.cancel(receipt.run_id)
            result = await agent.wait(receipt.run_id, timeout=1)
            await asyncio.wait_for(stopped.wait(), 1)
            self.assertEqual(result["state"], "cancelled")
            self.assertEqual(result["waiting_for"], [])
            self.assertEqual(result["model_calls"], 1)

    async def test_invalid_references_recover_before_output_or_side_effects(self):
        for refs in ((-1,), (1,), (True,), ("missing",), ({},), 3.14):
            with self.subTest(refs=refs):
                model = ControlledModel()
                executed = []
                async with Agent(model, [Tool("effect", "", {},
                                             lambda executed=executed: executed.append(True))],
                                 limits=Limits(retry_delay=0)) as agent:
                    receipt = await agent.submit("start")
                    _, answer = await model.next()
                    answer.set_result(Decision("invalid output", calls=(Call("effect"),),
                                               wait_for=refs))
                    view, answer = await model.next()
                    self.assertEqual(executed, [])
                    self.assertEqual(view["output"], None)
                    self.assertEqual(view["pending"], [])
                    answer.set_result(Decision("fixed"))
                    await agent.wait(receipt.run_id, timeout=1)

    async def test_default_only_waits_new_batch_not_existing_nonblocking_operation(self):
        model = ControlledModel()
        release = asyncio.Event()

        async def slow():
            await release.wait()
            return "slow fact"

        async with Agent(model, [Tool("slow", "", {}, slow),
                                 Tool("fast", "", {}, lambda: "fast fact")]) as agent:
            receipt = await agent.submit("start background operation")
            _, answer = await model.next()
            answer.set_result(Decision(calls=(Call("slow"),), wait_for=()))
            await until(agent, receipt.run_id, lambda e: e.type == "tool.started")
            update = await agent.submit("new independent query", run_id=receipt.run_id)
            view, answer = await model.next()
            self.assertEqual(view["inputs"][-1]["seq"], update.input_version)
            slow_id = view["pending"][0]["id"]
            answer.set_result(Decision(calls=(Call("fast"),)))
            view, answer = await model.next()
            self.assertEqual(view["waiting_for"], [])
            self.assertEqual([p["id"] for p in view["pending"]], [slow_id])
            answer.set_result(Decision("fast delivered", wait_for=(slow_id,)))
            await until(agent, receipt.run_id, lambda e: e.type == "output")
            release.set()
            _, answer = await model.next()
            answer.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)
