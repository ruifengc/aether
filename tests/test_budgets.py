import json
import unittest

import httpx
from tests_helper import ControlledModel

from aether import Agent, Call, ContextPolicy, Decision, Limits, Summary, Tool
from aether.chat import ChatModel, Settings


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshots_count_the_current_attempt_and_do_not_share_budget_state(self):
        model = ControlledModel()
        limits = Limits(model_calls=3, tool_calls=2, summary_calls=4, model_retries=1)
        async with Agent(model, limits=limits) as agent:
            first = await agent.submit("start", conversation_id="chat")
            view, reply = await model.next()
            self.assertEqual(view["model_calls"], 1)
            self.assertEqual(view["budget"]["model_calls"], {
                "max": 3, "used": 1, "remaining": 2, "unit": "attempts"})
            self.assertEqual(set(view["budget"]), {
                "model_calls", "tool_calls", "summary_calls", "recoveries"})
            self.assertEqual(agent.query(first.run_id)["budget"], view["budget"])
            view["budget"]["tool_calls"]["remaining"] = 100
            self.assertEqual(agent.query(first.run_id)["budget"]["tool_calls"]["remaining"], 2)
            reply.set_result(Decision("done"))
            old = await agent.wait(first.run_id, timeout=1)
            second = await agent.submit("continue", conversation_id="chat")
            view, reply = await model.next()
            self.assertEqual(view["budget"]["model_calls"]["used"], 1)
            reply.set_result(Decision("done"))
            await agent.wait(second.run_id, timeout=1)
            self.assertEqual(agent.query(first.run_id), old)

    async def test_default_adapter_sends_budget_and_structured_feedback_after_history(self):
        requests = []
        executed = []

        def handle(request):
            requests.append(json.loads(request.content))
            message = {"content": "limited"} if len(requests) == 2 else {
                "content": "must not be emitted", "tool_calls": [
                    {"function": {"name": "effect", "arguments": "{}"}} for _ in range(2)]}
            return httpx.Response(200, json={"choices": [{
                "finish_reason": "stop" if len(requests) == 2 else "tool_calls", "message": message}]})

        model = ChatModel(Settings("test", "https://model.invalid", "test", "test"),
                          transport=httpx.MockTransport(handle))
        async with Agent(model, [Tool("effect", "", {}, lambda: executed.append(True))],
                         limits=Limits(model_calls=3, tool_calls=1, retry_delay=0)) as agent:
            receipt = await agent.submit("read two records")
            result = await agent.wait(receipt.run_id, timeout=1)
        self.assertEqual(result["output"], "limited")
        self.assertEqual(result["recoveries"], 1)
        self.assertEqual(executed, [])
        states = [json.loads(r["messages"][1]["content"]) for r in requests]
        self.assertEqual(states[0]["budget"]["model_calls"]["used"], 1)
        self.assertEqual(states[1]["budget"]["model_calls"]["remaining"], 1)
        self.assertEqual(states[1]["feedback"]["code"], "tool_call_limit")
        self.assertEqual(list(states[1]), ["context", "pending", "completed", "budget", "feedback"])
        self.assertEqual(states[0]["context"], states[1]["context"])

    async def test_oversized_batch_can_shrink_without_partial_output_or_effects(self):
        model = ControlledModel()
        executed = []
        effect = Tool("effect", "", {}, lambda: executed.append(True))
        async with Agent(model, [effect], limits=Limits(
                model_calls=3, tool_calls=1, model_retries=1, retry_delay=0)) as agent:
            receipt = await agent.submit("work")
            _, reply = await model.next()
            reply.set_result(Decision("rejected", calls=(Call("effect"), Call("effect"))))
            view, reply = await model.next()
            self.assertEqual(view["output"], None)
            self.assertEqual(view["pending"], [])
            self.assertEqual(view["tool_calls"], 0)
            self.assertEqual(executed, [])
            self.assertEqual(view["feedback"]["requested"], 2)
            self.assertEqual(view["feedback"]["remaining"], 1)
            self.assertEqual(view["budget"]["recoveries"]["remaining"], 0)
            reply.set_result(Decision(calls=(Call("effect"),)))
            view, reply = await model.next()
            self.assertEqual(view["budget"]["model_calls"]["remaining"], 0)
            self.assertEqual(view["budget"]["tool_calls"]["remaining"], 0)
            self.assertEqual(view["feedback"], "")
            reply.set_result(Decision("done"))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "succeeded")
            self.assertEqual(result["model_calls"], 3)
            self.assertEqual(result["recoveries"], 1)
            self.assertEqual(len(executed), 1)

    async def test_exhausted_decision_recovery_or_tool_allowance_fails_explicitly(self):
        for models, retries, spend_tool in ((1, 2, False), (4, 0, False), (4, 2, True)):
            with self.subTest(models=models, retries=retries, spend_tool=spend_tool):
                model = ControlledModel()
                executed = []
                async with Agent(model, [Tool("effect", "", {},
                                             lambda executed=executed: executed.append(True))],
                                 limits=Limits(model_calls=models, tool_calls=1,
                                               model_retries=retries, retry_delay=0)) as agent:
                    receipt = await agent.submit("work")
                    _, reply = await model.next()
                    if spend_tool:
                        reply.set_result(Decision(calls=(Call("effect"),)))
                        _, reply = await model.next()
                    reply.set_result(Decision("rejected", calls=(Call("effect"), Call("effect"))))
                    result = await agent.wait(receipt.run_id, timeout=1)
                    self.assertEqual(result["state"], "failed")
                    self.assertEqual(result["error"], "tool_call_limit")
                    self.assertEqual(result["output"], None)
                    self.assertEqual(result["recoveries"], 0)
                    self.assertEqual(len(executed), int(spend_tool))

    async def test_repeated_oversized_batches_do_not_expand_recovery_allowance(self):
        model = ControlledModel()
        async with Agent(model, limits=Limits(model_calls=5, tool_calls=1,
                                             model_retries=2, retry_delay=0)) as agent:
            receipt = await agent.submit("work")
            for _ in range(3):
                _, reply = await model.next()
                reply.set_result(Decision(calls=(Call("unknown"), Call("unknown"))))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["error"], "tool_call_limit")
            self.assertEqual(result["recoveries"], 2)
            self.assertEqual(result["model_calls"], 3)
            self.assertEqual(result["tool_calls"], 0)

    async def test_budget_bytes_trigger_compaction_and_fit_the_same_input_allowance(self):
        model = ControlledModel()
        policy = ContextPolicy(input_bytes=2400, summary_bytes=200, overhead_bytes=200)
        calls = []

        async def summarize(previous, fragment, max_bytes):
            calls.append(fragment)
            return Summary("facts")

        async with Agent(model, context=policy, summarizer=summarize) as agent:
            receipt = await agent.submit("x" * 1900)
            view, reply = await model.next()
            self.assertTrue(calls)
            self.assertGreater(view["context"]["through"], 0)
            self.assertLessEqual(policy.size(view["context"], [], [], budget=view["budget"]),
                                 policy.input_bytes)
            reply.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)
