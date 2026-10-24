from __future__ import annotations

import asyncio
import json
import unittest

import httpx
from tests_helper import ControlledModel

from aether import Agent, ContextPolicy, Decision, Summary
from aether.chat import PROTOCOL_PROMPT, SUMMARY_SYSTEM, ChatModel, Settings


class PromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_registration_reaches_custom_models_without_inference_or_user_input(self):
        model = ControlledModel()
        async with Agent(model, system_prompt="Use evidence.") as agent:
            agent.register_prompt("Return JSON.")
            self.assertTrue(model.requests.empty())
            receipt = await agent.submit("question", conversation_id="chat")
            view, answer = await model.next()
            expected = "Use evidence.\n\nReturn JSON."
            self.assertEqual(view["system_prompt"], expected)
            self.assertEqual([m["content"] for m in view["context"]["messages"]], ["question"])
            answer.set_result(Decision("done"))
            first = await agent.wait(receipt.run_id, timeout=1)
            receipt = await agent.submit("follow up", conversation_id="chat")
            view, answer = await model.next()
            self.assertEqual(view["system_prompt"], expected)
            self.assertEqual(first["system_prompt"], expected)
            answer.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)

    async def test_prompt_configuration_stays_frozen_after_pruning_runs(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            with self.assertRaises(ValueError):
                await agent.submit("bad source", source="tool")
            agent.register_prompt("Before first valid input.")
            receipt = await agent.submit("question")
            _, answer = await model.next()
            with self.assertRaisesRegex(RuntimeError, "before the first"):
                agent.register_prompt("changed while deciding")
            answer.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)
            agent.prune(keep_runs=0)
            with self.assertRaisesRegex(RuntimeError, "before the first"):
                agent.register_prompt("changed after pruning")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            agent.register_prompt("after close")

    async def test_invalid_prompt_registration_does_not_modify_configuration(self):
        model = ControlledModel()
        for value in (None, 42):
            with self.assertRaises(TypeError):
                Agent(model, system_prompt=value)
        async with Agent(model) as agent:
            for value in (None, 42):
                with self.assertRaises(TypeError):
                    agent.register_prompt(value)
            for value in ("", " \n"):
                with self.assertRaises(ValueError):
                    agent.register_prompt(value)
            receipt = await agent.submit("question")
            view, answer = await model.next()
            self.assertNotIn("system_prompt", view)
            answer.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)

    async def test_prompt_survives_compaction_and_is_not_a_summarizer_instruction(self):
        model = ControlledModel()
        fragments = []
        policy = ContextPolicy(input_bytes=2400, summary_bytes=200,
                               overhead_bytes=200, keep_recent=2)
        prompt = "HOST-POLICY: preserve evidence identities and return JSON."

        async def summarize(previous, fragment, max_bytes):
            fragments.append(fragment)
            self.assertNotIn(prompt, previous)
            self.assertNotIn(prompt, fragment)
            return Summary("older facts")

        async with Agent(model, context=policy, summarizer=summarize, system_prompt=prompt) as agent:
            receipt = await agent.submit("archive " * 500)
            view, answer = await model.next()
            self.assertTrue(fragments)
            self.assertGreater(view["context"]["through"], 0)
            self.assertEqual(view["system_prompt"], prompt)
            self.assertLessEqual(policy.size(view["context"], [], [], system_prompt=prompt),
                                 policy.input_bytes)
            answer.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)

    async def test_oversized_prompt_is_counted_and_not_silently_compressed(self):
        model = ControlledModel()
        policy = ContextPolicy(input_bytes=2400, summary_bytes=200, overhead_bytes=200)
        prompt = "规则" * 1200

        async def summarize(*args):
            self.fail("fixed instructions cannot be summarized away")

        async with Agent(model, context=policy, summarizer=summarize, system_prompt=prompt) as agent:
            receipt = await agent.submit("question")
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "failed")
            self.assertIn("system prompt", result["error"])
            self.assertEqual(result["system_prompt"], prompt)
            self.assertTrue(model.requests.empty())

    async def test_valid_large_prompt_can_reduce_old_user_retention_to_fit(self):
        model = ControlledModel()
        policy = ContextPolicy(input_bytes=3000, summary_bytes=200,
                               overhead_bytes=200, keep_recent=0)
        prompt = "p" * 2000

        async def summarize(*args):
            return Summary("older user requirements and tool facts")

        async with Agent(model, context=policy, summarizer=summarize, system_prompt=prompt) as agent:
            await agent.submit("q" * 600, conversation_id="chat")
            receipt = await agent.submit("archive " * 500, conversation_id="chat", source="external")
            view, answer = await model.next()
            self.assertGreater(view["context"]["through"], 0)
            self.assertEqual(view["system_prompt"], prompt)
            self.assertLessEqual(policy.size(view["context"], [], [], system_prompt=prompt), 3000)
            answer.set_result(Decision("done"))
            self.assertEqual((await agent.wait(receipt.run_id, timeout=1))["state"], "succeeded")

    async def test_shared_adapter_keeps_business_prompts_isolated_and_out_of_summary(self):
        requests = []

        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            system = body["messages"][0]["content"]
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "content": "ALPHA" if "APP-ALPHA" in system else "BETA"}}]})

        model = ChatModel(Settings("test", "https://model.invalid", "test", "test"),
                          transport=httpx.MockTransport(handle))
        async with (Agent(model, system_prompt="APP-ALPHA") as alpha,
                    Agent(model, system_prompt="APP-BETA") as beta):
            receipts = await asyncio.gather(alpha.submit("same question"), beta.submit("same question"))
            results = await asyncio.gather(alpha.wait(receipts[0].run_id), beta.wait(receipts[1].run_id))
        self.assertEqual([result["output"] for result in results], ["ALPHA", "BETA"])
        for request in requests:
            system = request["messages"][0]["content"]
            self.assertTrue(system.startswith(PROTOCOL_PROMPT))
            self.assertEqual(system.count("APP-"), 1)
            self.assertNotIn("APP-", request["messages"][1]["content"])
        await model.summarize("", "facts", 200)
        self.assertEqual(requests[-1]["messages"][0]["content"], SUMMARY_SYSTEM)
        self.assertNotIn("APP-", json.dumps(requests[-1]))
