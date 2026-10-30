from __future__ import annotations

import asyncio
import unittest

from tests_helper import ControlledModel, event_of

from aether import Agent, Call, ContextPolicy, Decision, Limits, ModelError, Summary, Tool

POLICY = ContextPolicy(input_bytes=14000, summary_bytes=500, keep_recent=0)


async def read(page):
    return {"page": page, "body": str(page) * 8000}


READ = Tool("read", "read one page", {
    "type": "object", "properties": {"page": {"type": "integer"}},
}, read)


def plan():
    return Decision(calls=tuple(Call("read", {"page": i}) for i in range(3)),
                    usage={"prompt_tokens": 2, "completion_tokens": 1})


class SummaryRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_prefix_is_reused_within_the_actual_request_limit(self):
        for failure in ("invalid_response", "oversized", "timeout"):
            with self.subTest(failure=failure):
                calls = []

                async def summarize(previous, fragment, max_bytes, *, calls=calls, failure=failure):
                    calls.append((previous, fragment, max_bytes))
                    number = len(calls)
                    usage = {"prompt_tokens": number * 10, "completion_tokens": number}
                    if number == 2:
                        if failure == "invalid_response":
                            raise ModelError("invalid once", kind="invalid_response", usage=usage)
                        if failure == "timeout":
                            await asyncio.Event().wait()
                        return Summary("x" * max_bytes, usage)
                    return Summary(f"summary-{number}", usage)

                model = ControlledModel()
                async with Agent(model, [READ], context=POLICY, summarizer=summarize,
                                 limits=Limits(summary_calls=4, retry_delay=0,
                                               model_timeout=.1)) as agent:
                    receipt = await agent.submit("read then answer")
                    _, reply = await model.next()
                    reply.set_result(plan())
                    view, reply = await model.next()
                    self.assertEqual(view["context"]["summary"], "summary-4")
                    reply.set_result(Decision("done", usage={
                        "prompt_tokens": 2, "completion_tokens": 1}))
                    result = await agent.wait(receipt.run_id, timeout=1)
                    self.assertEqual(result["state"], "succeeded")
                    self.assertEqual(result["summary_calls"], 4)
                    self.assertEqual(result["recoveries"], 1)
                    self.assertEqual(len(calls), 4)
                    self.assertEqual(calls.count(calls[0]), 1)
                    self.assertEqual(calls[1], calls[2])
                    self.assertEqual(result["usage"], {
                        "prompt_tokens": 84 if failure == "timeout" else 104,
                        "completion_tokens": 10 if failure == "timeout" else 12,
                        "known": failure != "timeout"})
                    events = agent.query(receipt.run_id, after=0)["events"]
                    self.assertEqual(sum(e.type == "context.summary_reused" for e in events), 1)
                    self.assertFalse(agent._runs[receipt.run_id].summary_cache)

    async def test_partial_prefix_stays_uncommitted_and_new_input_survives_retry(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def summarize(previous, fragment, max_bytes):
            calls.append((previous, fragment, max_bytes))
            if len(calls) == 2:
                entered.set()
                await release.wait()
                raise ModelError("invalid once", kind="invalid_response")
            return Summary("facts")

        model = ControlledModel()
        async with Agent(model, [READ], context=POLICY, summarizer=summarize,
                         limits=Limits(summary_calls=8, retry_delay=0)) as agent:
            receipt = await agent.submit("choose A")
            _, reply = await model.next()
            reply.set_result(plan())
            await asyncio.wait_for(entered.wait(), 1)
            before = agent.query(receipt.run_id)["context"]
            self.assertEqual(before["through"], 0)
            self.assertEqual(before["summary"], "")
            self.assertEqual(len(before["messages"]), 4)
            await agent.submit("correction: choose B", run_id=receipt.run_id)
            release.set()
            view, reply = await model.next()
            self.assertIn("correction: choose B", [m["content"] for m in view["context"]["messages"]])
            self.assertEqual(calls.count(calls[0]), 2)
            self.assertFalse(agent._runs[receipt.run_id].summary_cache)
            reply.set_result(Decision("B"))
            self.assertEqual((await agent.wait(receipt.run_id, timeout=1))["output"], "B")

    async def test_late_piece_after_new_input_is_not_cached_for_retry(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def summarize(previous, fragment, max_bytes):
            calls.append((previous, fragment, max_bytes))
            if len(calls) == 1:
                entered.set()
                await release.wait()
            elif len(calls) == 2:
                raise ModelError("transient once", kind="transient")
            return Summary("facts")

        model = ControlledModel()
        async with Agent(model, [READ], context=POLICY, summarizer=summarize,
                         limits=Limits(summary_calls=8, retry_delay=0)) as agent:
            receipt = await agent.submit("read")
            _, reply = await model.next()
            reply.set_result(plan())
            await asyncio.wait_for(entered.wait(), 1)
            await agent.submit("new requirement", run_id=receipt.run_id)
            release.set()
            view, reply = await model.next()
            self.assertEqual(calls.count(calls[0]), 2)
            self.assertIn("new requirement", [m["content"] for m in view["context"]["messages"]])
            self.assertFalse(any(e.type == "context.summary_reused" for e in
                                 agent.query(receipt.run_id, after=0)["events"]))
            reply.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)

    async def test_context_reduction_does_not_reuse_different_fragments(self):
        calls = []

        async def summarize(previous, fragment, max_bytes):
            calls.append((previous, fragment, max_bytes))
            if len(calls) == 2:
                raise ModelError("context rejected once", kind="context")
            return Summary("facts")

        model = ControlledModel()
        async with Agent(model, [READ], context=POLICY, summarizer=summarize,
                         limits=Limits(summary_calls=12, retry_delay=0)) as agent:
            receipt = await agent.submit("read")
            _, reply = await model.next()
            reply.set_result(plan())
            _, reply = await model.next()
            reply.set_result(Decision("done"))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "succeeded")
            self.assertEqual(result["summary_calls"], len(calls))
            events = agent.query(receipt.run_id, after=0)["events"]
            self.assertFalse(any(e.type == "context.summary_reused" for e in events))

    async def test_cancellation_drops_cached_pieces_without_committing_history(self):
        entered = asyncio.Event()
        calls = 0

        async def summarize(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await asyncio.Event().wait()
            return Summary("facts")

        model = ControlledModel()
        async with Agent(model, [READ], context=POLICY, summarizer=summarize) as agent:
            receipt = await agent.submit("read")
            _, reply = await model.next()
            reply.set_result(plan())
            await asyncio.wait_for(entered.wait(), 1)
            await event_of(agent, receipt.run_id, "context.summary_started")
            await agent.cancel(receipt.run_id)
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "cancelled")
            self.assertEqual(result["context"]["through"], 0)
            self.assertEqual(result["context"]["summary"], "")
            self.assertFalse(agent._runs[receipt.run_id].summary_cache)

    async def test_reuse_does_not_expand_the_actual_request_allowance(self):
        calls = []

        async def summarize(previous, fragment, max_bytes):
            calls.append((previous, fragment, max_bytes))
            if len(calls) > 1:
                raise ModelError("still invalid", kind="invalid_response")
            return Summary("first piece")

        model = ControlledModel()
        async with Agent(model, [READ], context=POLICY, summarizer=summarize,
                         limits=Limits(summary_calls=4, model_retries=10, retry_delay=0)) as agent:
            receipt = await agent.submit("read")
            _, reply = await model.next()
            reply.set_result(plan())
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["error"], "summary_call_limit")
            self.assertEqual(result["summary_calls"], 4)
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls.count(calls[0]), 1)
            self.assertEqual(result["context"]["through"], 0)
            self.assertFalse(agent._runs[receipt.run_id].summary_cache)
