from __future__ import annotations

import asyncio
import json
import unittest

import httpx
from tests_helper import ControlledModel, event_of

from aether import Agent, Call, ContextPolicy, Decision, Limits, ModelError, Summary, Tool
from aether.chat import ChatModel, Settings
from aether.context import Context, compact, json_bytes

SMALL = ContextPolicy(input_bytes=2400, summary_bytes=200, overhead_bytes=200, keep_recent=2)


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_participants_can_request_and_reply_without_completion_loop(self):
        class Service:
            async def decide(self, snapshot, tools):
                return Decision("7")

        model = ControlledModel()
        calls = []
        async with Agent(Service()) as service:
            async def ask():
                receipt = await service.submit({"sku": "A"})
                calls.append(receipt.run_id)
                return (await service.wait(receipt.run_id))["output"]

            async with Agent(model, [Tool("service", "", {}, ask)]) as agent:
                receipt = await agent.submit("stock?")
                _, reply = await model.next()
                reply.set_result(Decision(calls=(Call("service"),)))
                view, reply = await model.next()
                self.assertEqual(view["inputs"][-1]["content"], "7")
                reply.set_result(Decision("7"))
                await agent.wait(receipt.run_id, timeout=1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(service.query(calls[0])["model_calls"], 1)

    async def test_completed_turn_keeps_both_sides_and_old_snapshot_is_stable(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            first = await agent.submit("compare A and B", conversation_id="chat")
            _, answer = await model.next()
            answer.set_result(Decision("A is lighter"))
            old = await agent.wait(first.run_id, timeout=1)
            second = await agent.submit("choose the lighter one", conversation_id="chat")
            view, answer = await model.next()
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertEqual([m["content"] for m in view["context"]["messages"]],
                             ["compare A and B", "A is lighter", "choose the lighter one"])
            self.assertEqual(agent.query(first.run_id), old)
            answer.set_result(Decision("A"))
            await agent.wait(second.run_id, timeout=1)

    async def test_active_conversation_routes_to_same_run_and_records_question(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            first = await agent.submit("start", conversation_id="chat")
            _, answer = await model.next()
            answer.set_result(Decision("which city?", wait=True))
            await event_of(agent, first.run_id, "run.waiting")
            second = await agent.submit("Hangzhou", conversation_id="chat")
            self.assertEqual(first.run_id, second.run_id)
            view, answer = await model.next()
            self.assertEqual([m["source"] for m in view["context"]["messages"]],
                             ["user", "assistant", "user"])
            answer.set_result(Decision("done"))
            await agent.wait(first.run_id, timeout=1)

    async def test_cancelled_conversation_can_continue_with_explicit_outcome(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            first = await agent.submit("old", conversation_id="chat")
            await model.next()
            await agent.cancel(first.run_id)
            await agent.wait(first.run_id, timeout=1)
            second = await agent.submit("new", conversation_id="chat")
            view, answer = await model.next()
            self.assertEqual(view["context"]["messages"][1]["content"]["state"], "cancelled")
            self.assertEqual(view["recoveries"], 0)
            answer.set_result(Decision("new answer"))
            await agent.wait(second.run_id, timeout=1)

    async def test_conversation_scope_is_checked_for_deduplication_and_runs(self):
        model = ControlledModel()
        async with Agent(model) as agent:
            first = await agent.submit("x", conversation_id="one", message_id="m")
            duplicate = await agent.submit("x", conversation_id="one", message_id="m")
            self.assertTrue(duplicate.duplicate)
            self.assertEqual(duplicate.conversation_id, "one")
            with self.assertRaises(ValueError):
                await agent.submit("x", conversation_id="two", message_id="m")
            with self.assertRaises(ValueError):
                await agent.submit("x", conversation_id="two", run_id=first.run_id)

    async def test_new_input_during_compaction_survives_and_query_does_not_infer(self):
        model = ControlledModel()
        started, release = asyncio.Event(), asyncio.Event()

        async def summarize(previous, fragment, max_bytes):
            started.set()
            await release.wait()
            return Summary("user initially asked for A", {"prompt_tokens": 4, "completion_tokens": 2})

        async with Agent(model, context=SMALL, summarizer=summarize) as agent:
            receipt = await agent.submit("A " * 2500)
            await asyncio.wait_for(started.wait(), 1)
            await agent.submit("correction: choose B", conversation_id=receipt.conversation_id)
            for _ in range(5):
                self.assertEqual(agent.query(receipt.run_id)["model_calls"], 0)
            release.set()
            view, answer = await model.next()
            self.assertEqual(view["context"]["messages"][-1]["content"], "correction: choose B")
            self.assertIn("A", view["context"]["summary"])
            self.assertLessEqual(SMALL.size(view["context"], [], []), SMALL.input_bytes)
            answer.set_result(Decision("B"))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertGreater(result["summary_calls"], 1)
            self.assertGreater(result["usage"]["prompt_tokens"], 0)

    async def test_compaction_can_be_cancelled_without_late_answer(self):
        model = ControlledModel()
        started = asyncio.Event()

        async def summarize(*args):
            started.set()
            await asyncio.Event().wait()

        async with Agent(model, context=SMALL, summarizer=summarize) as agent:
            receipt = await agent.submit("x" * 5000)
            await asyncio.wait_for(started.wait(), 1)
            await agent.cancel(receipt.run_id)
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "cancelled")
            self.assertEqual(result["model_calls"], 0)

    async def test_completed_operation_metadata_survives_lossy_context_compaction(self):
        model = ControlledModel()

        async def summarize(*args):
            # Even a summary that omits operation metadata cannot erase execution facts.
            return Summary("tool supplied a large document")

        async with Agent(model, [Tool("lookup", "", {}, lambda: "x" * 5000)],
                         context=SMALL, summarizer=summarize) as agent:
            receipt = await agent.submit("read")
            _, reply = await model.next()
            reply.set_result(Decision(calls=(Call("lookup"),)))
            view, reply = await model.next()
            operation = view["completed"][0]
            self.assertEqual(operation["name"], "lookup")
            self.assertEqual(operation["arguments"], {})
            self.assertEqual(operation["status"], "completed")
            self.assertLessEqual(operation["context_message_id"], view["context"]["through"])
            self.assertLessEqual(SMALL.size(view["context"], [{}], [],
                                          completed=view["completed"]), SMALL.input_bytes)
            reply.set_result(Decision("done"))
            await agent.wait(receipt.run_id, timeout=1)

    async def test_multibyte_long_message_is_fully_presented_to_summarizer(self):
        context = Context()
        context.append("tool", "资料🙂" * 2000, operation_id="op")
        fragments = []

        async def summarize(previous, fragment, max_bytes):
            fragments.append(fragment)
            return Summary("tool op supplied data")

        update = await compact(context.view(), SMALL, [], [], summarize,
                               budget=SMALL.input_bytes, max_calls=30)
        pieces = [json.loads(fragment) for fragment in fragments]
        self.assertGreater(len(pieces), 1)
        offset = 0
        for piece in pieces:
            self.assertEqual((piece["id"], piece["source"], piece["offset"]), (1, "tool", offset))
            offset += len(piece["fragment"])
        recovered = json.loads("".join(piece["fragment"] for piece in pieces))
        self.assertEqual(recovered, context.messages[0])
        context.apply(update)
        self.assertEqual(context.messages, [])
        self.assertEqual(context.through, 1)
        self.assertLess(json_bytes(context.summary), SMALL.summary_bytes)

    async def test_summary_fragments_include_json_escaping_in_input_budget(self):
        context = Context()
        context.append("tool", '\\\n"🙂' * 1000)
        fragments = []

        async def summarize(previous, fragment, max_bytes):
            request_size = SMALL.overhead_bytes + 256 + json_bytes({
                "previous_summary": previous, "next_fragment": fragment})
            self.assertLessEqual(request_size, SMALL.input_bytes)
            fragments.append(fragment)
            return Summary("facts")

        await compact(context.view(), SMALL, [], [], summarize,
                      budget=SMALL.input_bytes, max_calls=30)
        pieces = [json.loads(fragment) for fragment in fragments]
        self.assertEqual(json.loads("".join(piece["fragment"] for piece in pieces)),
                         context.messages[0])

    async def test_compaction_keeps_normal_messages_and_structured_values_whole(self):
        context = Context()
        context.append("user", {"output_fields": ["project", "choices", "total"],
                                "reference": {"batch": "A-17", "unit": "g"}})
        for index in range(10):
            context.append("tool", {"stage": index, "value": "事实" * 70},
                           operation_id=f"op-{index}")
        fragments = []

        async def summarize(previous, fragment, max_bytes):
            fragments.append(json.loads(fragment))
            return Summary("facts")

        update = await compact(context.view(), SMALL, [], [], summarize,
                               budget=SMALL.input_bytes, max_calls=30)
        self.assertGreater(len(fragments), 1)
        self.assertTrue(all(isinstance(fragment, list) for fragment in fragments))
        self.assertEqual([message for fragment in fragments for message in fragment],
                         [m for m in context.messages if m["id"] <= update["through"]])

    async def test_later_summary_failure_does_not_commit_a_partial_prefix(self):
        context = Context()
        context.append("tool", "资料🙂" * 1000)
        before = context.view()
        calls = 0

        async def summarize(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ModelError("summary unavailable", kind="transient")
            return Summary("first fragment")

        with self.assertRaises(ModelError):
            await compact(context.view(), SMALL, [], [], summarize,
                          budget=SMALL.input_bytes, max_calls=30)
        self.assertEqual(calls, 2)
        self.assertEqual(context.view(), before)

    async def test_small_user_requirements_survive_an_unfaithful_summary(self):
        context = Context()
        context.append("user", "archive " * 700)
        context.append("user", {"output_fields": ["project", "revision", "total"]})
        context.append("tool", "facts " * 500)
        context.append("user", "Revision is now R9; applies from stage 9 only.")
        before = context.view()

        async def summarize(*args):
            return Summary("old revision R0; forgotten output fields")

        update = await compact(context.view(), SMALL, [], [], summarize,
                               budget=SMALL.input_bytes, max_calls=30)
        # Even if new information arrives during summarization, both original
        # instructions and new messages must remain independently readable.
        context.append("user", "Use R10 instead.")
        context.apply(update)
        kept = {m["id"]: m for m in context.messages}
        self.assertNotIn(1, kept)  # A large user document is still compacted.
        self.assertEqual(kept[2], before["messages"][1])
        self.assertEqual(kept[4], before["messages"][3])
        self.assertEqual(kept[5]["content"], "Use R10 instead.")
        self.assertLessEqual(SMALL.size(context.view(), [], []), SMALL.input_bytes)

    async def test_retained_user_records_stay_bounded_when_allowance_shrinks(self):
        context = Context()
        for index in range(12):
            context.append("user", f"Amendment {index}: " + "x" * 120)
            context.append("tool", "data " * 100)

        async def summarize(*args):
            return Summary("facts")

        update = await compact(context.view(), SMALL, [], [], summarize,
                               budget=SMALL.input_bytes, max_calls=30)
        context.apply(update)
        original_through = context.through
        old_ids = set(update["retained"])
        self.assertLessEqual(json_bytes([m for m in context.messages if m["id"] in old_ids]),
                             SMALL.input_bytes // 4)
        reduced = ContextPolicy(input_bytes=1600, summary_bytes=200,
                                overhead_bytes=200, keep_recent=0)
        update = await compact(context.view(), reduced, [], [], summarize,
                               budget=reduced.input_bytes, max_calls=30)
        context.apply(update)
        self.assertGreaterEqual(context.through, original_through)
        self.assertLessEqual(reduced.size(context.view(), [], []), reduced.input_bytes)
        self.assertLessEqual(json_bytes(context.messages), reduced.input_bytes // 4)

    async def test_resummarizing_retained_instructions_never_rewinds_the_cursor(self):
        context = Context(summary="earlier facts", through=100, sequence=102, messages=[
            {"id": 1, "source": "user", "content": "Keep original output keys."},
            {"id": 101, "source": "tool", "content": "new facts"},
            {"id": 102, "source": "assistant", "content": "latest output"},
        ])

        async def summarize(*args):
            return Summary("updated facts")

        update = await compact(context.view(), SMALL, [], [], summarize,
                               budget=SMALL.input_bytes)
        context.apply(update)
        self.assertEqual(context.through, 100)
        self.assertEqual([m["id"] for m in context.messages], [1, 101, 102])

    async def test_failed_compaction_preserves_original_history_and_is_bounded(self):
        async def summarize(*args):
            return Summary("too long " * 100)

        async with Agent(ControlledModel(), context=SMALL, summarizer=summarize,
                         limits=Limits(model_retries=1, retry_delay=0)) as agent:
            receipt = await agent.submit("x" * 5000)
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["recoveries"], 1)
            self.assertEqual(result["summary_calls"], 2)
            self.assertEqual(result["context"]["messages"][0]["content"], "x" * 5000)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def model(self, handle):
        return ChatModel(Settings("fake", "https://model.invalid/v1", "test", "test"),
                         transport=httpx.MockTransport(handle))

    def response(self, content="done", *, finish="stop", calls=None):
        return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": {
            "content": content, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}})

    async def test_transient_retry_uses_new_input_without_repeating_successful_tool(self):
        started, release = asyncio.Event(), asyncio.Event()
        executions, payloads = [], []

        async def handle(request):
            payloads.append(json.loads(request.content))
            if len(payloads) == 1:
                return self.response(calls=[{"function": {"name": "lookup", "arguments": "{}"}}],
                                     content="", finish="tool_calls")
            if len(payloads) == 2:
                started.set()
                await release.wait()
                return httpx.Response(503)
            return self.response("new answer")

        async with Agent(self.model(handle), [Tool("lookup", "", {}, lambda: executions.append(1))],
                         limits=Limits(retry_delay=0)) as agent:
            receipt = await agent.submit("old")
            await asyncio.wait_for(started.wait(), 1)
            await agent.submit("new", run_id=receipt.run_id)
            release.set()
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["output"], "new answer")
            self.assertEqual(executions, [1])
            sent = json.loads(payloads[-1]["messages"][1]["content"])
            self.assertEqual(sent["context"]["messages"][-1]["content"], "new")
            self.assertEqual(result["recoveries"], 1)

    async def test_malformed_arguments_and_truncation_are_repaired_without_partial_tools(self):
        payloads, executions = [], []

        def handle(request):
            payloads.append(json.loads(request.content))
            if len(payloads) == 1:
                return self.response(calls=[{"function": {"name": "lookup", "arguments": "{"}}],
                                     finish="tool_calls")
            if len(payloads) == 2:
                return self.response("partial answer", finish="length")
            return self.response()

        async with Agent(self.model(handle), [Tool("lookup", "", {}, lambda: executions.append(1))],
                         limits=Limits(retry_delay=0)) as agent:
            receipt = await agent.submit("start")
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["output"], "done")
            self.assertEqual(executions, [])
            self.assertEqual(result["recoveries"], 2)
            self.assertGreater(payloads[-1]["max_tokens"], payloads[0]["max_tokens"])
            self.assertEqual(result["usage"]["prompt_tokens"], 30)

    async def test_context_rejection_reduces_allowance_and_compacts_before_retry(self):
        requests = []

        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
            return self.response()

        async def summarize(*args):
            return Summary("user asked to start")

        async with Agent(self.model(handle), context=SMALL, summarizer=summarize,
                         limits=Limits(retry_delay=0)) as agent:
            receipt = await agent.submit("x" * 1600)
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["state"], "succeeded")
            self.assertGreater(result["summary_calls"], 0)
            self.assertIn("user asked", result["context"]["summary"])

    async def test_auth_failure_is_terminal_and_next_turn_is_still_possible(self):
        requests = []

        def handle(request):
            requests.append(1)
            return httpx.Response(401) if len(requests) == 1 else self.response()

        async with Agent(self.model(handle), limits=Limits(retry_delay=0)) as agent:
            first = await agent.submit("start", conversation_id="chat")
            failed = await agent.wait(first.run_id, timeout=1)
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["recoveries"], 0)
            second = await agent.submit("try again", conversation_id="chat")
            self.assertEqual((await agent.wait(second.run_id, timeout=1))["output"], "done")

    async def test_cancel_during_backoff_stops_further_requests(self):
        requests = []

        def handle(request):
            requests.append(1)
            return httpx.Response(429, headers={"retry-after": "30"})

        async with Agent(self.model(handle)) as agent:
            receipt = await agent.submit("start")
            await event_of(agent, receipt.run_id, "model.retrying")
            await agent.cancel(receipt.run_id)
            self.assertEqual((await agent.wait(receipt.run_id, timeout=1))["state"], "cancelled")
            self.assertEqual(requests, [1])

    async def test_invalid_json_and_network_failure_have_safe_retry_categories(self):
        for response in (httpx.Response(200, text="{private gateway detail"), httpx.Response(502)):
            with self.subTest(status=response.status_code):
                model = self.model(lambda request, response=response: response)
                with self.assertRaises(ModelError) as caught:
                    await model.decide({"inputs": [], "pending": []}, [])
                self.assertIn(caught.exception.kind, {"transient", "invalid_response"})
                self.assertNotIn("private", str(caught.exception))

    async def test_empty_and_inconsistent_decisions_are_corrected_internally(self):
        model = ControlledModel()
        async with Agent(model, limits=Limits(retry_delay=0)) as agent:
            receipt = await agent.submit("start")
            _, first = await model.next()
            first.set_result(Decision())
            view, second = await model.next()
            self.assertTrue(view["feedback"])
            second.set_result(Decision(wait=True, calls=(Call("unused"),)))
            _, third = await model.next()
            third.set_result(Decision("done"))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["output"], "done")
            self.assertEqual(result["tool_calls"], 0)
            self.assertEqual(result["recoveries"], 2)

    async def test_bad_response_shapes_and_usage_do_not_escape_recovery(self):
        for choice in (None, [], {"message": [], "finish_reason": "stop"}):
            model = self.model(lambda request, choice=choice: httpx.Response(200, json={
                "choices": [choice], "usage": "invalid"}))
            with self.assertRaises(ModelError) as caught:
                await model.decide({"inputs": [], "pending": []}, [])
            self.assertEqual(caught.exception.kind, "invalid_response")
