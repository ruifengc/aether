from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import httpx

from aether import Agent
from aether.adapters.openai import OpenAIModel, Settings
from aether.errors import ModelError


class OpenAIModelTests(unittest.IsolatedAsyncioTestCase):
    def settings(self):
        return Settings("test-credential", "https://model.invalid/v1", "test-model", "test", "low")

    async def test_agent_reuses_and_closes_model_connections(self):
        requests = []
        closed = []

        class Transport(httpx.MockTransport):
            async def aclose(self):
                closed.append(True)
                await super().aclose()

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                        "message": {"content": "done"}}]})

        model = OpenAIModel(self.settings(), transport=Transport(handle), timeout=7)
        with patch("httpx.AsyncClient", wraps=httpx.AsyncClient) as clients:
            async with Agent(model) as agent:
                for question in ("first", "second"):
                    receipt = await agent.submit(question)
                    await agent.wait(receipt.run_id, timeout=1)
                await model.summarize("", "fragment", 500)
                self.assertEqual(closed, [])
            self.assertEqual(clients.call_count, 1)
        self.assertEqual(closed, [True])
        await agent.close()
        self.assertEqual(closed, [True])
        with self.assertRaisesRegex(ModelError, "closed"):
            await model.decide({"inputs": [], "pending": []}, [])
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(r.extensions["timeout"]["read"] == 7 for r in requests))

    def make_model(self, *args, **kwargs):
        model = OpenAIModel(*args, **kwargs)
        self.addAsyncCleanup(model.aclose)
        return model

    async def test_shutdown_stops_request_before_closing_connection(self):
        started = asyncio.Event()
        order = []

        async def handle(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                order.append("request stopped")

        class Transport(httpx.MockTransport):
            async def aclose(self):
                order.append("connection closed")
                await super().aclose()

        async with Agent(self.make_model(self.settings(), transport=Transport(handle))) as agent:
            receipt = await agent.submit("start")
            await asyncio.wait_for(started.wait(), 1)
            await agent.close()
            self.assertEqual(agent.query(receipt.run_id)["state"], "cancelled")
        self.assertEqual(order, ["request stopped", "connection closed"])

    async def test_explicit_thinking_switch_controls_decisions_and_summary(self):
        for thinking, effort in ((None, None), (None, "high"), (True, "high"), (False, "high")):
            requests = []

            def handle(request, requests=requests):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                            "message": {"content": "done"}}]})

            model = self.make_model(Settings("test", "https://model.invalid", "test", "deepseek",
                                       effort, thinking), transport=httpx.MockTransport(handle))
            await model.decide({"inputs": [], "pending": []}, [])
            await model.summarize("", "fragment", 500)
            for request in requests:
                if thinking is None:
                    self.assertNotIn("thinking", request)
                else:
                    self.assertEqual(request["thinking"]["type"],
                                     "enabled" if thinking else "disabled")
                self.assertEqual(request.get("reasoning_effort"),
                                 None if thinking is False else effort)


    async def test_transport_timeout_is_classified_without_url_or_private_detail(self):
        def handle(request):
            raise httpx.ReadTimeout("private gateway URL", request=request)

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        with self.assertRaises(ModelError) as caught:
            await model.decide({"inputs": [], "pending": []}, [])
        self.assertEqual(caught.exception.kind, "transient")
        self.assertNotIn("private", str(caught.exception))

    async def test_retry_after_nonfinite_values_cannot_break_retry_scheduling(self):
        for value in ("nan", "inf", "-1", "not a number"):
            model = self.make_model(self.settings(), transport=httpx.MockTransport(
                lambda request, value=value: httpx.Response(429, headers={"retry-after": value})))
            with self.assertRaises(ModelError) as caught:
                await model.decide({"inputs": [], "pending": []}, [])
            self.assertEqual(caught.exception.retry_after, 0)

    async def test_native_calls_use_snapshot_with_distinct_sources(self):
        captured = []

        def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call1", "type": "function", "function": {
                    "name": "lookup", "arguments": '{"id": "A"}'}}],
            }}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}})

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        state = {"inputs": [{"source": "user", "content": "goal"},
                            {"source": "tool", "content": {"value": 1}}],
                 "pending": []}
        definition = {"name": "lookup", "description": "Look up an ID.", "parameters": {
            "type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}
        reply = await model.decide(state, [definition])
        self.assertEqual(captured[0]["tools"][0], {"type": "function", "function": definition})
        self.assertEqual(reply.calls[0].name, "lookup")
        self.assertEqual(reply.calls[0].arguments, {"id": "A"})
        self.assertIsNone(reply.wait_for)
        body = json.loads(captured[0]["messages"][1]["content"])
        self.assertEqual([row["source"] for row in body["inputs"]], ["user", "tool"])
        self.assertNotIn("test-credential", json.dumps(captured))

    async def test_wait_is_a_runtime_decision_not_tool_approval(self):
        def handle(request):
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
                "content": "which city?",
                "tool_calls": [{"function": {"name": "await_input",
                                             "arguments": "{}"}}],
            }}]})

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        reply = await model.decide({"inputs": [], "pending": []}, [])
        self.assertTrue(reply.wait)
        self.assertEqual(reply.content, "which city?")

    async def test_output_with_wait_preserves_body_and_rejects_another_output_argument(self):
        for arguments in ("{}", '{"question":"another output"}'):
            def handle(request, arguments=arguments):
                return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls",
                    "message": {"content": "partial result", "tool_calls": [{"function": {
                        "name": "await_input", "arguments": arguments}}]}}]})

            model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
            if arguments == "{}":
                reply = await model.decide({"inputs": [], "pending": []}, [])
                self.assertTrue(reply.wait)
                self.assertEqual(reply.content, "partial result")
            else:
                with self.assertRaises(ModelError):
                    await model.decide({"inputs": [], "pending": []}, [])

    async def test_wait_can_be_silent_when_only_pending_results_are_needed(self):
        def handle(request):
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
                "tool_calls": [{"function": {"name": "await_input", "arguments": "{}"}}],
            }}]})

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        reply = await model.decide({"inputs": [], "pending": [{"id": "slow"}]}, [])
        self.assertTrue(reply.wait)
        self.assertEqual(reply.content, "")

    async def test_critical_wait_is_decoded_alongside_native_calls_without_tool_argument_changes(self):
        def handle(request):
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
                "content": "submitted", "tool_calls": [
                    {"function": {"name": "await_input", "arguments": '{"wait_for":[0,"old"]}'}},
                    {"function": {"name": "lookup", "arguments": '{"id":"A"}'}}]}}]})

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        reply = await model.decide({"inputs": [], "pending": [{"id": "old"}]}, [])
        self.assertFalse(reply.wait)
        self.assertEqual(reply.wait_for, (0, "old"))
        self.assertEqual(reply.calls[0].arguments, {"id": "A"})
        self.assertEqual(reply.content, "submitted")

    async def test_invalid_wait_shapes_and_duplicate_control_calls_are_rejected(self):
        for arguments, count in ((
                '{"wait_for":true}', 1), ('{"wait_for":[true]}', 1),
                ('{"wait_for":[{}]}', 1), ('{"wait_for":null}', 1), ('{}', 2)):
            def handle(request, arguments=arguments, count=count):
                return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls",
                    "message": {"tool_calls": [{"function": {
                        "name": "await_input", "arguments": arguments}}] * count}}]})

            model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
            with self.assertRaises(ModelError):
                await model.decide({"inputs": [], "pending": []}, [])

    async def test_explicit_empty_wait_for_with_calls_means_nonblocking_not_silent_wait(self):
        def handle(request):
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": '{"id":"A"}'}},
                    {"function": {"name": "await_input", "arguments": '{"wait_for":[]}'}}]}}]})

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        reply = await model.decide({"inputs": [], "pending": []}, [])
        self.assertEqual(reply.wait_for, ())
        self.assertFalse(reply.wait)
        self.assertEqual(reply.calls[0].arguments, {"id": "A"})

    async def test_tool_progress_preserves_history_prefix_and_all_snapshot_values(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                       "message": {"content": "done"}}]})

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        state = {"context": {"summary": "", "through": 0, "messages": [
            {"source": "user", "content": "goal"},
            {"source": "tool", "content": "long stable history " * 1000}]},
                 "pending": [{"id": "operation-one"}], "completed": []}
        before = deepcopy(state)
        await model.decide(state, [])
        state.update(pending=[], completed=[{"id": "operation-one", "state": "completed"}],
                     feedback="use the completed result")
        await model.decide(state, [])
        content = [r["messages"][1]["content"] for r in requests]
        prefix = '{"context": ' + json.dumps(state["context"], ensure_ascii=False)
        self.assertTrue(all(body.startswith(prefix) for body in content))
        self.assertEqual(json.loads(content[0]), before)
        self.assertEqual(json.loads(content[1]), state)

    async def test_tokenhub_affinity_follows_conversation_across_runs_and_not_summaries(self):
        requests = []

        def handle(request):
            requests.append((json.loads(request.content), dict(request.headers)))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                       "message": {"content": "done"}}]})

        settings = Settings("test-credential", "https://model.invalid/v1", "test-model", "tencent")
        model = self.make_model(settings, transport=httpx.MockTransport(handle))
        state = {"context": {"messages": []}, "pending": [],
                 "conversation_id": "conversation-A", "run_id": "run-one"}
        await model.decide(state, [])
        state["run_id"] = "run-two"
        await model.decide(state, [])
        await model.summarize("", "fragment", 100)
        state["conversation_id"] = "conversation-B"
        await model.decide(state, [])
        for index, identity in ((0, "conversation-A"), (1, "conversation-A"),
                                (3, "conversation-B")):
            body, headers = requests[index]
            self.assertEqual(body["prompt_cache_key"], identity)
            self.assertEqual(headers["x-session-id"], identity)
            self.assertNotIn("run_id", body["messages"][1]["content"])
        self.assertNotIn("prompt_cache_key", requests[2][0])
        self.assertNotIn("x-session-id", requests[2][1])

    async def test_other_providers_do_not_receive_tokenhub_routing_fields(self):
        requests = []

        def handle(request):
            requests.append((json.loads(request.content), dict(request.headers)))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                       "message": {"content": "done"}}]})

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        await model.decide({"inputs": [], "pending": [],
                            "conversation_id": "conversation-A"}, [])
        self.assertNotIn("prompt_cache_key", requests[0][0])
        self.assertNotIn("x-session-id", requests[0][1])

    async def test_non_ascii_conversation_ids_have_stable_header_safe_routing(self):
        requests = []

        def handle(request):
            requests.append((json.loads(request.content), dict(request.headers)))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                       "message": {"content": "done"}}]})

        settings = Settings("test-credential", "https://model.invalid/v1", "test-model", "tencent")
        model = self.make_model(settings, transport=httpx.MockTransport(handle))
        for conversation in ("中文会话", "中文会话", "line\nbreak"):
            await model.decide({"inputs": [], "pending": [],
                                "conversation_id": conversation}, [])
        keys = [body["prompt_cache_key"] for body, _ in requests]
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])
        for (_body, headers), key in zip(requests, keys, strict=True):
            self.assertEqual(headers["x-session-id"], key)
            self.assertRegex(key, r"^[a-f0-9]{64}$")

    async def test_error_does_not_expose_gateway_body_or_credentials(self):
        def handle(request):
            return httpx.Response(401, text="private gateway detail test-credential")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            model = self.make_model(self.settings(), transport=httpx.MockTransport(handle), record_path=path)
            with self.assertRaisesRegex(RuntimeError, "model HTTP 401") as error:
                await model.decide({"inputs": [], "pending": []}, [])
            self.assertNotIn("test-credential", str(error.exception))
            self.assertNotIn("test-credential", path.read_text())
            self.assertNotIn("test-credential", repr(self.settings()))


    async def test_control_name_conflict_fails_before_any_http_request(self):
        def handle(request):
            self.fail("configuration conflict must not send an HTTP request")

        model = self.make_model(self.settings(), transport=httpx.MockTransport(handle))
        with self.assertRaisesRegex(ModelError, "reserved") as error:
            await model.decide({"inputs": [], "pending": []}, [{
                "name": "await_input", "description": "", "parameters": {}}])
        self.assertEqual(error.exception.kind, "fatal")
