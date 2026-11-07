from __future__ import annotations

import importlib.util
import json
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx

from aether.chat import Settings
from aether.errors import ModelError

spec = importlib.util.spec_from_file_location(
    "backend_probe", Path(__file__).resolve().parents[1] / "scripts/probe_backend.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_probe_request_limit_blocks_before_sending(self):
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                        "message": {"content": "OK"}}]})

        model = probe.Probe(Settings("test", "https://model.invalid", "test", "test"),
                            transport=httpx.MockTransport(handle), rows=[], label="test",
                            request_limit=1)
        await model.decide({"inputs": [], "pending": []}, [])
        with self.assertRaisesRegex(ModelError, "probe request limit"):
            await model.decide({"inputs": [], "pending": []}, [])
        self.assertEqual(len(requests), 1)

    async def test_native_comparison_replays_actual_tool_results_and_optional_reasoning(self):
        for preserve in (True, False):
            requests = []

            class Model:
                settings = SimpleNamespace(model="test", provider="tencent")
                max_tokens = 4096

                async def _request(self, payload, requests=requests):
                    requests.append(deepcopy(payload))
                    turn = len(requests)
                    if turn <= 2:
                        name = "lookup" if turn == 1 else "verify"
                        args = {"sku": "SKU-A"} if turn == 1 else {"receipt": "RCPT-829"}
                        message = {"content": None, "reasoning_content": "synthetic reasoning",
                                   "tool_calls": [{"id": f"call-{turn}", "type": "function",
                                       "function": {"name": name, "arguments": json.dumps(args)}}]}
                        finish = "tool_calls"
                    else:
                        message, finish = {"content": json.dumps(probe.EXPECTED)}, "stop"
                    return {"choices": [{"message": message, "finish_reason": finish}]}

            result = await probe.native(Model(), preserve=preserve)
            self.assertTrue(result["passed"])
            messages = requests[-1]["messages"]
            self.assertEqual([m["role"] for m in messages],
                             ["system", "user", "assistant", "tool", "assistant", "tool"])
            self.assertEqual("reasoning_content" in messages[2], preserve)
            self.assertEqual(json.loads(messages[3]["content"])["receipt"], "RCPT-829")
            self.assertTrue(json.loads(messages[5]["content"])["valid"])
            self.assertEqual(len({r["prompt_cache_key"] for r in requests}), 1)
