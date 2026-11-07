from __future__ import annotations

import unittest

from aether import Agent, Decision


class UsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_details_include_truncated_attempt_and_report_partial_coverage(self):
        responses = iter([
            Decision(usage={"prompt_tokens": 100, "completion_tokens": 20,
                            "prompt_cache_hit_tokens": 80,
                            "prompt_tokens_details": {"cached_tokens": 80},
                            "completion_tokens_details": {"reasoning_tokens": 18}},
                     finish_reason="length"),
            Decision("done", usage={"prompt_tokens": 150, "completion_tokens": 30,
                                    "prompt_cache_hit_tokens": 120}),
        ])

        class Model:
            async def decide(self, snapshot, tools):
                return next(responses)

        async with Agent(Model()) as agent:
            receipt = await agent.submit("test")
            result = await agent.wait(receipt.run_id, timeout=2)
            events = agent.query(receipt.run_id, after=0)["events"]
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["usage"], {
            "prompt_tokens": 250, "completion_tokens": 50, "known": True,
            "prompt_cache_hit_tokens": 200, "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 18}})
        reports = result["usage_reports"]
        self.assertEqual(reports["calls"], 2)
        self.assertEqual(reports["fields"]["prompt_cache_hit_tokens"], 2)
        self.assertEqual(reports["fields"]["completion_tokens_details.reasoning_tokens"], 1)
        completed = [e for e in events if e.type == "model.completed"]
        self.assertEqual([e.data["finish_reason"] for e in completed], ["length", "stop"])
        self.assertTrue(all(e.data["elapsed_s"] >= 0 for e in completed))

    async def test_missing_or_invalid_counts_do_not_become_measured_zeroes(self):
        class Model:
            async def decide(self, snapshot, tools):
                return Decision("done", usage={"prompt_tokens": -1, "completion_tokens": True,
                                               "prompt_cache_hit_tokens": None})

        async with Agent(Model()) as agent:
            receipt = await agent.submit("test")
            result = await agent.wait(receipt.run_id, timeout=2)
        self.assertFalse(result["usage"]["known"])
        self.assertNotIn("prompt_cache_hit_tokens", result["usage"])
        self.assertEqual(result["usage_reports"], {"calls": 1, "fields": {}})
