import unittest

from examples.reviewed_research import STAGE, ReviewedModel, run_demo
from tests_helper import ControlledModel

from aether import Agent, Decision, Limits


class ReviewedResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_candidate_can_recheck_in_the_same_budget(self):
        result = await run_demo()
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["output"], "7")
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(result["tool_calls"], 3)
        self.assertEqual([i["name"] for i in result["inputs"] if i["source"] == "tool"],
                         [STAGE.name, "read_inventory", STAGE.name])

    async def test_review_cannot_bypass_the_original_decision_allowance(self):
        result = await run_demo(model_calls=3)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error"], "model_call_limit")
        self.assertEqual(result["model_calls"], 3)
        self.assertIsNone(result["output"])

    async def test_research_and_review_usage_are_both_accounted_without_candidate_output(self):
        research, review = ControlledModel(), ControlledModel()
        async with Agent(ReviewedModel(research, review), [STAGE]) as agent:
            receipt = await agent.submit("research")
            _, reply = await research.next()
            reply.set_result(Decision("candidate", finish_reason="tool_calls",
                                      usage={"prompt_tokens": 5, "completion_tokens": 6}))
            view, reply = await review.next()
            self.assertIsNone(view["output"])
            self.assertEqual(view["usage"], {"prompt_tokens": 5, "completion_tokens": 6,
                                            "known": True})
            reply.set_result(Decision("reviewed", usage={"prompt_tokens": 2, "completion_tokens": 3}))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["output"], "reviewed")
            self.assertEqual(result["usage"], {"prompt_tokens": 7, "completion_tokens": 9,
                                              "known": True})

    async def test_new_input_discards_stale_review_before_final_delivery(self):
        research, review = ControlledModel(), ControlledModel()
        async with Agent(ReviewedModel(research, review), [STAGE],
                         limits=Limits(model_calls=4, tool_calls=2)) as agent:
            receipt = await agent.submit("initial request")
            _, reply = await research.next()
            reply.set_result(Decision("old candidate"))
            view, reply = await review.next()
            self.assertEqual(view["inputs"][-1]["content"]["candidate"], "old candidate")
            await agent.submit("updated requirements", run_id=receipt.run_id)
            reply.set_result(Decision("old approved answer"))
            view, reply = await research.next()
            self.assertEqual(view["discarded_decisions"], 1)
            self.assertIsNone(view["output"])
            reply.set_result(Decision("new candidate"))
            view, reply = await review.next()
            self.assertEqual(view["inputs"][-1]["content"]["candidate"], "new candidate")
            reply.set_result(Decision("new approved answer"))
            result = await agent.wait(receipt.run_id, timeout=1)
            self.assertEqual(result["output"], "new approved answer")
            self.assertEqual(result["model_calls"], 4)
