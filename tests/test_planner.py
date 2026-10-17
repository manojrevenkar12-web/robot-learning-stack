import json

import pytest

from rls.agent.planner import LLMPlanner, PlanError, RulePlanner, Step, validate

VISIBLE = {"red", "green", "blue"}


def test_rule_planner_parses_multi_step() -> None:
    plan = RulePlanner().plan("Put the red cube in the bin, then stack the blue cube on the green cube.", VISIBLE)
    assert [s.skill for s in plan] == ["pick_place", "stack"]
    assert plan[1].instruction == "stack the blue cube on the green cube"


@pytest.mark.parametrize(
    "plan,msg",
    [
        ([Step("fly", {"object": "red"})], "unknown skill"),
        ([Step("pick_place", {"object": "purple"})], "no purple cube"),
        ([Step("stack", {"object": "red", "support": "red"})], "itself"),
        ([Step("pick_place", {"object": "red"}), Step("stack", {"object": "blue", "support": "red"})], "already in the bin"),
    ],
)
def test_validator_rejects_bad_plans(plan, msg) -> None:
    with pytest.raises(PlanError, match=msg):
        validate(plan, VISIBLE)


class _Fake:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, system: str, user: str) -> str:
        assert "visible_cubes" in user
        return self.reply


def test_llm_plan_used_when_valid_and_falls_back_when_not() -> None:
    good = json.dumps([{"skill": "stack", "args": {"object": "red", "support": "green"}}])
    p = LLMPlanner(_Fake("Sure! " + good))
    assert p.plan("stack red on green", VISIBLE)[0].args == {"object": "red", "support": "green"} and not p.used_fallback
    bad = json.dumps([{"skill": "stack", "args": {"object": "red", "support": "purple"}}])
    p = LLMPlanner(_Fake(bad))
    plan = p.plan("put the red cube in the bin", VISIBLE)
    assert p.used_fallback and plan[0].skill == "pick_place" and "purple" in (p.last_error or "")
