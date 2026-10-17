"""Task planning for multi-step instructions: language -> validated sequence of robot skills.

Two planners share one output format and one validator:

* :class:`RulePlanner` parses the instruction grammar the skills understand
  (deterministic, no network, used in tests and as the fallback);
* :class:`LLMPlanner` asks a language model (Anthropic Messages API) for a plan
  as JSON. Its output is never executed directly: it must pass the same
  validator (known skills, known and visible objects, no object used as both
  target and support), otherwise the rule planner's plan is used.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

SKILLS: dict[str, dict[str, Any]] = {
    "pick_place": {"args": ["object"], "doc": "put the <object> cube in the bin", "template": "put the {object} cube in the bin"},
    "stack": {
        "args": ["object", "support"],
        "doc": "stack the <object> cube on the <support> cube",
        "template": "stack the {object} cube on the {support} cube",
    },
}


@dataclass
class Step:
    skill: str
    args: dict[str, str]

    @property
    def instruction(self) -> str:
        return SKILLS[self.skill]["template"].format(**self.args)


class PlanError(ValueError):
    pass


def validate(plan: list[Step], visible: set[str]) -> list[Step]:
    if not plan:
        raise PlanError("empty plan")
    in_bin: set[str] = set()
    for i, st in enumerate(plan):
        if st.skill not in SKILLS:
            raise PlanError(f"step {i}: unknown skill {st.skill!r}")
        if set(st.args) != set(SKILLS[st.skill]["args"]):
            raise PlanError(f"step {i}: {st.skill} needs args {SKILLS[st.skill]['args']}, got {sorted(st.args)}")
        for v in st.args.values():
            if v not in visible:
                raise PlanError(f"step {i}: no {v} cube in view (visible: {sorted(visible)})")
            if v in in_bin:
                raise PlanError(f"step {i}: the {v} cube is already in the bin")
        if st.skill == "stack" and st.args["object"] == st.args["support"]:
            raise PlanError(f"step {i}: cannot stack a cube on itself")
        if st.skill == "pick_place":
            in_bin.add(st.args["object"])
    return plan


class Planner(Protocol):
    def plan(self, instruction: str, visible: set[str]) -> list[Step]: ...


class RulePlanner:
    PATTERNS = (
        (re.compile(r"(?:put|place|drop)\s+the\s+(\w+)\s+cube\s+(?:in|into)\s+the\s+bin"), lambda m: Step("pick_place", {"object": m[1]})),
        (re.compile(r"stack\s+the\s+(\w+)\s+cube\s+on(?:\s+top\s+of)?\s+the\s+(\w+)\s+cube"), lambda m: Step("stack", {"object": m[1], "support": m[2]})),
    )

    def plan(self, instruction: str, visible: set[str]) -> list[Step]:
        parts = re.split(r",?\s*(?:and\s+)?then\s+|;\s*|\.\s+", instruction.lower().strip().rstrip("."))
        steps = []
        for part in parts:
            for pat, make in self.PATTERNS:
                m = pat.search(part)
                if m:
                    steps.append(make(m))
                    break
            else:
                raise PlanError(f"cannot parse {part!r}")
        return validate(steps, visible)


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Thin wrapper; only used when ANTHROPIC_API_KEY is set."""

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=400, system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


SYSTEM = (
    "You turn a robot task into a JSON list of skill calls. Available skills: "
    + json.dumps({k: {"args": v["args"], "meaning": v["doc"]} for k, v in SKILLS.items()})
    + ". Only use cubes listed as visible. Reply with JSON only, e.g. "
    + '[{"skill": "pick_place", "args": {"object": "red"}}].'
)


class LLMPlanner:
    def __init__(self, client: LLMClient, fallback: Planner | None = None) -> None:
        self.client = client
        self.fallback = fallback or RulePlanner()
        self.last_error: str | None = None
        self.used_fallback = False

    def plan(self, instruction: str, visible: set[str]) -> list[Step]:
        self.used_fallback = False
        try:
            raw = self.client.complete(SYSTEM, json.dumps({"instruction": instruction, "visible_cubes": sorted(visible)}))
            m = re.search(r"\[.*\]", raw, re.S)
            data = json.loads(m.group(0) if m else raw)
            return validate([Step(d["skill"], {k: str(v) for k, v in d["args"].items()}) for d in data], visible)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.used_fallback = True
            return self.fallback.plan(instruction, visible)


def default_planner() -> Planner:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return LLMPlanner(AnthropicClient())
        except Exception:
            pass
    return RulePlanner()


def plan_to_json(plan: list[Step]) -> str:
    return json.dumps([asdict(s) for s in plan])
