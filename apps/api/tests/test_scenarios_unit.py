"""Unit journey steps (J-S2..4, and J-S1 ops) run through ops_judge. No consult."""
from __future__ import annotations

import pytest

from app.eval.agents.ops_judge import get_fn_map, judge_ops
from app.eval.load import load_suite
from app.eval.schema import AgentExpect, GoldAgent, GoldScenario, JourneyStep

_AGENTS = {r.id: r for r in load_suite("agents") if isinstance(r, GoldAgent)}
_SCENARIOS = {r.id: r for r in load_suite("scenarios") if isinstance(r, GoldScenario)}


def _resolve_args(step: JourneyStep, prior: dict[str, dict]) -> dict:
    args = dict(step.args)
    src = args.pop("trip_id_from", None)
    if src is not None:
        args["trip_id"] = prior[src]["trip_id"]
    return args


def _inline_expect(step: JourneyStep) -> AgentExpect:
    exp = step.expect
    return AgentExpect(
        ordered_ids_prefix=list(exp.ordered_ids_prefix),
        vehicle_id=exp.vehicle_id,
        status=exp.status,
        has_trip_id=exp.has_trip_id,
        kind=exp.kind,
    )


def _run_unit_steps(scenario: GoldScenario) -> None:
    prior: dict[str, dict] = {}
    fn_map = get_fn_map()
    for step in scenario.steps:
        if step.mode != "unit":
            continue
        args = _resolve_args(step, prior)
        if step.assert_ref:
            ref = _AGENTS[step.assert_ref]
            assert step.call == ref.fn, (step.id, step.call, ref.fn)
            res = judge_ops(step.call, args, ref.expect)
            assert res["status"] == "pass", (scenario.id, step.id, res.get("evidence"))
            ids = (res.get("evidence") or {}).get("actual_ids") or []
            prior[step.id] = {"trip_id": ids[0] if ids else None}
            continue
        res = judge_ops(step.call, args, _inline_expect(step))
        assert res["status"] == "pass", (scenario.id, step.id, res.get("evidence"))
        trip_id = ((res.get("evidence") or {}).get("actual_ids") or [None])[0]
        if step.expect.status_in:
            live = fn_map[step.call](**args)
            assert live.get("status") in step.expect.status_in, (step.id, live.get("status"))
            trip_id = live.get("trip_id") or trip_id
        prior[step.id] = {"trip_id": trip_id}


@pytest.mark.unit
def test_fn_map_includes_pharmacy_dispatch_and_track():
    keys = get_fn_map()
    assert "dispatch" in keys
    assert "track_courier" in keys


@pytest.mark.unit
@pytest.mark.parametrize("sid", ["J-S2-BEDS", "J-S3-PHARM", "J-S4-SOS"])
def test_unit_journeys_via_ops_judge(sid: str):
    _run_unit_steps(_SCENARIOS[sid])


@pytest.mark.unit
def test_js1_ops_steps_skip_consult():
    _run_unit_steps(_SCENARIOS["J-S1-NSTEMI"])
