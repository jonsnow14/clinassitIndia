import pytest

from app.eval.load import load_manifest, load_suite
from app.eval.schema import GoldAdversarial, GoldClinical, GoldScenario, GoldSuggested

_LANDED = {
    "agents": "n_agents",
    "clinical": "n_clinical",
    "suggested": "n_suggested",
    "scenarios": "n_scenarios",
    "adversarial": "n_adversarial",
}


@pytest.mark.unit
def test_landed_jsonl_counts_match_manifest_census():
    census = load_manifest()["census"]
    for suite, field in _LANDED.items():
        rows = load_suite(suite)
        assert len(rows) == census[field], (suite, len(rows), census[field])


@pytest.mark.unit
def test_clinical_gold_validates_and_snake_is_track():
    rows = load_suite("clinical")
    assert all(isinstance(r, GoldClinical) for r in rows)
    ids = {r.id for r in rows}
    assert ids == {
        "C-GOLDEN-NSTEMI-HIEN",
        "C-GOLDEN-NSTEMI-EN",
        "C-GOLDEN-NSTEMI-HI",
        "C-STEMI-ST",
        "C-HF",
        "C-DM2",
        "C-DKA",
        "C-COUGH",
        "C-SNAKE-OOD",
        "C-EMPTY-VITALS",
        "C-NEG-WELL",
        "C-DM2-ACS",
    }
    snake = next(r for r in rows if r.id == "C-SNAKE-OOD")
    assert snake.gate == "track"
    assert snake.expect.ood_not_in_index is True
    assert snake.expect.expected_rule_fail == ["S-OOD-UNGROUNDED"]
    acs = [r for r in rows if r.id != "C-SNAKE-OOD"]
    assert all(r.gate == "enforce" for r in acs)


@pytest.mark.unit
def test_adversarial_gold_parses():
    rows = load_suite("adversarial")
    assert all(isinstance(r, GoldAdversarial) for r in rows)
    assert {r.id for r in rows} == {
        "X-IGNORE-ICMR",
        "X-MEGA-DOSE",
        "X-NO-REFER",
        "X-ROLEPLAY",
    }
    assert all(r.gate == "track" for r in rows)


@pytest.mark.unit
def test_suggested_and_scenario_types():
    assert all(isinstance(r, GoldSuggested) for r in load_suite("suggested"))
    scenarios = load_suite("scenarios")
    assert all(isinstance(r, GoldScenario) for r in scenarios)
    assert {r.id for r in scenarios} == {
        "J-S1-NSTEMI",
        "J-S2-BEDS",
        "J-S3-PHARM",
        "J-S4-SOS",
    }
    j2 = next(r for r in scenarios if r.id == "J-S2-BEDS")
    assert all(s.mode == "unit" for s in j2.steps)
    j1 = next(r for r in scenarios if r.id == "J-S1-NSTEMI")
    assert j1.steps[0].call == "consult"
    assert j1.steps[0].mode == "replay"
