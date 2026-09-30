import json

import pytest

from app.eval.agents.suggested_judge import judge_suggested
from app.eval.load import GOLD_DIR, load_suite
from app.eval.schema import GoldSuggested
from app.models import ClinicalCard
from app.paths import REPO_ROOT
from app.rag.clinical_agent import DISCLAIMER, _suggested

_SUGGESTED_ROWS = load_suite("suggested")
_FIXTURE = REPO_ROOT / "evals" / "fixtures" / "cards" / "golden_nstemi.json"


@pytest.mark.unit
def test_suggested_gold_ids():
    assert {row.id for row in _SUGGESTED_ROWS} == {
        "SUG-ACS-URGENT-MEDS",
        "SUG-URGENT-NO-MEDS",
        "SUG-ROUTINE",
        "SUG-PRIORITY-REFER",
    }


@pytest.mark.unit
@pytest.mark.parametrize("row", _SUGGESTED_ROWS, ids=[r.id for r in _SUGGESTED_ROWS])
def test_suggested_gold_row(row: GoldSuggested):
    res = judge_suggested(row.card, row.expect)
    assert res["status"] == "pass", res.get("evidence")
    assert isinstance(res["evidence"], dict)


@pytest.mark.unit
def test_sug_acs_urgent_meds_exact_five_action_order():
    row = next(r for r in _SUGGESTED_ROWS if r.id == "SUG-ACS-URGENT-MEDS")
    card = ClinicalCard.model_validate(row.card)
    assert _suggested(card) == ["beds", "transport", "pharmacy", "expert", "sos"]
    assert "sos" in _suggested(card)


@pytest.mark.unit
def test_suggested_always_includes_sos_on_routine():
    row = next(r for r in _SUGGESTED_ROWS if r.id == "SUG-ROUTINE")
    assert _suggested(ClinicalCard.model_validate(row.card)) == ["sos"]


@pytest.mark.unit
def test_golden_nstemi_fixture_is_clinical_card():
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    card = ClinicalCard.model_validate(payload)
    assert card.disclaimer == DISCLAIMER
    assert card.suggested_actions == ["beds", "transport", "pharmacy", "expert", "sos"]
    assert _suggested(card) == list(card.suggested_actions)


@pytest.mark.unit
def test_suggested_jsonl_exists():
    assert (GOLD_DIR / "suggested.jsonl").is_file()
    assert _FIXTURE.is_file()
