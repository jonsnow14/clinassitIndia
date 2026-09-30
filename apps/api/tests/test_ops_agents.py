import pytest
from pydantic import ValidationError

from app.eval.agents.ops_judge import judge_ops
from app.eval.load import load_manifest, load_suite
from app.eval.run import run_unit_ops
from app.eval.schema import AgentExpect, GoldAgent, GoldRetrieval

_AGENT_ROWS = load_suite("agents")


@pytest.mark.unit
def test_agents_gold_matches_manifest_census():
    census = load_manifest()["census"]
    assert len(_AGENT_ROWS) == census["n_agents"] == 9
    assert {row.id for row in _AGENT_ROWS} == {
        "A-BEDS-ICU",
        "A-BEDS-O2",
        "A-PHARM-DAPT",
        "A-PHARM-ISOSORBIDE",
        "A-TX-O2",
        "A-TX-NO-UNAVAIL",
        "A-EXP-ORDER",
        "A-SOS-SEC",
        "A-SOS-COM",
    }


@pytest.mark.unit
@pytest.mark.parametrize("row", _AGENT_ROWS, ids=[r.id for r in _AGENT_ROWS])
def test_ops_agent_gold_row(row: GoldAgent):
    res = judge_ops(row.fn, row.args, row.expect)
    assert res["status"] == "pass", res.get("evidence")
    assert isinstance(res["evidence"], dict)


@pytest.mark.unit
def test_unit_ops_runner_pass_rate():
    code, scorecard = run_unit_ops()
    assert code == 0
    assert len(scorecard) == 9
    assert all(row["status"] == "pass" for row in scorecard)


@pytest.mark.unit
def test_judge_ops_evaluates_nstemi_fields():
    res = judge_ops(
        "list_beds",
        {"need": "icu"},
        AgentExpect(first_n_accepts_nstemi=99),
    )
    assert res["status"] == "fail"
    assert res["evidence"]["reason"] == "first_n_accepts_nstemi"


@pytest.mark.unit
def test_judge_ops_maps_agent_exception_to_error():
    res = judge_ops("track_transport", {"trip_id": "missing"}, AgentExpect(has_trip_id=True))
    assert res["status"] == "error"
    assert res["evidence"]["reason"] == "agent_exception"
    assert isinstance(res["evidence"], dict)


@pytest.mark.unit
def test_judge_ops_dict_result_type_mismatch_is_error():
    res = judge_ops(
        "dispatch_transport",
        {"kind": "ambulance", "need_oxygen": True},
        AgentExpect(first_n_accepts_nstemi=3),
    )
    assert res["status"] == "error"
    assert res["evidence"]["reason"] == "type_mismatch"


@pytest.mark.unit
def test_retrieve_raw_forbids_queries_even_if_empty():
    with pytest.raises(ValidationError):
        GoldRetrieval.model_validate(
            {
                "id": "R-XOR",
                "suite": "retrieval",
                "query_source": "retrieve_raw",
                "query": "q",
                "queries": [],
                "in_coverage": True,
                "gate": "track",
                "expect": {"pdfs": []},
            }
        )
