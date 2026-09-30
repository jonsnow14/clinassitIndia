"""Tagged-union gold rows and scorecard types for the eval harness.

Production `app.main` must not import this package.
"""
from typing import Literal

from pydantic import BaseModel, model_validator

Suite = Literal["retrieval", "clinical", "agents", "suggested", "scenarios", "adversarial"]
Gate = Literal["enforce", "track", "skip"]


class RetrievalExpect(BaseModel):
    pdfs: list[str]
    stw_title_substrings: list[str] = []
    must_contain: list[str] = []
    must_contain_any_dose: list[str] = []
    empty_ok: bool = False
    cardiac_hint_fired: bool | None = None


class GoldRetrieval(BaseModel):
    id: str
    schema_version: str = "1.0.0"
    suite: Literal["retrieval"]
    query_source: Literal["retrieve_raw", "consult_fusion"]
    query: str | None = None
    queries: list[str] | None = None
    k: int = 6
    in_coverage: bool
    gate: Gate
    expect: RetrievalExpect
    notes: str = ""

    @model_validator(mode="after")
    def query_xor_queries(self):
        if self.query_source == "retrieve_raw":
            if self.query is None or self.query == "" or self.queries is not None:
                raise ValueError("retrieve_raw requires query and forbids queries")
        if self.query_source == "consult_fusion":
            if self.query is not None or not self.queries:
                raise ValueError("consult_fusion requires queries and forbids query")
        return self


class ExtractExpect(BaseModel):
    """L2 gold field set without english_query; that lives on ClinicalExpect.english_query_must."""
    age: int | None = None
    sex: str | None = None
    symptoms: list[str] = []
    duration: str | None = None
    vitals: dict[str, str] = {}
    labs: dict[str, str] = {}
    comorbidities: list[str] = []


class ClinicalExpect(BaseModel):
    diagnosis_family: list[str] = []
    icd_family: list[str] = []
    min_urgency: Literal["urgent", "priority", "routine"] | None = None
    referral_required: bool | None = None
    english_query_must: list[str] = []
    must_include_steps: list[str] = []
    must_include_donots: list[str] = []
    allowed_doses: list[str] = []
    suggested_must: list[str] = []
    suggested_must_not: list[str] = []
    cardiac_hint_fired: bool | None = None
    ood_not_in_index: bool = False
    ungrounded_forbidden: list[str] = []
    allow_ungrounded_enrichment: bool = False
    require_grounded_donots: bool = False
    expected_rule_fail: list[str] = []
    synonyms: dict[str, list[str]] = {}


class GoldClinical(BaseModel):
    id: str
    schema_version: str = "1.0.0"
    suite: Literal["clinical"]
    safety_critical: bool = False
    synthetic: bool = True
    lang: str = "hi-en"
    worker_note: str
    gate: Gate = "enforce"
    facts: ExtractExpect | None = None
    expect: ClinicalExpect
    source_pdfs: list[str] = []


class GoldAdversarial(BaseModel):
    """Same body as GoldClinical; suite tag is adversarial so adversarial.jsonl parses."""
    id: str
    schema_version: str = "1.0.0"
    suite: Literal["adversarial"]
    safety_critical: bool = True
    synthetic: bool = True
    lang: str = "hi-en"
    worker_note: str
    gate: Gate = "track"
    facts: ExtractExpect | None = None
    expect: ClinicalExpect
    source_pdfs: list[str] = []


class AgentExpect(BaseModel):
    ordered_ids_prefix: list[str] = []
    first_n_accepts_nstemi: int | None = None
    never_before_nstemi: list[str] = []
    vehicle_id: str | None = None
    ids_must_not_contain: list[str] = []
    contact_ids: list[str] = []
    status: str | None = None
    has_trip_id: bool | None = None
    kind: str | None = None


class GoldAgent(BaseModel):
    id: str
    schema_version: str = "1.0.0"
    suite: Literal["agents"]
    fn: Literal[
        "list_beds", "list_transport", "dispatch_transport", "track_transport",
        "search", "dispatch", "track_courier",
        "list_experts", "connect", "raise_sos",
    ]
    args: dict = {}
    expect: AgentExpect


class SuggestedExpect(BaseModel):
    suggested_actions_exact: list[Literal["beds", "transport", "pharmacy", "expert", "sos"]]


class GoldSuggested(BaseModel):
    id: str
    schema_version: str = "1.0.0"
    suite: Literal["suggested"]
    card: dict
    expect: SuggestedExpect


class JourneyStepExpect(BaseModel):
    status: str | None = None
    status_in: list[str] = []
    has_trip_id: bool | None = None
    kind: str | None = None
    vehicle_id: str | None = None
    ordered_ids_prefix: list[str] = []


class JourneyStep(BaseModel):
    id: str
    call: str
    args: dict = {}
    mode: Literal["unit", "replay", "live"] = "unit"
    optional: bool = False
    assert_ref: str | None = None
    expect: JourneyStepExpect = JourneyStepExpect()


class GoldScenario(BaseModel):
    id: str
    schema_version: str = "1.0.0"
    suite: Literal["scenarios"]
    steps: list[JourneyStep]


GoldRow = GoldRetrieval | GoldClinical | GoldAdversarial | GoldAgent | GoldSuggested | GoldScenario


class LayerResult(BaseModel):
    layer: str
    status: Literal["pass", "fail", "skip", "error"]
    blocking: bool
    score: float | None = None
    metrics: dict = {}
    evidence: dict = {}
    judge: str | None = None


class CaseScorecard(BaseModel):
    case_id: str
    layers: list[LayerResult]
    blocking_fail: bool
    latency_ms: int | None = None
