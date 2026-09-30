"""JSONL gold loader with tagged-union parse and manifest census checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import REPO_ROOT
from .schema import (
    GoldAdversarial,
    GoldAgent,
    GoldClinical,
    GoldRetrieval,
    GoldRow,
    GoldScenario,
    GoldSuggested,
)

GOLD_DIR = REPO_ROOT / "evals" / "gold"
MANIFEST_PATH = GOLD_DIR / "manifest.json"

SUITE_MODELS: dict[str, type] = {
    "retrieval": GoldRetrieval,
    "clinical": GoldClinical,
    "agents": GoldAgent,
    "suggested": GoldSuggested,
    "scenarios": GoldScenario,
    "adversarial": GoldAdversarial,
}

SUITE_FILES: dict[str, str] = {
    "retrieval": "retrieval.jsonl",
    "clinical": "clinical.jsonl",
    "agents": "agents.jsonl",
    "suggested": "suggested.jsonl",
    "scenarios": "scenarios.jsonl",
    "adversarial": "adversarial.jsonl",
}

CENSUS_FIELD: dict[str, str] = {
    "clinical": "n_clinical",
    "agents": "n_agents",
    "suggested": "n_suggested",
    "scenarios": "n_scenarios",
    "adversarial": "n_adversarial",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
    return rows


def parse_gold_row(data: dict[str, Any]) -> GoldRow:
    suite = data.get("suite")
    model = SUITE_MODELS.get(suite) if isinstance(suite, str) else None
    if model is None:
        raise ValueError(f"unknown or missing suite {suite!r}")
    return model.model_validate(data)


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(MANIFEST_PATH)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _census_mismatch(suite: str, rows: list[GoldRow], census: dict[str, Any]) -> str | None:
    if suite == "retrieval":
        n_raw = sum(1 for r in rows if isinstance(r, GoldRetrieval) and r.query_source == "retrieve_raw")
        n_fus = sum(1 for r in rows if isinstance(r, GoldRetrieval) and r.query_source == "consult_fusion")
        expected_raw = census.get("n_retrieval_raw")
        expected_fus = census.get("n_retrieval_fusion")
        if expected_raw is not None and n_raw != expected_raw:
            return f"retrieval.jsonl retrieve_raw={n_raw}, manifest n_retrieval_raw={expected_raw}"
        if expected_fus is not None and n_fus != expected_fus:
            return f"retrieval.jsonl consult_fusion={n_fus}, manifest n_retrieval_fusion={expected_fus}"
        return None
    field = CENSUS_FIELD.get(suite)
    if field is None:
        return None
    expected = census.get(field)
    if expected is not None and len(rows) != expected:
        return f"{SUITE_FILES[suite]} has {len(rows)} rows, manifest {field}={expected}"
    return None


def load_suite(suite: str) -> list[GoldRow]:
    if suite not in SUITE_FILES:
        raise ValueError(f"unknown suite {suite!r}")
    path = GOLD_DIR / SUITE_FILES[suite]
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [parse_gold_row(obj) for obj in load_jsonl(path)]
    census = (load_manifest().get("census") or {}) if MANIFEST_PATH.is_file() else {}
    mismatch = _census_mismatch(suite, rows, census)
    if mismatch:
        raise ValueError(mismatch)
    return rows
