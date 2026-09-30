"""L6 suggested_actions judge for constructed ClinicalCard fixtures.

Unit gold asserts exact order of `_suggested()`; consult-trace gold later uses set membership.
"""
from __future__ import annotations

from typing import Any

from app.models import ClinicalCard
from app.rag.clinical_agent import _suggested

from ..schema import SuggestedExpect


def _layer(status: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer": "suggested",
        "status": status,
        "blocking": True,
        "score": 1.0 if status == "pass" else 0.0,
        "metrics": {},
        "evidence": evidence,
        "judge": "suggested_judge",
    }


def judge_suggested(card: ClinicalCard | dict[str, Any], expect: SuggestedExpect) -> dict[str, Any]:
    try:
        parsed = card if isinstance(card, ClinicalCard) else ClinicalCard.model_validate(card)
    except Exception as exc:
        return _layer(
            "error",
            {
                "reason": "invalid_card",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )
    got = list(_suggested(parsed))
    want = list(expect.suggested_actions_exact)
    if got != want:
        return _layer(
            "fail",
            {
                "reason": "suggested_actions_exact",
                "expected": want,
                "actual": got,
            },
        )
    return _layer("pass", {"actual": got})
