"""Deterministic L7 ops-agent judge.

Lazy FN_MAP so importing `app.eval` does not load production agents until a case runs.
"""
from __future__ import annotations

from typing import Any, Callable

from ..schema import AgentExpect

_FN_MAP: dict[str, Callable[..., Any]] | None = None


def get_fn_map() -> dict[str, Callable[..., Any]]:
    global _FN_MAP
    if _FN_MAP is None:
        from app.agents.expert import connect, list_experts
        from app.agents.ops import dispatch_transport, list_beds, list_transport, track_transport
        from app.agents.pharmacy import dispatch, search, track_courier
        from app.agents.security import raise_sos

        _FN_MAP = {
            "list_beds": list_beds,
            "list_transport": list_transport,
            "dispatch_transport": dispatch_transport,
            "track_transport": track_transport,
            "search": search,
            "dispatch": dispatch,
            "track_courier": track_courier,
            "list_experts": list_experts,
            "connect": connect,
            "raise_sos": raise_sos,
        }
    return _FN_MAP


def _layer(status: str, evidence: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    if score is None:
        score = 1.0 if status == "pass" else 0.0
    return {
        "layer": "ops",
        "status": status,
        "blocking": True,
        "score": score,
        "metrics": {},
        "evidence": evidence,
        "judge": "ops_judge",
    }


def _sequence_ids(result: Any) -> list[str]:
    if isinstance(result, list):
        ids: list[str] = []
        for item in result:
            if not isinstance(item, dict):
                raise TypeError("expected list of mappings")
            rid = item.get("id")
            ids.append(str(rid) if rid is not None else "")
        return ids
    if isinstance(result, dict):
        vehicle = result.get("vehicle")
        vid = vehicle.get("id") if isinstance(vehicle, dict) else None
        rid = result.get("trip_id") or result.get("id") or vid
        return [str(rid)] if rid else []
    raise TypeError(f"result must be list or dict, got {type(result).__name__}")


def _contact_ids(result: Any) -> list[str]:
    if not isinstance(result, dict):
        raise TypeError("contact_ids requires a dict result")
    contacts = result.get("contacts_notified") or []
    ids: list[str] = []
    for item in contacts:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def _ban_ids(result: Any) -> list[str]:
    if isinstance(result, list):
        return _sequence_ids(result)
    if isinstance(result, dict):
        ids = _contact_ids(result)
        vehicle = result.get("vehicle")
        if isinstance(vehicle, dict) and vehicle.get("id"):
            ids.append(str(vehicle["id"]))
        return ids or _sequence_ids(result)
    raise TypeError(f"result must be list or dict, got {type(result).__name__}")


def _vehicle_id(result: Any) -> str | None:
    if isinstance(result, dict):
        if result.get("vehicle_id"):
            return str(result["vehicle_id"])
        vehicle = result.get("vehicle")
        if isinstance(vehicle, dict) and vehicle.get("id"):
            return str(vehicle["id"])
        return None
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return result[0].get("id")
    raise TypeError(f"vehicle_id requires a dict result, got {type(result).__name__}")


def _require_list(result: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(result, list):
        raise TypeError(f"{field} requires a list result, got {type(result).__name__}")
    if any(not isinstance(item, dict) for item in result):
        raise TypeError(f"{field} requires a list of mappings")
    return result


def judge_ops(call: str, args: dict[str, Any], expect: AgentExpect) -> dict[str, Any]:
    fn_map = get_fn_map()
    if call not in fn_map:
        return _layer("error", {"reason": "unknown_call", "call": call})

    try:
        result = fn_map[call](**args)
    except Exception as exc:
        return _layer(
            "error",
            {
                "reason": "agent_exception",
                "call": call,
                "args": args,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )

    try:
        return _assert_expect(result, expect)
    except TypeError as exc:
        return _layer(
            "error",
            {
                "reason": "type_mismatch",
                "call": call,
                "detail": str(exc),
                "result_type": type(result).__name__,
            },
        )


def _assert_expect(result: Any, expect: AgentExpect) -> dict[str, Any]:
    actual_ids = _sequence_ids(result) if isinstance(result, (list, dict)) else []

    if expect.ordered_ids_prefix:
        prefix = expect.ordered_ids_prefix
        got = actual_ids[: len(prefix)]
        if got != prefix:
            return _layer(
                "fail",
                {
                    "reason": "ordered_ids_prefix",
                    "expected_prefix": prefix,
                    "actual_ids": actual_ids,
                },
            )

    if expect.first_n_accepts_nstemi is not None:
        rows = _require_list(result, "first_n_accepts_nstemi")
        n = expect.first_n_accepts_nstemi
        if len(rows) < n:
            return _layer(
                "fail",
                {
                    "reason": "first_n_accepts_nstemi",
                    "expected_n": n,
                    "actual_len": len(rows),
                    "actual_ids": actual_ids,
                },
            )
        bad = [r.get("id") for r in rows[:n] if not r.get("accepts_nstemi")]
        if bad:
            return _layer(
                "fail",
                {
                    "reason": "first_n_accepts_nstemi",
                    "expected_n": n,
                    "non_nstemi_in_prefix": bad,
                    "actual_ids": actual_ids,
                },
            )

    if expect.never_before_nstemi:
        rows = _require_list(result, "never_before_nstemi")
        n = expect.first_n_accepts_nstemi
        if n is None:
            n = 0
            for row in rows:
                if row.get("accepts_nstemi"):
                    n += 1
                else:
                    break
        head = [r.get("id") for r in rows[:n]]
        hit = [fid for fid in expect.never_before_nstemi if fid in head]
        if hit:
            return _layer(
                "fail",
                {
                    "reason": "never_before_nstemi",
                    "forbidden_in_prefix": hit,
                    "prefix_ids": head,
                    "actual_ids": actual_ids,
                },
            )

    if expect.ids_must_not_contain:
        present = _ban_ids(result)
        hit = [fid for fid in expect.ids_must_not_contain if fid in present]
        if hit:
            return _layer(
                "fail",
                {
                    "reason": "ids_must_not_contain",
                    "forbidden_present": hit,
                    "actual_ids": present,
                },
            )

    if expect.vehicle_id is not None:
        got = _vehicle_id(result)
        if got != expect.vehicle_id:
            return _layer(
                "fail",
                {
                    "reason": "vehicle_id",
                    "expected": expect.vehicle_id,
                    "actual": got,
                },
            )

    if expect.contact_ids:
        got = _contact_ids(result)
        if set(got) != set(expect.contact_ids):
            return _layer(
                "fail",
                {
                    "reason": "contact_ids",
                    "expected": sorted(expect.contact_ids),
                    "actual": sorted(got),
                },
            )

    if expect.status is not None:
        if not isinstance(result, dict):
            raise TypeError("status requires a dict result")
        got = result.get("status")
        if got != expect.status:
            return _layer(
                "fail",
                {
                    "reason": "status",
                    "expected": expect.status,
                    "actual": got,
                },
            )

    if expect.has_trip_id is not None:
        if not isinstance(result, dict):
            raise TypeError("has_trip_id requires a dict result")
        tid = result.get("trip_id")
        present = bool(tid)
        if present != expect.has_trip_id:
            return _layer(
                "fail",
                {
                    "reason": "has_trip_id",
                    "expected": expect.has_trip_id,
                    "actual_trip_id": tid,
                },
            )

    if expect.kind is not None:
        if not isinstance(result, dict):
            raise TypeError("kind requires a dict result")
        got = result.get("kind")
        if got != expect.kind:
            return _layer(
                "fail",
                {
                    "reason": "kind",
                    "expected": expect.kind,
                    "actual": got,
                },
            )

    evidence: dict[str, Any] = {}
    if actual_ids:
        evidence["actual_ids"] = actual_ids
    return _layer("pass", evidence)
