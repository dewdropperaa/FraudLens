from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

StageStatus = Literal["PASS", "FAIL", "SKIPPED"]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    return []


class TraceEngine:
    def __init__(self, claim_id: str) -> None:
        self._trace: Dict[str, Any] = {
            "claim_id": str(claim_id or ""),
            "stages": [],
        }

    def add_stage(self, payload: Dict[str, Any]) -> None:
        stage = {
            "stage": str(payload.get("stage") or "UNKNOWN"),
            "status": str(payload.get("status") or "SKIPPED"),
            "timestamp": payload.get("timestamp") or _utc_iso(),
            "inputs": _safe_dict(payload.get("inputs")),
            "outputs": _safe_dict(payload.get("outputs")),
            "decision_snapshot": str(payload.get("decision_snapshot") or ""),
            "flags": _safe_list(payload.get("flags")),
            "reason": str(payload.get("reason") or ""),
        }
        self._trace["stages"].append(stage)

    def export(self) -> Dict[str, Any]:
        return {
            "claim_id": self._trace.get("claim_id", ""),
            "stages": list(self._trace.get("stages", [])),
        }
