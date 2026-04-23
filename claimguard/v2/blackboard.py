from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from claimguard.v2.schemas import BlackboardEntry, RoutingDecision


@dataclass(frozen=True)
class AgentContract:
    name: str
    requires: tuple[str, ...]


class BlackboardValidationError(ValueError):
    pass


class SharedBlackboard:
    def __init__(
        self,
        request_payload: Dict[str, Any],
        routing: RoutingDecision,
        *,
        extracted_text: str,
        structured_data: Dict[str, Any],
    ) -> None:
        self._state: Dict[str, Any] = {
            "request": {
                "text": extracted_text,
                "data": dict(structured_data),
            },
            "routing_decision": routing.model_dump(),
            "entries": {},
            "memory_context": [],
            "memory_status": "OK",
            "extracted_text": extracted_text,
            "structured_data": dict(structured_data),
            "verified_structured_data": dict(structured_data),
            "field_verification": [],
            "identity": {
                "cin": "",
                "ipp": "",
                "cin_found": False,
                "ipp_found": False,
                "status": "MISSING",
            },
            "pre_validation": {
                "document_type": "UNKNOWN",
                "injection_detected": False,
                "failed": False,
                "flags": [],
            },
            "security_flags": [],
            "degraded_security_mode": False,
        }

    def inject_memory_context(self, similar_cases: List[Dict[str, Any]]) -> None:
        """Inject retrieved memory context before agents run.

        Each case dict: claim_id, cin, fraud_label, similarity, summary,
                        hospital, doctor, diagnosis, ts_score, timestamp.
        Only cases above the similarity threshold should be passed in —
        callers (the orchestrator) are responsible for pre-filtering.
        """
        self._state["memory_context"] = list(similar_cases)

    def set_memory_status(self, status: str) -> None:
        normalized = str(status or "").strip().upper()
        if normalized not in {"OK", "DEGRADED", "DISABLED"}:
            normalized = "DEGRADED"
        self._state["memory_status"] = normalized

    def require(self, required_agent_names: Iterable[str]) -> None:
        missing = [name for name in required_agent_names if name not in self._entries]
        if missing:
            raise BlackboardValidationError(
                f"Missing required prior context from: {', '.join(missing)}"
            )

    def append(
        self,
        agent_name: str,
        *,
        score: float,
        confidence: float,
        explanation: str,
        claims: List[Dict[str, Any]] | None = None,
        hallucination_flags: List[str] | None = None,
        hallucination_penalty: float = 0.0,
    ) -> None:
        entry = BlackboardEntry(score=score, confidence=confidence, explanation=explanation)
        payload = entry.model_dump()
        payload["claims"] = list(claims or [])
        payload["hallucination_flags"] = list(hallucination_flags or [])
        payload["hallucination_penalty"] = float(hallucination_penalty)
        self._entries[agent_name] = payload

    @property
    def _entries(self) -> Dict[str, Dict[str, Any]]:
        return self._state["entries"]

    @property
    def memory_context(self) -> List[Dict[str, Any]]:
        return list(self._state.get("memory_context", []))

    @property
    def memory_status(self) -> str:
        return str(self._state.get("memory_status", "DEGRADED")).upper()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._state)

    @property
    def extracted_text(self) -> str:
        return str(self._state.get("extracted_text", ""))

    @property
    def structured_data(self) -> Dict[str, Any]:
        return dict(self._state.get("structured_data", {}))

    @property
    def verified_structured_data(self) -> Dict[str, Any]:
        return dict(self._state.get("verified_structured_data", {}))

    @property
    def field_verification(self) -> List[Dict[str, Any]]:
        rows = self._state.get("field_verification", [])
        return list(rows) if isinstance(rows, list) else []

    def set_field_verification(self, rows: List[Dict[str, Any]]) -> None:
        normalized_rows = [dict(item) for item in (rows or []) if isinstance(item, dict)]
        verified: Dict[str, Any] = {}
        for row in normalized_rows:
            if bool(row.get("verified")):
                field = str(row.get("field") or "").strip()
                value = row.get("value")
                if field:
                    verified[field] = value
        self._state["field_verification"] = normalized_rows
        self._state["verified_structured_data"] = verified

    def set_identity_validation(self, payload: Dict[str, Any]) -> None:
        identity = dict(payload or {})
        self._state["identity"] = {
            "cin": str(identity.get("cin") or ""),
            "ipp": str(identity.get("ipp") or ""),
            "cin_found": bool(identity.get("cin_found", False)),
            "ipp_found": bool(identity.get("ipp_found", False)),
            "status": str(identity.get("status") or "MISSING"),
        }

    @property
    def pre_validation(self) -> Dict[str, Any]:
        return dict(self._state.get("pre_validation", {}))

    @property
    def system_flags(self) -> List[str]:
        flags = self._state.get("security_flags", [])
        return [str(flag) for flag in flags] if isinstance(flags, list) else []

    def set_pre_validation(self, payload: Dict[str, Any]) -> None:
        normalized = dict(payload or {})
        self._state["pre_validation"] = normalized
        self._state["security_flags"] = list(normalized.get("security_flags", []))
        self._state["degraded_security_mode"] = bool(normalized.get("degraded_security_mode", False))

    def get_agent_input(self) -> Dict[str, Any]:
        """Single source of truth delivered to every agent."""
        return {
            "text": self.extracted_text,
            "data": self.verified_structured_data,
        }
