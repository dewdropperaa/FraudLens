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
    def __init__(self, request_payload: Dict[str, Any], routing: RoutingDecision) -> None:
        self._state: Dict[str, Any] = {
            "request": request_payload,
            "routing_decision": routing.model_dump(),
            "entries": {},
            "memory_context": [],
        }

    def inject_memory_context(self, similar_cases: List[Dict[str, Any]]) -> None:
        """Inject retrieved memory context before agents run.

        Each case dict: claim_id, cin, fraud_label, similarity, summary,
                        hospital, doctor, diagnosis, ts_score, timestamp.
        Only cases above the similarity threshold should be passed in —
        callers (the orchestrator) are responsible for pre-filtering.
        """
        self._state["memory_context"] = list(similar_cases)

    def require(self, required_agent_names: Iterable[str]) -> None:
        missing = [name for name in required_agent_names if name not in self._entries]
        if missing:
            raise BlackboardValidationError(
                f"Missing required prior context from: {', '.join(missing)}"
            )

    def append(self, agent_name: str, *, score: float, confidence: float, explanation: str) -> None:
        entry = BlackboardEntry(score=score, confidence=confidence, explanation=explanation)
        self._entries[agent_name] = entry.model_dump()

    @property
    def _entries(self) -> Dict[str, Dict[str, Any]]:
        return self._state["entries"]

    @property
    def memory_context(self) -> List[Dict[str, Any]]:
        return list(self._state.get("memory_context", []))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._state)
