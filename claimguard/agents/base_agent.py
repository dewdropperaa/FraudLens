from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, goal: str):
        self.name = name
        self.role = role
        self.goal = goal

    @abstractmethod
    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def _ensure_contract(self, result: Dict[str, Any] | None) -> Dict[str, Any]:
        def _error(reason: str) -> Dict[str, Any]:
            return {
                "agent": self.name,
                "status": "ERROR",
                "output": {},
                "score": 0.0,
                "reason": reason,
            }

        if result is None:
            return _error("Agent returned None")
        if not isinstance(result, dict):
            return _error("Agent returned non-dict output")
        if not {"agent", "status", "output", "score", "reason"}.issubset(result.keys()):
            return _error("Agent returned invalid contract")
        payload = dict(result)
        payload["agent"] = str(payload.get("agent") or self.name)
        payload["status"] = str(payload.get("status") or "ERROR").upper()
        payload["output"] = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        payload["score"] = float(payload.get("score") or 0.0)
        payload["reason"] = str(payload.get("reason") or "")
        if payload["status"] not in {"DONE", "ERROR"}:
            return _error("Agent returned invalid status")
        if payload["output"] in ({}, None):
            return _error("EMPTY_OUTPUT")
        if payload["status"] == "DONE" and payload["reason"].strip() == "":
            return _error("Missing reason for DONE status")
        return payload

    def run(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        return self._ensure_contract(self.analyze(blackboard))

    def safe_run(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = self.run(blackboard)
            if result is None:
                return {
                    "agent": self.name,
                    "status": "ERROR",
                    "reason": "Agent returned None",
                    "score": 0,
                    "output": {},
                }
            return self._ensure_contract(result)
        except Exception as e:
            return {
                "agent": self.name,
                "status": "ERROR",
                "reason": str(e),
                "score": 0,
                "output": {},
            }
