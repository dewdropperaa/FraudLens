from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List

from claimguard.v2.tools import execute_tool as run_registered_tool
from claimguard.v2.tools import list_tools as list_registered_tools


logger = logging.getLogger("claimguard.agents.base_agent")


class BaseAgent(ABC):
    _observability: Dict[str, int] = {
        "runs": 0,
        "tool_usage": 0,
        "llm_fallback": 0,
        "tool_failures": 0,
    }

    def __init__(self, name: str, role: str, goal: str):
        self.name = name
        self.role = role
        self.goal = goal
        self.tools = []

    @abstractmethod
    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def _build_result(  # SCORE-FIX
        self,
        status: str,
        score: float,
        reason: str,
        output: Dict[str, Any],
        flags: List[str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "agent": self.__class__.__name__,
            "status": str(status or "ERROR").upper(),
            "score": round(float(score), 2),
            "reason": str(reason or "").strip(),
            "output": output or {},
            "flags": flags or [],
        }

    def _ensure_contract(self, result: Dict[str, Any] | None) -> Dict[str, Any]:
        def _error(reason: str) -> Dict[str, Any]:
            return self._build_result(  # SCORE-FIX
                status="ERROR",
                score=0.0,
                reason=reason,
                output={},
                flags=["CONTRACT_ERROR"],
            )

        if result is None:
            return _error("Agent returned None")
        if not isinstance(result, dict):
            return _error("Agent returned non-dict output")
        required = {"agent", "status", "output", "score", "reason", "flags"}  # SCORE-FIX
        if not required.issubset(result.keys()):
            return _error("Agent returned invalid contract")
        payload = dict(result)
        payload["agent"] = str(payload.get("agent") or self.__class__.__name__)  # SCORE-FIX
        payload["status"] = str(payload.get("status") or "ERROR").upper()
        payload["output"] = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        payload["score"] = float(payload.get("score") or 0.0)
        payload["reason"] = str(payload.get("reason") or "")
        payload["flags"] = payload.get("flags") if isinstance(payload.get("flags"), list) else []  # SCORE-FIX
        if payload["status"] not in {"DONE", "ERROR", "TIMEOUT"}:  # SCORE-FIX
            return _error("Agent returned invalid status")
        if payload["status"] == "DONE" and payload["reason"].strip() == "":
            return _error("Missing reason for DONE status")
        return payload

    def _require_llm_response_bundle(self, llm_result: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(llm_result, dict):
            raise RuntimeError("LLM_RESPONSE_LOST")
        raw_response = llm_result.get("response")
        if raw_response is None or not str(raw_response).strip():
            raise RuntimeError("LLM_RESPONSE_LOST")
        if "parsed" not in llm_result:
            raise RuntimeError("LLM_RESPONSE_LOST")
        print("[LLM OUTPUT USED]")
        return llm_result

    def execute_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[TOOL SELECTED] {tool_name}")
        available = set(list_registered_tools())
        if tool_name not in available:
            return {
                "tool": tool_name,
                "status": "ERROR",
                "output": {},
                "confidence": 0.0,
                "reason": f"Tool '{tool_name}' is not registered",
            }
        result = run_registered_tool(tool_name, input_data)
        print(f"[TOOL EXECUTED] {tool_name}")
        return result

    def run_tool_pipeline(self, claim_data: Dict[str, Any], tool_inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        print("[TOOL PIPELINE START]")
        selected = list(tool_inputs.keys())
        print(f"[TOOLS SELECTED] {selected}")
        results: Dict[str, Dict[str, Any]] = {}
        for tool_name, input_data in tool_inputs.items():
            results[tool_name] = self.execute_tool(tool_name, input_data if isinstance(input_data, dict) else {})
        print(f"[TOOLS EXECUTED] {list(results.keys())}")
        for tool_name, result in results.items():
            print(f"[TOOL RESULT USED] {tool_name} status={result.get('status')} confidence={result.get('confidence')}")
            BaseAgent._observability["tool_usage"] += 1
            if str(result.get("status") or "ERROR").upper() != "DONE":
                BaseAgent._observability["tool_failures"] += 1
        return results

    @staticmethod
    def should_use_llm_fallback(tool_results: Dict[str, Dict[str, Any]], threshold: float = 0.65) -> bool:
        for result in tool_results.values():
            status = str(result.get("status") or "ERROR").upper()
            confidence = float(result.get("confidence") or 0.0)
            if status != "DONE" or confidence < threshold:
                return True
        return False

    def enforce_tool_trace(self, tool_results: Dict[str, Dict[str, Any]], llm_fallback_used: bool) -> None:
        if not tool_results:
            raise RuntimeError("TOOL_TRACE_MISSING")
        print(f"[LLM FALLBACK USED] {llm_fallback_used}")
        if llm_fallback_used:
            BaseAgent._observability["llm_fallback"] += 1
        print("[AGENT MIGRATION STATUS] tool_first_enforced")
        runs = max(1, BaseAgent._observability["runs"])
        print(f"[TOOL USAGE RATE] {BaseAgent._observability['tool_usage'] / runs:.2f}")
        print(f"[LLM FALLBACK RATE] {BaseAgent._observability['llm_fallback'] / runs:.2f}")
        print(f"[TOOL FAILURE RATE] {BaseAgent._observability['tool_failures'] / max(1, BaseAgent._observability['tool_usage']):.2f}")

    def build_agent_result(
        self,
        *,
        output: Dict[str, Any],
        score: float,
        reason: str,
        tools_used: List[str],
        llm_fallback_used: bool,
        status: str = "DONE",
        flags: List[str] | None = None,
    ) -> Dict[str, Any]:
        return self._ensure_contract(  # SCORE-FIX
            self._build_result(
                status=status,
                score=float(score),
                reason=reason,
                output=output,
                flags=flags or [],
            )
        )

    def run(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        return self._ensure_contract(self.analyze(blackboard))

    def safe_run(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        try:
            BaseAgent._observability["runs"] += 1
            result = self.run(blackboard)
            if not isinstance(result, dict):  # SCORE-FIX
                raise ValueError("Agent returned non-dict")
            if "score" not in result or result["score"] is None:  # SCORE-FIX
                raise ValueError("Agent returned no score")
            if float(result.get("score", 0.0)) == 0.0 and str(result.get("status", "")).upper() == "DONE":  # SCORE-FIX
                logger.warning(
                    "[AGENT SCORE WARNING] %s returned score=0 with status=DONE — check scoring logic",
                    self.__class__.__name__,
                )
            return self._ensure_contract(result)
        except Exception as e:
            logger.error("[AGENT FAIL] %s crashed: %s", self.__class__.__name__, e)  # SCORE-FIX
            return self._build_result(  # SCORE-FIX
                status="ERROR",
                score=0.0,
                reason=f"Agent crashed: {str(e)}",
                output={},
                flags=["AGENT_EXCEPTION"],
            )
