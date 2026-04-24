from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from claimguard.agents.base_agent import BaseAgent
from claimguard.agents.llm_consistency import run_agent_consistency_check
from claimguard.agents.memory_utils import process_memory_context
from claimguard.services.graph_fraud import get_graph_detector

GRAPH_SYSTEM_PROMPT = """You are a network fraud analyst.

You interpret graph risk as probabilistic, not absolute.
You MUST base your reasoning on the provided OCR text and verified fields. You MUST produce DIFFERENT outputs for different inputs. Generic responses are forbidden.

You look for:
- indirect connections
- clusters of suspicious behavior
- weak signals that could indicate emerging fraud

You NEVER ignore moderate risk scores — they may indicate hidden fraud networks.

Treat tool output as graph-derived features and probabilities; escalate when indirect or clustered risk persists.

MEMORY AWARENESS:
- When memory_context is present, check if the same hospital or doctor appears in past fraud cases.
- If a provider (hospital/doctor) appears repeatedly in fraud memory: escalate network risk.
- Memory strengthens graph signals — a moderate graph score + fraud in memory = high combined risk.
- Memory is ADVISORY — never override probabilistic graph analysis with memory alone."""


class GraphAgent(BaseAgent):
    system_prompt: str = GRAPH_SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__(
            name="Graph Agent",
            role="Network Fraud Analyst",
            goal="Interpret probabilistic graph risk, clusters, and indirect links — do not dismiss moderate scores",
        )
        project_root = Path(__file__).resolve().parents[1]
        self._detector = get_graph_detector(project_root)

    def analyze_graph_risk(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._detector.score_claim(claim_data)
        except ValueError as exc:
            # Missing/non-empty patient_id, provider_id, or claim_id — cannot place the claim in the graph.
            return {
                "fraud_probability": 0.5,
                "pattern_detected": "insufficient_graph_input",
                "risk_nodes": [],
                "graph_input_error": str(exc),
            }

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        tool_results = self.run_tool_pipeline(
            claim_data,
            {
                "fraud_detector": {
                    "documents": claim_data.get("documents") or [],
                    "document_extractions": claim_data.get("document_extractions") or [],
                    "amount": claim_data.get("amount", 0),
                }
            },
        )
        graph_out = self.analyze_graph_risk(claim_data)
        fraud_probability = float(graph_out.get("fraud_probability", 0.5))
        score = max(0.0, min(100.0, round((1.0 - fraud_probability) * 100.0, 2)))
        reasoning = (
            f"Graph risk uses structured graph features and tool signals; "
            f"fraud_probability={fraud_probability:.4f}"
        )
        llm_fallback = self.should_use_llm_fallback(tool_results)
        if llm_fallback:
            print("[LLM FALLBACK USED] True")
            reasoning, _ = run_agent_consistency_check(
                agent_name=self.name,
                claim_data=claim_data,
                draft_reasoning=reasoning,
            )
        else:
            print("[LLM FALLBACK USED] False")

        payload = {
            "agent_name": self.name,
            "decision": score > 60 and fraud_probability < 0.5,
            "score": score,
            "confidence": round(max(0.2, min(0.95, score / 100.0)), 2),
            "reasoning": reasoning,
            "explanation": reasoning,
            "details": {"graph_output": graph_out, "tool_results": tool_results},
        }
        self.enforce_tool_trace(tool_results, llm_fallback)
        return self.build_agent_result(
            output=payload,
            score=float(payload["score"]),
            reason=reasoning,
            tools_used=list(tool_results.keys()),
            llm_fallback_used=llm_fallback,
        )
