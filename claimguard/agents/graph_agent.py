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
        memory_status = str(claim_data.get("memory_status", "OK") or "OK").upper()
        raw_amt = claim_data.get("amount", claim_data.get("claim_amount", 0.0))
        amount_parse_ok = True
        try:
            float(raw_amt if raw_amt is not None else 0.0)
        except (TypeError, ValueError):
            amount_parse_ok = False

        history = claim_data.get("history", [])
        if not isinstance(history, list):
            history = []

        graph_out = self.analyze_graph_risk(claim_data)
        fraud_probability = float(graph_out["fraud_probability"])
        pattern = graph_out.get("pattern_detected", "")
        risk_nodes = graph_out.get("risk_nodes", [])
        score = max(0.0, min(100.0, round((1.0 - fraud_probability) * 100.0, 2)))

        if pattern == "insufficient_graph_input":
            score = min(score, 55.0)
        if not amount_parse_ok:
            score = min(score, 40.0)

        no_graph_data = (
            fraud_probability < 0.01
            and len(risk_nodes) <= 3
            and pattern in ("no_graph_pattern", "no_pattern", "")
        )
        if no_graph_data:
            score = min(score, 75.0)

        if len(history) == 0 and no_graph_data:
            score = min(score, 70.0)

        decision = score > 60 and fraud_probability < 0.5
        moderate_band = 0.12 <= fraud_probability < 0.5
        reasoning_parts = [
            f"Graph pattern={pattern}",
            f"fraud_probability={fraud_probability:.4f}",
            f"risk_nodes={len(risk_nodes)}",
        ]
        if no_graph_data:
            reasoning_parts.append(
                "Limited graph data — score capped due to insufficient relationship history"
            )
        if moderate_band:
            reasoning_parts.append(
                "Moderate graph risk: probabilistic signal — review indirect ties, clusters, and risk_nodes."
            )
        reasoning = "; ".join(reasoning_parts)
        details = dict(graph_out)
        details["system_prompt_version"] = "graph_v2_network"
        details["moderate_graph_risk_band"] = moderate_band
        details["no_graph_data"] = no_graph_data
        if pattern == "insufficient_graph_input":
            details.setdefault("validation_flags", []).append("missing_graph_ids")

        # Memory context integration
        memory_adjusted_score, memory_insights = process_memory_context(
            agent_name=self.name,
            claim_data=claim_data,
            current_score=float(score),
            current_cin=str(claim_data.get("patient_id") or ""),
        )
        if memory_adjusted_score != float(score):
            score = max(0.0, min(100.0, memory_adjusted_score))
            decision = score > 60 and fraud_probability < 0.5
        details["memory_insights"] = memory_insights
        if memory_status != "OK":
            score = max(0.0, min(100.0, score - 12.0))
            decision = score > 60 and fraud_probability < 0.5
            reasoning = (
                f"{reasoning}; Memory status is {memory_status}; "
                "reduced confidence in graph/pattern correlation."
            )
        details["memory_status"] = memory_status
        confidence = max(0.2, min(0.95, score / 100.0))

        llm_explanation, llm_meta = run_agent_consistency_check(
            agent_name=self.name,
            claim_data=claim_data,
            draft_reasoning=reasoning,
        )
        details["llm_consistency"] = llm_meta
        return {
            "agent_name": self.name,
            "decision": decision,
            "score": score,
            "confidence": round(confidence, 2),
            "reasoning": llm_explanation,
            "explanation": llm_explanation,
            "details": details,
            "memory_insights": memory_insights,
        }
