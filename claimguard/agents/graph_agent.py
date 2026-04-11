from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent
from claimguard.services.graph_fraud import get_graph_detector

GRAPH_SYSTEM_PROMPT = """You are a network fraud analyst.

You interpret graph risk as probabilistic, not absolute.

You look for:
- indirect connections
- clusters of suspicious behavior
- weak signals that could indicate emerging fraud

You NEVER ignore moderate risk scores — they may indicate hidden fraud networks.

Treat tool output as graph-derived features and probabilities; escalate when indirect or clustered risk persists."""


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
        raw_amt = claim_data.get("amount", claim_data.get("claim_amount", 0.0))
        amount_parse_ok = True
        try:
            float(raw_amt if raw_amt is not None else 0.0)
        except (TypeError, ValueError):
            amount_parse_ok = False

        graph_out = self.analyze_graph_risk(claim_data)
        fraud_probability = float(graph_out["fraud_probability"])
        score = max(0.0, min(100.0, round((1.0 - fraud_probability) * 100.0, 2)))
        if graph_out.get("pattern_detected") == "insufficient_graph_input":
            score = min(score, 55.0)
        if not amount_parse_ok:
            score = min(score, 40.0)
        decision = fraud_probability < 0.5
        moderate_band = 0.12 <= fraud_probability < 0.5
        reasoning_parts = [
            f"Graph pattern={graph_out['pattern_detected']}",
            f"fraud_probability={fraud_probability:.4f}",
            f"risk_nodes={len(graph_out['risk_nodes'])}",
        ]
        if moderate_band:
            reasoning_parts.append(
                "Moderate graph risk: probabilistic signal — review indirect ties, clusters, and risk_nodes."
            )
        reasoning = "; ".join(reasoning_parts)
        details = dict(graph_out)
        details["system_prompt_version"] = "graph_v2_network"
        details["moderate_graph_risk_band"] = moderate_band
        if graph_out.get("pattern_detected") == "insufficient_graph_input":
            details.setdefault("validation_flags", []).append("missing_graph_ids")
        return {
            "agent_name": self.name,
            "decision": decision,
            "score": score,
            "reasoning": reasoning,
            "details": details,
        }
