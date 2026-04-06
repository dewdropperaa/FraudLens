from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base_agent import BaseAgent
from claimguard.services.graph_fraud import get_graph_detector


class GraphAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Graph Agent",
            role="Graph Fraud Detection Specialist",
            goal="Detect fraud through graph relationships between patients, providers, and claims",
        )
        project_root = Path(__file__).resolve().parents[1]
        self._detector = get_graph_detector(project_root)

    def analyze_graph_risk(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._detector.score_claim(claim_data)

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        graph_out = self.analyze_graph_risk(claim_data)
        fraud_probability = float(graph_out["fraud_probability"])
        score = max(0.0, min(100.0, round((1.0 - fraud_probability) * 100.0, 2)))
        decision = fraud_probability < 0.5
        reasoning = (
            f"Graph pattern={graph_out['pattern_detected']}; "
            f"fraud_probability={fraud_probability:.4f}; "
            f"risk_nodes={len(graph_out['risk_nodes'])}"
        )
        return {
            "agent_name": self.name,
            "decision": decision,
            "score": score,
            "reasoning": reasoning,
            "details": graph_out,
        }
