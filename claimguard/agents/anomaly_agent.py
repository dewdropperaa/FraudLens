from typing import Dict, Any
from .base_agent import BaseAgent
import random


class AnomalyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Anomaly Agent",
            role="Behavior Analysis Specialist",
            goal="Detect anomalous behavior patterns in claim submissions"
        )

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        patient_id = claim_data.get("patient_id", "")
        amount = claim_data.get("amount", 0)
        history = claim_data.get("history", [])

        score = 100
        reasoning = []
        details = {}

        if len(history) > 0:
            avg_amount = sum(h.get("amount", 0) for h in history) / len(history)
            if amount > avg_amount * 3:
                score -= 30
                reasoning.append(f"Claim amount is {amount/avg_amount:.1f}x higher than average")
                details["amount_ratio"] = round(amount / avg_amount, 2)

            recent_claims = [h for h in history if h.get("recent", False)]
            if len(recent_claims) > 3:
                score -= 60
                reasoning.append(f"High frequency of recent claims: {len(recent_claims)}")
                details["recent_claims_count"] = len(recent_claims)

        if amount > 10000:
            score -= 15
            reasoning.append("High claim amount detected")
            details["high_amount_flag"] = True

        if len(claim_data.get("documents", [])) < 2:
            score -= 10
            reasoning.append("Insufficient documentation")
            details["doc_count"] = len(claim_data.get("documents", []))

        score = max(0, score)
        decision = score >= 50

        return {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": "; ".join(reasoning) if reasoning else "No anomalies detected",
            "details": details
        }
