from typing import Dict, Any
from .base_agent import BaseAgent


class PolicyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Policy Agent",
            role="Insurance Coverage Validation Specialist",
            goal="Validate insurance coverage and policy compliance"
        )

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        insurance = claim_data.get("insurance", "")
        amount = claim_data.get("amount", 0)
        history = claim_data.get("history", [])

        score = 100
        reasoning = []
        details = {}

        if insurance not in ["CNSS", "CNOPS"]:
            score -= 60
            reasoning.append(f"Invalid insurance provider: {insurance}")
            details["insurance_provider"] = insurance
        else:
            details["insurance_provider"] = insurance

        if insurance == "CNSS":
            if amount > 30000:
                score -= 30
                reasoning.append("Amount exceeds CNSS coverage limit")
                details["coverage_limit_exceeded"] = True
                details["cnss_limit"] = 30000
        elif insurance == "CNOPS":
            if amount > 50000:
                score -= 30
                reasoning.append("Amount exceeds CNOPS coverage limit")
                details["coverage_limit_exceeded"] = True
                details["cnops_limit"] = 50000

        if len(history) > 0:
            total_claimed = sum(h.get("amount", 0) for h in history)
            annual_limit = 100000 if insurance == "CNSS" else 150000

            if total_claimed + amount > annual_limit:
                score -= 40
                reasoning.append(f"Total claims exceed annual limit of {annual_limit}")
                details["annual_limit_exceeded"] = True
                details["annual_limit"] = annual_limit
                details["total_claimed"] = total_claimed

        if len(history) > 0:
            recent_rejections = sum(1 for h in history if h.get("decision") == "REJECTED")
            if recent_rejections > 2:
                score -= 25
                reasoning.append(f"Multiple recent rejections: {recent_rejections}")
                details["recent_rejections"] = recent_rejections

        score = max(0, score)
        decision = score >= 50

        return {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": "; ".join(reasoning) if reasoning else "Policy coverage validated",
            "details": details
        }
