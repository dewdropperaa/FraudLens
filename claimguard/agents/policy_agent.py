from typing import Any, Dict

from .base_agent import BaseAgent

POLICY_SYSTEM_PROMPT = """You are a compliance and fraud risk analyst.

You do not blindly apply policy rules.

You question:
- borderline cases
- repeated near-limit claims
- patterns exploiting policy thresholds

You assume attackers may optimize claims to stay just below limits.

Treat tool output as structured limits and signals — combine with cumulative and marginal abuse patterns."""


class PolicyAgent(BaseAgent):
    system_prompt: str = POLICY_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(
            name="Policy Agent",
            role="Compliance & Policy Risk Analyst",
            goal="Apply CNSS/CNOPS rules while flagging threshold gaming and borderline abuse via tool signals",
        )

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        insurance = claim_data.get("insurance", "")
        try:
            amount = float(claim_data.get("amount", 0) or 0)
        except (TypeError, ValueError):
            amount = 0.0
        history = claim_data.get("history", [])
        if not isinstance(history, list):
            history = []

        score = 100
        reasoning = []
        details: Dict[str, Any] = {"system_prompt_version": "policy_v2_forensic"}

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
            "details": details,
        }
