from typing import Dict, Any
from .base_agent import BaseAgent
import statistics


class PatternAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Pattern Agent",
            role="Statistical Fraud Detection Specialist",
            goal="Identify statistical patterns indicative of fraud"
        )

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        patient_id = claim_data.get("patient_id", "")
        amount = claim_data.get("amount", 0)
        history = claim_data.get("history", [])
        insurance = claim_data.get("insurance", "")

        score = 100
        reasoning = []
        details = {}

        if len(patient_id) < 8 or not patient_id.isdigit():
            score -= 30
            reasoning.append("Suspicious patient ID detected")
            details["suspicious_patient_id"] = True

        if len(history) > 0:
            amounts = [h.get("amount", 0) for h in history]
            # Need at least 2 prior amounts to compute spread; require a large spike vs history
            # (mean * 4) so normal follow-up claims (e.g. 2x history) are not penalized.
            if len(amounts) >= 2:
                std_dev = statistics.stdev(amounts)
                mean = statistics.mean(amounts)
                z_score = abs(amount - mean) / std_dev if std_dev > 0 else 0

                if z_score > 2.5 and amount > mean * 4:
                    score -= 35
                    reasoning.append(f"Amount is {z_score:.1f} standard deviations from mean")
                    details["z_score"] = round(z_score, 2)

        if len(history) > 0:
            claim_intervals = []
            for i in range(1, len(history)):
                prev_date = history[i-1].get("date", "")
                curr_date = history[i].get("date", "")
                if prev_date and curr_date:
                    claim_intervals.append(1)

            if len(claim_intervals) > 0 and len(claim_intervals) < 7:
                score -= 20
                reasoning.append("Suspiciously short intervals between claims")
                details["avg_interval_days"] = 7

        if len(history) > 0:
            similar_amounts = sum(1 for h in history if abs(h.get("amount", 0) - amount) < 100)
            if similar_amounts > 2:
                score -= 25
                reasoning.append(f"Multiple claims with similar amounts: {similar_amounts}")
                details["similar_amount_count"] = similar_amounts

        if insurance == "CNSS" and amount > 50000:
            score -= 15
            reasoning.append("Unusually high claim for CNSS")
            details["cnss_high_amount"] = True

        recent_claims = [h for h in history if h.get("recent", False)]
        if len(recent_claims) > 3:
            score -= 50
            reasoning.append(f"Pattern of multiple recent claims detected: {len(recent_claims)}")
            details["recent_claims_pattern"] = len(recent_claims)

        score = max(0, score)
        decision = score >= 50

        return {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": "; ".join(reasoning) if reasoning else "No suspicious patterns detected",
            "details": details
        }
