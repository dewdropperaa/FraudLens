from typing import Dict, Any
from .base_agent import BaseAgent


class IdentityAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Identity Agent",
            role="Identity Verification Specialist",
            goal="Verify patient identity and detect identity fraud"
        )

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        patient_id = claim_data.get("patient_id", "")
        history = claim_data.get("history", [])

        score = 100
        reasoning = []
        details = {}

        if len(patient_id) < 8:
            score -= 60
            reasoning.append("Patient ID format is invalid")
            details["id_length"] = len(patient_id)

        if not patient_id.isdigit():
            score -= 60
            reasoning.append("Patient ID contains non-numeric characters")
            details["id_numeric"] = False

        if len(history) > 0:
            unique_patients = set(h.get("patient_id", "") for h in history)
            if len(unique_patients) > 1:
                score -= 50
                reasoning.append("Multiple patient IDs detected in history")
                details["unique_patient_ids"] = len(unique_patients)

        if len(history) > 0:
            patient_names = set(h.get("patient_name", "") for h in history if h.get("patient_name"))
            if len(patient_names) > 1:
                score -= 35
                reasoning.append("Inconsistent patient names in history")
                details["unique_names"] = len(patient_names)

        if len(history) > 5:
            score -= 10
            reasoning.append("High claim frequency may indicate identity misuse")
            details["total_claims"] = len(history)

        score = max(0, score)
        decision = score >= 50

        return {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": "; ".join(reasoning) if reasoning else "Identity verified successfully",
            "details": details
        }
