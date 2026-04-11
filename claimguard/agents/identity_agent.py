from typing import Any, Dict

from .base_agent import BaseAgent

IDENTITY_SYSTEM_PROMPT = """You are an identity verification specialist.

You assume identity fields can be spoofed or reused.

You look for:
- inconsistencies across history
- repeated identifiers across unrelated claims
- subtle variations in identity data

You NEVER assume identity is valid just because it matches format.

Treat tool output as probabilistic signals — escalate uncertainty when history conflicts or IDs collide."""


class IdentityAgent(BaseAgent):
    system_prompt: str = IDENTITY_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(
            name="Identity Agent",
            role="Identity Verification Specialist",
            goal="Detect spoofed or reused identity signals and cross-record inconsistency using tool output only",
        )

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        raw_pid = claim_data.get("patient_id", "")
        patient_id = "" if raw_pid is None else str(raw_pid)
        history = claim_data.get("history", [])
        if not isinstance(history, list):
            history = []

        score = 100
        reasoning = []
        details: Dict[str, Any] = {"system_prompt_version": "identity_v2_forensic"}

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
            "details": details,
        }
