from typing import Any, Dict, List

from claimguard.agents.base_agent import BaseAgent
from claimguard.agents.llm_consistency import run_agent_consistency_check
from claimguard.agents.memory_utils import process_memory_context

POLICY_SYSTEM_PROMPT = """You are a compliance and fraud risk analyst.

You do not blindly apply policy rules.
You MUST base your reasoning on the provided OCR text and verified fields. You MUST produce DIFFERENT outputs for different inputs. Generic responses are forbidden.

You question:
- borderline cases
- repeated near-limit claims
- patterns exploiting policy thresholds

You assume attackers may optimize claims to stay just below limits.

Treat tool output as structured limits and signals — combine with cumulative and marginal abuse patterns.

MEMORY AWARENESS:
- When memory_context is present, check if this identity has previously gamed policy thresholds.
- If past cases show repeated near-limit claims from the same CIN: flag systematic policy exploitation.
- Memory is ADVISORY — do not override rule-based compliance checks with memory alone."""

_MEDICAL_EVIDENCE_KEYWORDS: tuple[str, ...] = (
    "medical",
    "médical",
    "clinique",
    "consultation",
    "diagnostic",
    "traitement",
    "ordonnance",
    "prescription",
    "hospitalisation",
    "facture",
    "invoice",
    "honoraires",
)


class PolicyAgent(BaseAgent):
    system_prompt: str = POLICY_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(
            name="Policy Agent",
            role="Compliance & Policy Risk Analyst",
            goal="Apply CNSS/CNOPS rules while flagging threshold gaming and borderline abuse via tool signals",
        )

    @staticmethod
    def _build_corpus(claim_data: Dict[str, Any]) -> str:
        parts: List[str] = []
        for doc in (claim_data.get("documents") or []):
            parts.append(str(doc).lower())
        for ex in (claim_data.get("document_extractions") or []):
            if isinstance(ex, dict):
                parts.append((ex.get("file_name") or "").lower())
                parts.append((ex.get("extracted_text") or "").lower())
        return " ".join(parts)

    @staticmethod
    def _has_medical_evidence(corpus: str) -> bool:
        hits = sum(1 for kw in _MEDICAL_EVIDENCE_KEYWORDS if kw in corpus)
        return hits >= 2

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        insurance = claim_data.get("insurance", "")
        try:
            amount = float(claim_data.get("amount", 0) or 0)
        except (TypeError, ValueError):
            amount = 0.0
        history = claim_data.get("history", [])
        if not isinstance(history, list):
            history = []

        doc_count = max(
            len(claim_data.get("documents") or []),
            len(claim_data.get("document_extractions") or []),
        )
        corpus = self._build_corpus(claim_data)

        score = 100
        reasoning: List[str] = []
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

        if doc_count == 0:
            score -= 20
            reasoning.append(
                "No supporting documents — policy compliance cannot be fully verified"
            )
            details["no_documents_for_policy"] = True
        elif not self._has_medical_evidence(corpus):
            score -= 15
            reasoning.append(
                "Documents do not contain sufficient medical/billing evidence to support the claim"
            )
            details["insufficient_medical_evidence"] = True

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
        decision = score > 60

        # Memory context integration
        memory_adjusted_score, memory_insights = process_memory_context(
            agent_name=self.name,
            claim_data=claim_data,
            current_score=float(score),
            current_cin=str(claim_data.get("patient_id") or ""),
        )
        if memory_adjusted_score != float(score):
            score = max(0, int(memory_adjusted_score))
            decision = score > 60
        details["memory_insights"] = memory_insights

        llm_explanation, llm_meta = run_agent_consistency_check(
            agent_name=self.name,
            claim_data=claim_data,
            draft_reasoning="; ".join(reasoning) if reasoning else "Policy coverage validated",
        )
        details["llm_consistency"] = llm_meta
        payload = {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": llm_explanation,
            "explanation": llm_explanation,
            "details": details,
            "memory_insights": memory_insights,
        }
        return self._ensure_contract(
            {
                "agent": self.name,
                "status": "DONE",
                "output": payload,
                "score": float(payload["score"]),
                "reason": llm_explanation,
            }
        )
