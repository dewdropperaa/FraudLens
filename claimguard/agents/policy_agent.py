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
        tool_results = self.run_tool_pipeline(
            claim_data,
            {
                "document_classifier": {
                    "documents": claim_data.get("documents") or [],
                    "document_extractions": claim_data.get("document_extractions") or [],
                },
                "fraud_detector": {
                    "documents": claim_data.get("documents") or [],
                    "document_extractions": claim_data.get("document_extractions") or [],
                    "amount": claim_data.get("amount", 0),
                },
            },
        )
        classifier_out = tool_results["document_classifier"].get("output") or {}
        fraud_out = tool_results["fraud_detector"].get("output") or {}
        missing_docs = len(classifier_out.get("missing_docs") or [])
        risk_indicators = int(fraud_out.get("risk_indicators") or 0)
        score = max(0.0, 100.0 - (missing_docs * 18.0) - (risk_indicators * 12.0))
        decision = score > 60
        reasoning = "Policy decision based on rule and classification tools"

        llm_fallback = self.should_use_llm_fallback(tool_results)
        if llm_fallback:
            print("[LLM FALLBACK USED] True")
            reasoning, _ = run_agent_consistency_check(
                agent_name=self.name,
                claim_data=claim_data,
                draft_reasoning=reasoning,
            )
        else:
            print("[LLM FALLBACK USED] False")

        payload = {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": reasoning,
            "explanation": reasoning,
            "details": {
                "missing_docs": classifier_out.get("missing_docs") or [],
                "risk_indicators": risk_indicators,
                "tool_results": tool_results,
            },
        }
        self.enforce_tool_trace(tool_results, llm_fallback)
        return self.build_agent_result(
            output=payload,
            score=float(payload["score"]),
            reason=reasoning,
            tools_used=list(tool_results.keys()),
            llm_fallback_used=llm_fallback,
        )
