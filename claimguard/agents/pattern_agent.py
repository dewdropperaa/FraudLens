from __future__ import annotations

import statistics
from datetime import date
from typing import Any, Dict, List, Optional

from claimguard.agents.base_agent import BaseAgent
from claimguard.agents.llm_consistency import run_agent_consistency_check
from claimguard.agents.memory_utils import process_memory_context

PATTERN_SYSTEM_PROMPT = """You are a fraud pattern analyst.

You detect repetition, timing patterns, and behavioral signatures.
You MUST base your reasoning on the provided OCR text and verified fields. You MUST produce DIFFERENT outputs for different inputs. Generic responses are forbidden.

You are skeptical of:
- repeated "clean" claims
- evenly spaced claims (automation signal)
- patterns that look artificially consistent

You assume attackers may try to mimic legitimate patterns.

You highlight ANY statistical irregularity or suspicious regularity.

Treat tool output as raw signals, not final truth — surface weak or ambiguous patterns for review.

MEMORY AWARENESS:
- When memory_context is present, look for recurring patterns across past cases.
- If past fraud cases show the same hospital or doctor: flag systematic provider fraud.
- If the same billing pattern (diagnosis + amount range) recurs: flag templated fraud.
- Memory is ADVISORY — do not override your statistical analysis with memory alone."""


def _parse_history_date(h: Dict[str, Any]) -> Optional[date]:
    raw = h.get("date") or h.get("claim_date")
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            return None
    return None


class PatternAgent(BaseAgent):
    system_prompt: str = PATTERN_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(
            name="Pattern Agent",
            role="Fraud Pattern Analyst",
            goal="Detect repetition, timing regularity, and artificially consistent claim behavior using tool signals",
        )

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        tool_results = self.run_tool_pipeline(
            claim_data,
            {
                "fraud_detector": {
                    "documents": claim_data.get("documents") or [],
                    "document_extractions": claim_data.get("document_extractions") or [],
                    "amount": claim_data.get("amount", 0),
                }
            },
        )
        fraud_out = tool_results["fraud_detector"].get("output") or {}
        score = 100.0  # SCORE-FIX
        flags: List[str] = []  # SCORE-FIX
        memory_status = str(claim_data.get("memory_status") or "").upper()
        history = claim_data.get("history") or []
        if memory_status == "DISABLED" or not history:
            score = 72.0
            reasoning = "History unavailable — pattern analysis limited"
            payload = {
                "agent_name": self.name,
                "status": "REVIEW",
                "decision": True,
                "score": round(score, 2),
                "confidence": 72.0,
                "reasoning": reasoning,
                "explanation": reasoning,
                "signals": ["MEMORY_DISABLED"],
                "data_used": {"fraud_detector": fraud_out},
                "details": {"tool_results": tool_results, "memory_status": memory_status},
            }
            assert payload["score"] is not None
            assert str(payload["explanation"]).strip() != ""
            return self._build_result(status="DONE", score=score, reason=reasoning, output=payload, flags=["MEMORY_DISABLED"])  # SCORE-FIX
        if bool(fraud_out.get("known_fraud_pattern")):
            score -= 40
            flags.append("KNOWN_FRAUD_PATTERN")
        if bool(fraud_out.get("partial_pattern_match")):
            score -= 20
            flags.append("PARTIAL_PATTERN")
        if bool(fraud_out.get("provider_high_risk")):
            score -= 15
            flags.append("PROVIDER_HIGH_RISK")
        if bool(fraud_out.get("patient_frequency_anomaly")):
            score -= 20
            flags.append("PATIENT_FREQUENCY_ANOMALY")
        if not flags:
            score += 5
        score = max(0.0, min(100.0, score))
        decision = score > 60
        reasoning = "Pattern analysis completed using behavioral signals"

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
            "status": "PASS" if score >= 70 else ("REVIEW" if score >= 40 else "FAIL"),
            "decision": decision,
            "score": round(score, 2),
            "confidence": round(max(20.0, min(95.0, score)), 2),
            "reasoning": reasoning,
            "explanation": reasoning,
            "signals": list(flags),
            "data_used": fraud_out,
            "details": {"tool_results": tool_results, "risk_indicators": int(fraud_out.get("risk_indicators") or 0)},
        }
        assert payload["score"] is not None
        assert str(payload["explanation"]).strip() != ""
        self.enforce_tool_trace(tool_results, llm_fallback)
        return self._build_result(  # SCORE-FIX
            status="DONE",
            score=float(payload["score"]),
            reason=reasoning,
            output=payload,
            flags=flags,
        )
