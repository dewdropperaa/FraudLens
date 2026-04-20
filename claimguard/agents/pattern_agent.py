from __future__ import annotations

import statistics
from datetime import date
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from .memory_utils import process_memory_context

PATTERN_SYSTEM_PROMPT = """You are a fraud pattern analyst.

You detect repetition, timing patterns, and behavioral signatures.

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
        raw_pid = claim_data.get("patient_id", "")
        patient_id = "" if raw_pid is None else str(raw_pid)
        try:
            amount = float(claim_data.get("amount", 0) or 0)
        except (TypeError, ValueError):
            amount = 0.0
        history = claim_data.get("history", [])
        if not isinstance(history, list):
            history = []
        insurance = claim_data.get("insurance", "")

        doc_count = max(
            len(claim_data.get("documents") or []),
            len(claim_data.get("document_extractions") or []),
        )

        score = 100
        reasoning: List[str] = []
        details: Dict[str, Any] = {}

        if len(history) == 0:
            score -= 15
            reasoning.append(
                "No claim history available — pattern analysis requires prior data points"
            )
            details["no_history_for_pattern"] = True
            if doc_count < 2:
                score -= 10
                reasoning.append(
                    "Insufficient data: no history and minimal documents to establish behavioral baseline"
                )
                details["insufficient_baseline_data"] = True

        if len(history) > 0:
            amounts = [float(h.get("amount", 0) or 0) for h in history]
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

            # Near-identical historical amounts (scripted / templated billing).
            if len(amounts) >= 3:
                mean_amt = statistics.mean(amounts)
                if mean_amt > 0:
                    var = sum((x - mean_amt) ** 2 for x in amounts) / len(amounts)
                    std_m = var**0.5
                    cv = std_m / mean_amt
                    if cv < 0.06:
                        score -= 22
                        reasoning.append(
                            "Historical claim amounts are unnaturally uniform — possible templated pattern"
                        )
                        details["amount_uniformity_cv"] = round(cv, 4)

        # Parse claim dates and measure spacing regularity (evenly spaced = automation signal).
        parsed_dates = sorted(
            d
            for d in (_parse_history_date(h) for h in history if isinstance(h, dict))
            if d is not None
        )
        if len(parsed_dates) >= 3:
            gaps = [(parsed_dates[i] - parsed_dates[i - 1]).days for i in range(1, len(parsed_dates))]
            gaps = [g for g in gaps if g > 0]
            if len(gaps) >= 2:
                mg = statistics.mean(gaps)
                if mg >= 1:
                    sg = statistics.stdev(gaps) if len(gaps) >= 2 else 0.0
                    rel = sg / mg if mg else 1.0
                    details["claim_spacing_days_mean"] = round(mg, 2)
                    details["claim_spacing_days_std"] = round(sg, 2)
                    if rel < 0.12:
                        score -= 28
                        reasoning.append("Claim dates are evenly spaced — possible automated or scripted cadence")
                        details["suspicious_timing_regularity"] = round(rel, 4)

        if len(history) > 0:
            claim_intervals: List[int] = []
            for i in range(1, len(history)):
                prev_date = history[i - 1].get("date", "") if isinstance(history[i - 1], dict) else ""
                curr_date = history[i].get("date", "") if isinstance(history[i], dict) else ""
                if prev_date and curr_date:
                    try:
                        d1 = date.fromisoformat(str(prev_date))
                        d2 = date.fromisoformat(str(curr_date))
                        interval = abs((d2 - d1).days)
                        claim_intervals.append(interval)
                    except ValueError:
                        claim_intervals.append(0)

            short_intervals = [d for d in claim_intervals if d < 14]
            if len(short_intervals) > 0 and len(short_intervals) >= len(claim_intervals) // 2 + 1:
                score -= 20
                reasoning.append("Suspiciously short intervals between claims")
                details["avg_interval_days"] = round(sum(short_intervals) / len(short_intervals), 1)
            elif claim_intervals:
                details["avg_interval_days"] = round(sum(claim_intervals) / len(claim_intervals), 1)

        if len(history) > 0:
            similar_amounts = sum(1 for h in history if abs(float(h.get("amount", 0) or 0) - amount) < 100)
            if similar_amounts > 2:
                score -= 25
                reasoning.append(f"Multiple claims with similar amounts: {similar_amounts}")
                details["similar_amount_count"] = similar_amounts

        if insurance == "CNSS" and amount > 50000:
            score -= 15
            reasoning.append("Unusually high claim for CNSS")
            details["cnss_high_amount"] = True

        recent_claims = [h for h in history if isinstance(h, dict) and h.get("recent", False)]
        if len(recent_claims) > 3:
            score -= 50
            reasoning.append(f"Pattern of multiple recent claims detected: {len(recent_claims)}")
            details["recent_claims_pattern"] = len(recent_claims)

        score = max(0, score)
        decision = score > 60

        details["system_prompt_version"] = "pattern_v2_forensic"

        default_msg = (
            "No suspicious patterns detected"
            if len(history) > 0
            else "Insufficient data to perform pattern analysis"
        )

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

        return {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": "; ".join(reasoning) if reasoning else default_msg,
            "details": details,
            "memory_insights": memory_insights,
        }
