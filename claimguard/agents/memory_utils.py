"""Shared memory context utilities for legacy deterministic agents.

Each agent calls `process_memory_context()` to:
  1. Analyse retrieved similar cases from the blackboard/claim_data.
  2. Optionally adjust the agent's raw score based on fraud history.
  3. Return a structured `memory_insights` dict and a score delta.

Memory is ADVISORY — it never overrides the agent's own analysis.
If similarity < threshold (already filtered by the retriever) the context
should not have been injected, but agents apply a secondary guard here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

MEMORY_SIMILARITY_THRESHOLD: float = 0.7

# Maximum positive or negative adjustment memory can apply to a score (0-100 scale).
_MAX_MEMORY_PENALTY: float = 20.0
_MAX_MEMORY_BOOST: float = 10.0


def process_memory_context(
    agent_name: str,
    claim_data: Dict[str, Any],
    current_score: float,
    current_cin: str = "",
) -> Tuple[float, Dict[str, Any]]:
    """
    Analyse `memory_context` from `claim_data` and return:
      (adjusted_score, memory_insights_dict)

    `adjusted_score` is on the same 0-100 scale as the calling agent's score.
    """
    memory_context: List[Dict[str, Any]] = claim_data.get("memory_context", [])

    if not memory_context:
        return current_score, _empty_insights()

    # Secondary threshold guard — callers may pass unfiltered context.
    relevant = [
        c for c in memory_context
        if float(c.get("similarity", 0.0)) >= MEMORY_SIMILARITY_THRESHOLD
    ]
    if not relevant:
        return current_score, _empty_insights()

    fraud_labels = {"fraud", "suspicious"}
    fraud_matches = [c for c in relevant if c.get("fraud_label", "").lower() in fraud_labels]
    clean_matches = [c for c in relevant if c.get("fraud_label", "").lower() == "clean"]

    # Identity reuse: same CIN seen in memory cases
    identity_reuse = False
    if current_cin:
        identity_reuse = any(
            c.get("cin", "").strip().upper() == current_cin.strip().upper()
            for c in relevant
        )

    # Score delta computation
    delta: float = 0.0
    impact_parts: List[str] = []
    notes_parts: List[str] = []

    if fraud_matches:
        # Penalty proportional to number of fraud cases and their avg similarity
        avg_sim = sum(float(c.get("similarity", 0.7)) for c in fraud_matches) / len(fraud_matches)
        raw_penalty = min(_MAX_MEMORY_PENALTY, len(fraud_matches) * 8.0 * avg_sim)
        delta -= raw_penalty
        impact_parts.append(
            f"Memory contains {len(fraud_matches)} fraud/suspicious case(s) — "
            f"score reduced by {raw_penalty:.1f}"
        )

    if identity_reuse and fraud_matches:
        extra_penalty = min(10.0, _MAX_MEMORY_PENALTY * 0.5)
        delta -= extra_penalty
        notes_parts.append(
            f"CIN '{current_cin}' was previously associated with fraud — identity reuse flagged"
        )

    if clean_matches and not fraud_matches:
        # Mild confidence boost when memory shows clean history
        avg_sim = sum(float(c.get("similarity", 0.7)) for c in clean_matches) / len(clean_matches)
        boost = min(_MAX_MEMORY_BOOST, len(clean_matches) * 3.0 * avg_sim)
        delta += boost
        impact_parts.append(
            f"Memory contains {len(clean_matches)} clean similar case(s) — "
            f"score boosted by {boost:.1f}"
        )

    # Contradiction detection: if memory says clean but agent score is low
    if clean_matches and current_score < 40:
        notes_parts.append(
            "Memory contradiction: past similar cases were clean but current analysis "
            "shows high risk — contradiction noted, score NOT overridden"
        )

    # Collect recurring providers
    hospitals = {c.get("hospital", "") for c in relevant if c.get("hospital")}
    doctors = {c.get("doctor", "") for c in relevant if c.get("doctor")}
    if hospitals:
        notes_parts.append(f"Recurring hospital(s): {', '.join(sorted(hospitals))}")
    if doctors:
        notes_parts.append(f"Recurring doctor(s): {', '.join(sorted(doctors))}")

    adjusted_score = max(0.0, min(100.0, current_score + delta))

    insights: Dict[str, Any] = {
        "similar_cases_found": len(relevant),
        "fraud_matches": len(fraud_matches),
        "identity_reuse_detected": identity_reuse,
        "impact_on_score": "; ".join(impact_parts) if impact_parts else "No score impact",
        "notes": "; ".join(notes_parts) if notes_parts else "Memory advisory only",
    }
    return adjusted_score, insights


def _empty_insights() -> Dict[str, Any]:
    return {
        "similar_cases_found": 0,
        "fraud_matches": 0,
        "identity_reuse_detected": False,
        "impact_on_score": "No memory context available",
        "notes": "",
    }
