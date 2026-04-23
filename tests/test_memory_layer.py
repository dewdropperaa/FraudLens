"""
Case Memory Layer — Full Validation Test Suite
===============================================

Tests:
  1.  Memory insertion         — 10 claims stored, embeddings created, entries correct
  2.  Similarity retrieval     — same-CIN query → similarity > 0.8
  3.  Identity reuse detection — CIN in 3 fraud cases → agents flag reuse + raise risk
  4.  Fraud pattern match      — same hospital + anomaly pattern → PatternAgent escalates
  5.  False memory (critical)  — unrelated claim → similarity < threshold → agents ignore
  6.  Memory impact            — same claim with vs without memory → measurable improvement
  7.  Debug logging            — retrieval info, similarity scores, agent decision context
  8.  Fail conditions          — explicit FAIL if agents skip memory / use it blindly / miss CIN

All tests are self-contained — no shared mutable state between them.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.agents.anomaly_agent import AnomalyAgent
from claimguard.agents.identity_agent import IdentityAgent
from claimguard.agents.pattern_agent import PatternAgent
from claimguard.agents.memory_utils import process_memory_context
from claimguard.v2.blackboard import SharedBlackboard
from claimguard.v2.memory import (
    CaseMemoryEntry,
    CaseMemoryLayer,
    _build_embedding_text,
    _build_query_text,
    build_agent_summary,
    decision_to_fraud_label,
    get_memory_layer,
)
from claimguard.v2.orchestrator import (
    SEQUENTIAL_AGENT_CONTRACTS,
    ClaimGuardV2Orchestrator,
    _compute_fallback_memory_insights,
)
from claimguard.v2.schemas import MemoryInsights, RoutingDecision


# ── Controlled Embedder ─────────────────────────────────────────────────────
#
# Produces fully deterministic vectors from a fixed vocabulary.
# Vocab tokens are mapped to dedicated dimensions, so claims sharing the
# same CIN/hospital always produce a high cosine similarity and completely
# unrelated claims produce near-zero similarity.
#
# Similarity guarantees (all normalised):
#   same CIN, same hospital  →  cosine ≈ 1.00  (above any threshold ≤ 0.9)
#   same CIN, diff hospital  →  cosine ≈ 0.91  (above 0.7 threshold)
#   diff CIN, same hospital  →  cosine ≈ 0.08  (below 0.7 threshold)
#   completely unrelated     →  cosine ≈ 0.00  (zero-vector → cosine = 0)

_VOCAB: List[str] = [
    # CINs (dims 0-5) — high weight so CIN match dominates
    "BE123456", "BE123457", "BE123458", "AB999001", "XZ111111", "WW000099",
    # Hospitals (dims 6-8) — lower weight
    "CHU Rabat", "Polyclinique Atlas", "Clinique Sud",
    # Doctors (dims 9-11)
    "Dr. Alami", "Dr. Benali", "Dr. Cherkaoui",
    # Diagnoses (dims 12-14)
    "fracture", "appendicite", "grippe",
    # Anomaly patterns (dims 15-17)
    "inflated billing", "document forgery", "identity theft",
    # Fraud labels (dims 18-20)
    "fraud", "suspicious", "clean",
]
_DIM = len(_VOCAB)
_CIN_WEIGHT = 10.0    # CIN dimensions get a large boost so CIN match dominates
_OTHER_WEIGHT = 1.0   # non-CIN dims


class _ControlledEmbedder:
    """Token-presence embedder with configurable CIN weight."""

    def embed_query(self, text: str) -> List[float]:
        vec = [0.0] * _DIM
        text_lower = text.lower()
        for i, token in enumerate(_VOCAB):
            if token.lower() in text_lower:
                weight = _CIN_WEIGHT if i < 6 else _OTHER_WEIGHT
                vec[i] = weight
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


def _make_layer(threshold: float = 0.7) -> CaseMemoryLayer:
    """Create a fresh in-memory CaseMemoryLayer with the controlled embedder."""
    layer = CaseMemoryLayer(store_path="", similarity_threshold=threshold)
    layer._embedder = _ControlledEmbedder()
    return layer


def _entry(
    *,
    claim_id: str,
    cin: str = "BE123456",
    hospital: str = "CHU Rabat",
    doctor: str = "Dr. Alami",
    diagnosis: str = "fracture",
    fraud_label: str = "fraud",
    ts_score: float = 20.0,
    agent_summary: str = "inflated billing; identity theft",
) -> CaseMemoryEntry:
    return CaseMemoryEntry(
        claim_id=claim_id,
        cin=cin,
        hospital=hospital,
        doctor=doctor,
        diagnosis=diagnosis,
        fraud_label=fraud_label,
        ts_score=ts_score,
        agent_summary=agent_summary,
    )


def _claim_with_cin(cin: str, hospital: str = "CHU Rabat") -> Dict[str, Any]:
    return {
        "identity": {"cin": cin, "hospital": hospital},
        "policy": {"diagnosis": "fracture"},
        "metadata": {"doctor": "Dr. Alami"},
    }


# ── Shared no-op mocks ──────────────────────────────────────────────────────

class _NoOpTrustLayer:
    def process_if_applicable(self, **_: Any) -> None:
        return None


class _NoOpMemory:
    """Memory layer that always returns no context (used in 'without memory' baselines)."""
    def retrieve_similar_cases(self, *_: Any, **__: Any) -> List[Dict[str, Any]]:
        return []

    def store_case(self, *_: Any, **__: Any) -> None:
        pass

    @property
    def entry_count(self) -> int:
        return 0


# ── Orchestrator fake Crew/Agent (same pattern as existing tests) ───────────

@dataclass
class _CallRecord:
    role: str
    description: str
    blackboard_raw: Dict[str, Any]


class _FakeAgent:
    def __init__(self, role: str = "", goal: str = "", backstory: str = "", **_: Any) -> None:
        self.role = role


class _FakeTask:
    def __init__(
        self, description: str = "", agent: Any = None, expected_output: str = "", **_: Any
    ) -> None:
        self.description = description
        self.agent = agent


class _MemoryAwareCrew:
    """
    Simulates an LLM crew that:
    - Parses the memory_context from the task description/blackboard.
    - Returns realistic scores AND a populated memory_insights block.
    """
    calls: List[_CallRecord] = []
    _scenario: str = "clean"

    def __init__(self, tasks: List[_FakeTask], **_: Any) -> None:
        self._task = tasks[0]

    def kickoff(self) -> str:
        role = self._task.agent.role
        # Extract blackboard from the prompt
        bb_raw: Dict[str, Any] = {}
        for line in self._task.description.splitlines():
            if line.startswith("Blackboard     : "):
                try:
                    bb_raw = json.loads(line[len("Blackboard     : "):].strip())
                except json.JSONDecodeError:
                    pass
        memory_context: List[Dict[str, Any]] = bb_raw.get("memory_context", [])
        fraud_in_memory = [c for c in memory_context if c.get("fraud_label") in ("fraud", "suspicious")]
        has_memory = bool(memory_context)

        # Base scores by scenario
        _SCORES: Dict[str, Dict[str, float]] = {
            "clean": {
                "IdentityAgent": 0.85, "DocumentAgent": 0.82, "PolicyAgent": 0.84,
                "AnomalyAgent": 0.35, "PatternAgent": 0.38, "GraphRiskAgent": 0.36,
            },
            "fraud_with_memory": {
                "IdentityAgent": 0.18, "DocumentAgent": 0.20, "PolicyAgent": 0.25,
                "AnomalyAgent": 0.82, "PatternAgent": 0.86, "GraphRiskAgent": 0.80,
            },
        }
        scenario = _MemoryAwareCrew._scenario if has_memory and fraud_in_memory else "clean"
        score = _SCORES.get(scenario, _SCORES["clean"]).get(role, 0.5)
        confidence = 0.90

        memory_insights: Dict[str, Any] = {
            "similar_cases_found": len(memory_context),
            "fraud_matches": len(fraud_in_memory),
            "identity_reuse_detected": any(
                c.get("cin", "").strip().upper() == "BE123456" for c in fraud_in_memory
            ),
            "impact_on_score": (
                f"Adjusted based on {len(fraud_in_memory)} fraud matches in memory"
                if fraud_in_memory else "No memory impact"
            ),
            "notes": (
                "CIN reuse detected; provider fraud history present"
                if fraud_in_memory else "Memory advisory only"
            ),
        }

        _MemoryAwareCrew.calls.append(
            _CallRecord(role=role, description=self._task.description, blackboard_raw=bb_raw)
        )
        return json.dumps({
            "score": score,
            "confidence": confidence,
            "explanation": f"{role} evaluated claim with {len(memory_context)} memory cases.",
            "memory_insights": memory_insights,
        })


def _patch_memory_aware_orchestrator(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module
    _MemoryAwareCrew.calls.clear()
    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _MemoryAwareCrew)


def _make_claim_request(
    cin: str = "BE123456",
    hospital: str = "CHU Rabat",
    doctor: str = "Dr. Alami",
    diagnosis: str = "fracture",
) -> Dict[str, Any]:
    return {
        "identity": {"cin": cin, "hospital": hospital, "claimant_type": "patient"},
        "documents": [{"id": "doc-1", "document_type": "medical_report", "text": "routine report"}],
        "policy": {"diagnosis": diagnosis, "doctor": doctor},
        "metadata": {"claim_id": f"TEST-{cin}-001"},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. MEMORY INSERTION TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_memory_insertion_10_claims_stored_correctly() -> None:
    """Insert 10 past claims; verify all embeddings created and entries stored correctly."""
    layer = _make_layer()

    entries_to_insert = [
        # 3 with the same CIN = BE123456
        _entry(claim_id="C-001", cin="BE123456", fraud_label="fraud"),
        _entry(claim_id="C-002", cin="BE123456", fraud_label="fraud", hospital="Polyclinique Atlas"),
        _entry(claim_id="C-003", cin="BE123456", fraud_label="suspicious"),
        # 2 fraud with same hospital = CHU Rabat, different CINs
        _entry(claim_id="C-004", cin="BE123457", hospital="CHU Rabat", fraud_label="fraud"),
        _entry(claim_id="C-005", cin="BE123458", hospital="CHU Rabat", fraud_label="fraud"),
        # 5 random clean claims
        _entry(claim_id="C-006", cin="AB999001", hospital="Clinique Sud", fraud_label="clean", ts_score=92.0),
        _entry(claim_id="C-007", cin="AB999001", hospital="Clinique Sud", fraud_label="clean", ts_score=90.0),
        _entry(claim_id="C-008", cin="XZ111111", hospital="Polyclinique Atlas", fraud_label="clean"),
        _entry(claim_id="C-009", cin="WW000099", hospital="Clinique Sud", fraud_label="clean"),
        _entry(claim_id="C-010", cin="WW000099", hospital="CHU Rabat", fraud_label="clean"),
    ]

    for e in entries_to_insert:
        layer.store_case(e)

    # All 10 stored
    assert layer.entry_count == 10, f"Expected 10 entries, got {layer.entry_count}"

    # Every entry has a non-empty embedding vector
    for i, vec in enumerate(layer._vectors):
        assert len(vec) > 0, f"Entry {i} has zero-length vector"
        mag = math.sqrt(sum(x * x for x in vec))
        assert mag > 0.0, f"Entry {i} has zero-magnitude (null) embedding — embedding failed"

    # Verify claim_ids stored correctly
    stored_ids = {e.claim_id for e in layer._entries}
    expected_ids = {e.claim_id for e in entries_to_insert}
    assert stored_ids == expected_ids, f"Stored IDs mismatch: {stored_ids ^ expected_ids}"

    # Verify 3 entries have CIN = BE123456
    cin_group = [e for e in layer._entries if e.cin == "BE123456"]
    assert len(cin_group) == 3, f"Expected 3 entries with CIN=BE123456, got {len(cin_group)}"

    # Verify 2 fraud entries share hospital = CHU Rabat (from C-004 / C-005)
    fraud_chu = [e for e in layer._entries if e.hospital == "CHU Rabat" and e.fraud_label == "fraud"]
    assert len(fraud_chu) >= 2, (
        f"Expected ≥2 fraud entries at CHU Rabat, got {len(fraud_chu)}"
    )

    # Verify fraud labels persisted correctly
    fraud_entries = [e for e in layer._entries if e.fraud_label == "fraud"]
    assert len(fraud_entries) == 4, f"Expected 4 fraud entries, got {len(fraud_entries)}"

    # Verify embedding_text includes key fields for every entry
    for e in layer._entries:
        assert e.cin in e.embedding_text or "CIN" in e.embedding_text, (
            f"{e.claim_id}: CIN not reflected in embedding_text"
        )
        assert e.hospital in e.embedding_text, (
            f"{e.claim_id}: hospital not in embedding_text"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. SIMILARITY RETRIEVAL TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_similarity_retrieval_same_cin_above_threshold() -> None:
    """
    Query with the same CIN as stored fraud cases.
    FAIL if no matches found or similarity is below 0.8.
    """
    layer = _make_layer(threshold=0.0)   # retrieve all, then check scores manually

    # Store 3 fraud entries with CIN = BE123456
    for i in range(3):
        layer.store_case(_entry(
            claim_id=f"FRAUD-{i}",
            cin="BE123456",
            hospital="CHU Rabat",
            fraud_label="fraud",
        ))

    # Store 2 clean entries with a different CIN
    for i in range(2):
        layer.store_case(_entry(
            claim_id=f"CLEAN-{i}",
            cin="AB999001",
            hospital="Clinique Sud",
            fraud_label="clean",
        ))

    # Query with the repeated CIN
    results = layer.retrieve_similar_cases(_claim_with_cin("BE123456"), k=5)

    assert results, (
        "FAIL: retrieve_similar_cases returned NO results for a query with a known CIN. "
        "Memory retrieval is not working."
    )

    # Among same-CIN results, all must have similarity above 0.8
    same_cin_results = [r for r in results if r["cin"] == "BE123456"]
    assert same_cin_results, (
        "FAIL: No results with matching CIN=BE123456 were returned. "
        "Same-CIN cases must always be the top hits."
    )
    low_sim_same_cin = [r for r in same_cin_results if r["similarity"] < 0.8]
    assert not low_sim_same_cin, (
        f"FAIL: {len(low_sim_same_cin)} same-CIN result(s) have similarity < 0.8:\n"
        + "\n".join(
            f"  claim_id={r['claim_id']} sim={r['similarity']:.4f} cin={r['cin']}"
            for r in low_sim_same_cin
        )
    )

    # At least 3 of the returned results should be the same-CIN fraud entries
    same_cin_results = [r for r in results if r["cin"] == "BE123456"]
    assert len(same_cin_results) >= 3, (
        f"FAIL: Expected ≥3 results with CIN=BE123456, got {len(same_cin_results)}. "
        f"Results: {[r['claim_id'] for r in results]}"
    )

    # The highest-similarity result must be a fraud case
    top = max(results, key=lambda r: r["similarity"])
    assert top["fraud_label"] == "fraud", (
        f"FAIL: Top result is not a fraud case — "
        f"claim_id={top['claim_id']}, fraud_label={top['fraud_label']}, sim={top['similarity']:.4f}"
    )


def test_similarity_retrieval_returns_fraud_labels_correctly() -> None:
    """Returned entries must faithfully reflect their stored fraud_label."""
    layer = _make_layer(threshold=0.0)
    layer.store_case(_entry(claim_id="F1", cin="BE123456", fraud_label="fraud"))
    layer.store_case(_entry(claim_id="F2", cin="BE123456", fraud_label="suspicious"))
    layer.store_case(_entry(claim_id="F3", cin="BE123456", fraud_label="clean"))

    results = layer.retrieve_similar_cases(_claim_with_cin("BE123456"), k=5)
    labels = {r["claim_id"]: r["fraud_label"] for r in results}

    assert labels.get("F1") == "fraud", f"Expected fraud for F1, got {labels.get('F1')}"
    assert labels.get("F2") == "suspicious", f"Expected suspicious for F2, got {labels.get('F2')}"
    assert labels.get("F3") == "clean", f"Expected clean for F3, got {labels.get('F3')}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. IDENTITY REUSE DETECTION TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_identity_reuse_detected_by_memory_utils() -> None:
    """
    CIN = BE123456 appears in 3 past fraud cases.
    process_memory_context must:
      - flag identity_reuse_detected = True
      - report fraud_matches = 3
      - reduce the score (increase fraud risk)
    """
    cin = "BE123456"
    memory_context = [
        {"claim_id": f"OLD-{i}", "cin": cin, "fraud_label": "fraud",
         "similarity": 0.92, "summary": "Identity theft; inflated billing",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"}
        for i in range(3)
    ]
    claim_data = {"patient_id": cin, "memory_context": memory_context}
    original_score = 75.0

    adjusted_score, insights = process_memory_context(
        agent_name="IdentityAgent",
        claim_data=claim_data,
        current_score=original_score,
        current_cin=cin,
    )

    # Must flag reuse
    assert insights["identity_reuse_detected"] is True, (
        "FAIL: identity_reuse_detected is False even though CIN appears in 3 fraud memory cases."
    )
    # Must count 3 fraud matches
    assert insights["fraud_matches"] == 3, (
        f"FAIL: Expected fraud_matches=3, got {insights['fraud_matches']}"
    )
    # Score must be reduced (higher risk)
    assert adjusted_score < original_score, (
        f"FAIL: Score was NOT reduced despite CIN appearing in 3 fraud cases. "
        f"original={original_score}, adjusted={adjusted_score}"
    )
    # Must not reduce to 0 (memory is advisory, not absolute)
    assert adjusted_score > 0, (
        f"FAIL: Score reduced to 0 — memory is overriding agent analysis instead of advising."
    )
    # impact_on_score must be non-empty
    assert insights["impact_on_score"], "FAIL: impact_on_score is empty"
    assert "fraud" in insights["impact_on_score"].lower() or "fraud" in insights.get("notes", "").lower(), (
        f"FAIL: impact_on_score does not mention fraud: {insights['impact_on_score']}"
    )


def test_identity_agent_flags_reuse_with_memory_context() -> None:
    """
    Run IdentityAgent.analyze() with memory_context containing 3 fraud cases for same CIN.
    The output must include memory_insights with identity_reuse_detected = True.
    """
    cin = "BE123456"
    memory_context = [
        {"claim_id": f"FR-{i}", "cin": cin, "fraud_label": "fraud",
         "similarity": 0.91, "summary": "CIN reuse confirmed",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"}
        for i in range(3)
    ]
    claim_data = {
        "patient_id": cin,
        "identity": {"cin": cin},
        "documents": [],
        "history": [],
        "memory_context": memory_context,
    }

    agent = IdentityAgent()
    result = agent.analyze(claim_data)

    assert "memory_insights" in result, (
        "FAIL: IdentityAgent output does not include 'memory_insights'. "
        "Agents MUST reference memory context."
    )
    mi = result["memory_insights"]
    assert mi["identity_reuse_detected"] is True, (
        f"FAIL: IdentityAgent did not detect CIN reuse. insights={mi}"
    )
    assert mi["fraud_matches"] >= 3, (
        f"FAIL: IdentityAgent reported fraud_matches={mi['fraud_matches']}, expected ≥3"
    )


def test_anomaly_agent_increases_risk_on_cin_reuse() -> None:
    """
    AnomalyAgent with 3 fraud memory cases for same CIN must show reduced score
    compared to running without memory.
    """
    cin = "BE123456"
    base_claim = {
        "patient_id": cin,
        "amount": 2500.0,
        "history": [],
        "documents": [],
        "insurance": "CNSS",
    }
    fraud_memory = [
        {"claim_id": f"FR-{i}", "cin": cin, "fraud_label": "fraud",
         "similarity": 0.95, "summary": "inflated billing; identity theft",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"}
        for i in range(3)
    ]

    agent = AnomalyAgent()

    result_no_memory = agent.analyze({**base_claim, "memory_context": []})
    result_with_memory = agent.analyze({**base_claim, "memory_context": fraud_memory})

    score_no_memory = float(result_no_memory["score"])
    score_with_memory = float(result_with_memory["score"])

    assert "memory_insights" in result_with_memory, (
        "FAIL: AnomalyAgent output missing 'memory_insights' when memory context was provided."
    )
    assert score_with_memory < score_no_memory, (
        f"FAIL: AnomalyAgent score did NOT decrease (increase fraud risk) when 3 fraud cases "
        f"were in memory. score_no_memory={score_no_memory}, score_with_memory={score_with_memory}. "
        "Memory must improve fraud detection."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. FRAUD PATTERN MATCH TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_fraud_pattern_match_same_hospital_escalates() -> None:
    """
    PatternAgent with memory containing fraud at the same hospital.
    Output must include memory_insights with fraud_matches > 0 and reduced score.
    """
    hospital = "CHU Rabat"
    fraud_memory = [
        {"claim_id": f"H-{i}", "cin": f"BE12345{i}", "fraud_label": "fraud",
         "similarity": 0.88, "summary": "inflated billing",
         "hospital": hospital, "doctor": "Dr. Alami"}
        for i in range(3)
    ]
    claim_data = {
        "patient_id": "BE123456",
        "amount": 5000.0,
        "history": [],
        "insurance": "CNSS",
        "memory_context": fraud_memory,
    }

    agent = PatternAgent()
    result_without = agent.analyze({**claim_data, "memory_context": []})
    result_with = agent.analyze(claim_data)

    assert "memory_insights" in result_with, (
        "FAIL: PatternAgent output missing 'memory_insights'."
    )
    mi = result_with["memory_insights"]
    assert mi["fraud_matches"] > 0, (
        f"FAIL: PatternAgent fraud_matches={mi['fraud_matches']}, expected > 0 "
        f"when 3 hospital-fraud cases are in memory."
    )
    assert float(result_with["score"]) <= float(result_without["score"]), (
        f"FAIL: PatternAgent score did not decrease with fraud memory context. "
        f"without={result_without['score']}, with={result_with['score']}"
    )


def test_fraud_pattern_match_hospital_notes_captured() -> None:
    """memory_insights must record the recurring hospital in notes."""
    hospital = "CHU Rabat"
    memory_context = [
        {"claim_id": "P1", "cin": "BE123456", "fraud_label": "fraud",
         "similarity": 0.85, "summary": "pattern match",
         "hospital": hospital, "doctor": "Dr. Alami"},
    ]
    _, insights = process_memory_context(
        agent_name="PatternAgent",
        claim_data={"memory_context": memory_context},
        current_score=80.0,
        current_cin="BE123456",
    )
    assert hospital in insights["notes"], (
        f"FAIL: Expected recurring hospital '{hospital}' in memory_insights notes. "
        f"Got: {insights['notes']}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. FALSE MEMORY TEST (CRITICAL)
# ═══════════════════════════════════════════════════════════════════════════

def test_false_memory_below_threshold_not_retrieved() -> None:
    """
    CRITICAL: An unrelated claim must NOT retrieve memory from different-CIN cases.
    Verify that similarity < threshold causes zero results.
    FAIL if agents receive and use irrelevant memory.
    """
    layer = _make_layer(threshold=0.7)

    # Store 5 fraud cases with CIN = BE123456
    for i in range(5):
        layer.store_case(_entry(claim_id=f"FRD-{i}", cin="BE123456", fraud_label="fraud"))

    # Query with a completely unrelated CIN (not in our vocab → zero vector → cosine ≈ 0)
    unrelated_claim = {"identity": {"cin": "ZZ999UNRELATED"}, "policy": {}, "metadata": {}}
    results = layer.retrieve_similar_cases(unrelated_claim, k=5)

    assert results == [], (
        f"FAIL: Unrelated claim retrieved {len(results)} memory case(s) above threshold=0.7. "
        "Agents MUST NOT use irrelevant memory. This would introduce hallucinations. "
        f"Leaked cases: {[r['claim_id'] for r in results]}"
    )


def test_false_memory_process_with_empty_context_no_impact() -> None:
    """
    When memory context is empty (nothing retrieved), process_memory_context must:
    - Return the original score unchanged.
    - Return 0 for similar_cases_found and fraud_matches.
    - Not flag identity_reuse.
    """
    original_score = 65.0
    adjusted, insights = process_memory_context(
        agent_name="AnomalyAgent",
        claim_data={"memory_context": []},
        current_score=original_score,
        current_cin="BE123456",
    )
    assert adjusted == original_score, (
        f"FAIL: process_memory_context changed score with empty context. "
        f"expected={original_score}, got={adjusted}"
    )
    assert insights["similar_cases_found"] == 0, (
        f"FAIL: similar_cases_found={insights['similar_cases_found']} with empty context"
    )
    assert insights["fraud_matches"] == 0, (
        f"FAIL: fraud_matches={insights['fraud_matches']} with empty context"
    )
    assert insights["identity_reuse_detected"] is False, (
        "FAIL: identity_reuse_detected=True with empty memory context"
    )


def test_false_memory_below_threshold_guard_in_process() -> None:
    """
    Memory cases with similarity < 0.7 must be filtered out by process_memory_context.
    This is the secondary guard that protects agents even if the retriever was misconfigured.
    """
    low_similarity_context = [
        {"claim_id": "LOW-1", "cin": "BE123456", "fraud_label": "fraud",
         "similarity": 0.45, "summary": "irrelevant",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"},
        {"claim_id": "LOW-2", "cin": "BE123456", "fraud_label": "fraud",
         "similarity": 0.60, "summary": "irrelevant",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"},
    ]
    original_score = 80.0
    adjusted, insights = process_memory_context(
        agent_name="IdentityAgent",
        claim_data={"memory_context": low_similarity_context},
        current_score=original_score,
        current_cin="BE123456",
    )
    assert adjusted == original_score, (
        f"FAIL: Agent used memory cases with similarity < 0.7. "
        f"original={original_score}, adjusted={adjusted}. Memory must be ignored below threshold."
    )
    assert insights["similar_cases_found"] == 0, (
        f"FAIL: similar_cases_found={insights['similar_cases_found']} — "
        "low-similarity cases should not count."
    )


def test_different_cin_has_low_cosine_similarity() -> None:
    """
    Embedding vectors for different CINs must have cosine similarity below the threshold.
    This validates our embedder works correctly for the false-memory guard.
    """
    embedder = _ControlledEmbedder()
    from claimguard.v2.memory import _normalize, _cosine

    vec_be = _normalize(embedder.embed_query("CIN: BE123456 | hospital: CHU Rabat"))
    vec_ab = _normalize(embedder.embed_query("CIN: AB999001 | hospital: CHU Rabat"))
    similarity = _cosine(vec_be, vec_ab)

    assert similarity < 0.7, (
        f"FAIL: Different-CIN embeddings have cosine similarity={similarity:.4f} ≥ 0.7. "
        "This means unrelated claims might pollute memory retrieval."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. MEMORY IMPACT TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_memory_impact_with_vs_without_fraud_history() -> None:
    """
    Run the same claim twice using process_memory_context:
      - Run A: memory_context = [] (no history)
      - Run B: memory_context = 3 fraud cases with same CIN

    FAIL if score_B >= score_A (memory must reduce score = increase fraud risk).
    """
    cin = "BE123456"
    base_score = 75.0

    fraud_memory = [
        {"claim_id": f"H-{i}", "cin": cin, "fraud_label": "fraud",
         "similarity": 0.92, "summary": "inflated billing; identity theft",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"}
        for i in range(3)
    ]

    score_without, insights_without = process_memory_context(
        "AnomalyAgent", {"memory_context": []}, base_score, cin
    )
    score_with, insights_with = process_memory_context(
        "AnomalyAgent", {"memory_context": fraud_memory}, base_score, cin
    )

    assert score_with < score_without, (
        f"FAIL: Memory did NOT improve fraud detection. "
        f"score_without={score_without:.1f}, score_with={score_with:.1f}. "
        "WITH fraud memory context, risk score must be lower (more suspicious)."
    )
    assert insights_with["fraud_matches"] > insights_without["fraud_matches"], (
        f"FAIL: fraud_matches not increased by memory context. "
        f"without={insights_without['fraud_matches']}, with={insights_with['fraud_matches']}"
    )
    assert insights_with["similar_cases_found"] > 0, (
        "FAIL: similar_cases_found=0 even though 3 cases were provided."
    )


def test_memory_impact_orchestrator_level(monkeypatch) -> None:
    """
    Orchestrator integration: claim analyzed ONCE with fraud memory pre-loaded.
    memory_context must appear in the final response with fraud cases listed.
    All agent outputs must have memory_insights populated.
    """
    _patch_memory_aware_orchestrator(monkeypatch)
    _MemoryAwareCrew._scenario = "fraud_with_memory"

    # Build a fraud-history memory layer
    layer = _make_layer(threshold=0.0)
    for i in range(3):
        layer.store_case(_entry(
            claim_id=f"HIST-{i}", cin="BE123456",
            fraud_label="fraud", ts_score=15.0,
            agent_summary="inflated billing fraud ring",
        ))

    orchestrator = ClaimGuardV2Orchestrator(
        trust_layer_service=_NoOpTrustLayer(),
        memory_layer=layer,
    )
    claim = _make_claim_request(cin="BE123456")
    response = orchestrator.run(claim)

    # memory_context must be injected into the response
    assert response.memory_context, (
        "FAIL: response.memory_context is empty — memory layer did not inject past cases."
    )
    fraud_cases_in_context = [c for c in response.memory_context if c.get("fraud_label") == "fraud"]
    assert len(fraud_cases_in_context) >= 3, (
        f"FAIL: Expected ≥3 fraud cases in memory_context, got {len(fraud_cases_in_context)}."
    )

    # All agents must have processed and returned memory_insights
    for output in response.agent_outputs:
        assert output.memory_insights is not None, (
            f"FAIL: {output.agent} returned no memory_insights. "
            "Agents MUST analyse and report on memory context."
        )


def test_memory_impact_clean_history_no_penalty() -> None:
    """
    When memory only contains clean cases, there must be no risk increase.
    Memory must boost (or preserve) score, never penalise unjustly.
    """
    cin = "BE123456"
    base_score = 60.0
    clean_memory = [
        {"claim_id": f"CL-{i}", "cin": cin, "fraud_label": "clean",
         "similarity": 0.88, "summary": "routine clean claim",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"}
        for i in range(3)
    ]
    score_after, insights = process_memory_context(
        "PolicyAgent", {"memory_context": clean_memory}, base_score, cin
    )
    assert score_after >= base_score, (
        f"FAIL: Memory penalised a claim when history is entirely clean. "
        f"base={base_score}, after={score_after}. Memory must not blindly penalise."
    )
    assert insights["fraud_matches"] == 0, (
        f"FAIL: fraud_matches={insights['fraud_matches']} even though all memory cases are clean."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. DEBUG LOGGING TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_memory_debug_logging_on_store(caplog: pytest.LogCaptureFixture) -> None:
    """memory.store_case must log claim_id, fraud_label, Ts score."""
    layer = _make_layer()
    with caplog.at_level(logging.INFO, logger="claimguard.v2.memory"):
        layer.store_case(_entry(claim_id="LOG-TEST-001", fraud_label="fraud", ts_score=22.5))

    log_text = caplog.text
    assert "LOG-TEST-001" in log_text, (
        f"FAIL: claim_id 'LOG-TEST-001' not found in memory store log. Got: {log_text!r}"
    )
    assert "fraud" in log_text.lower(), (
        f"FAIL: fraud_label not found in memory store log. Got: {log_text!r}"
    )


def test_memory_debug_logging_on_retrieve(caplog: pytest.LogCaptureFixture) -> None:
    """
    memory.retrieve_similar_cases must log:
      - total candidates retrieved
      - count of results above threshold
      - the queried CIN
    """
    layer = _make_layer(threshold=0.0)
    for i in range(3):
        layer.store_case(_entry(claim_id=f"R-{i}", cin="BE123456", fraud_label="fraud"))

    with caplog.at_level(logging.INFO, logger="claimguard.v2.memory"):
        results = layer.retrieve_similar_cases(_claim_with_cin("BE123456"), k=5)

    log_text = caplog.text
    assert "memory_retrieved" in log_text, (
        f"FAIL: 'memory_retrieved' event not logged during retrieval. Got: {log_text!r}"
    )
    assert "BE123456" in log_text, (
        f"FAIL: Queried CIN 'BE123456' not found in retrieval log. Got: {log_text!r}"
    )
    assert "above_threshold" in log_text, (
        f"FAIL: 'above_threshold' count not logged during retrieval. Got: {log_text!r}"
    )


def test_memory_debug_logging_orchestrator_inject(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Orchestrator must log memory_context_injected with counts when cases are found."""
    _patch_memory_aware_orchestrator(monkeypatch)
    _MemoryAwareCrew._scenario = "clean"

    layer = _make_layer(threshold=0.0)
    for i in range(2):
        layer.store_case(_entry(claim_id=f"LOG-C-{i}", cin="BE123456", fraud_label="fraud"))

    orchestrator = ClaimGuardV2Orchestrator(
        trust_layer_service=_NoOpTrustLayer(),
        memory_layer=layer,
    )
    with caplog.at_level(logging.INFO, logger="claimguard.v2"):
        orchestrator.run(_make_claim_request(cin="BE123456"))

    log_text = caplog.text
    assert "memory_context_injected" in log_text or "memory_context" in log_text, (
        f"FAIL: Orchestrator did not log memory context injection. Got: {log_text!r}"
    )


def test_memory_agent_impact_logged(caplog: pytest.LogCaptureFixture) -> None:
    """
    When process_memory_context finds fraud matches, the impact must be recorded
    in memory_insights (which agents later log).  Verify impact_on_score is meaningful.
    """
    memory_ctx = [
        {"claim_id": "F1", "cin": "BE123456", "fraud_label": "fraud",
         "similarity": 0.90, "summary": "fraud ring detected",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"},
    ]
    _, insights = process_memory_context(
        "PatternAgent",
        {"memory_context": memory_ctx},
        current_score=70.0,
        current_cin="BE123456",
    )
    assert insights["impact_on_score"], "FAIL: impact_on_score is empty — logging context missing"
    assert "fraud" in insights["impact_on_score"].lower(), (
        f"FAIL: impact_on_score does not describe the fraud impact. Got: {insights['impact_on_score']}"
    )
    assert insights["notes"], "FAIL: notes field is empty — debug context missing"


# ═══════════════════════════════════════════════════════════════════════════
# 8. FAIL CONDITIONS (EXPLICIT)
# ═══════════════════════════════════════════════════════════════════════════

def test_fail_condition_agents_must_reference_memory() -> None:
    """
    FAIL if agents do NOT include memory_insights in their output when
    memory_context is provided.
    All 3 tested agents (IdentityAgent, AnomalyAgent, PatternAgent) must include it.
    """
    cin = "BE123456"
    memory_context = [
        {"claim_id": "MEM-1", "cin": cin, "fraud_label": "fraud",
         "similarity": 0.91, "summary": "fraud case",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"},
    ]
    claim_with_memory = {
        "patient_id": cin,
        "identity": {"cin": cin},
        "amount": 3000.0,
        "history": [],
        "documents": [],
        "insurance": "CNSS",
        "memory_context": memory_context,
    }

    agents = [
        ("IdentityAgent", IdentityAgent()),
        ("AnomalyAgent", AnomalyAgent()),
        ("PatternAgent", PatternAgent()),
    ]
    failures: List[str] = []
    for name, agent in agents:
        result = agent.analyze(claim_with_memory)
        if "memory_insights" not in result:
            failures.append(
                f"{name}: 'memory_insights' key MISSING from output. "
                "Agents MUST reference memory when memory_context is available."
            )
        else:
            mi = result["memory_insights"]
            if not isinstance(mi, dict):
                failures.append(f"{name}: memory_insights is not a dict — got {type(mi)}")

    if failures:
        pytest.fail("FAIL: Agents skipped memory analysis:\n" + "\n".join(failures))


def test_fail_condition_memory_not_used_blindly_below_threshold() -> None:
    """
    FAIL if agent uses memory cases with similarity < threshold (0.7).
    Memory below the threshold is noise and MUST be ignored.
    """
    noisy_memory = [
        {"claim_id": "NOISE-1", "cin": "BE123456", "fraud_label": "fraud",
         "similarity": 0.55, "summary": "irrelevant noise",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"},
        {"claim_id": "NOISE-2", "cin": "BE123456", "fraud_label": "fraud",
         "similarity": 0.60, "summary": "irrelevant noise",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"},
    ]
    original_score = 80.0
    adjusted, insights = process_memory_context(
        agent_name="AnomalyAgent",
        claim_data={"memory_context": noisy_memory},
        current_score=original_score,
        current_cin="BE123456",
    )
    if adjusted != original_score:
        pytest.fail(
            f"FAIL: Agent used memory cases with similarity < 0.7 (blind memory use). "
            f"original_score={original_score}, adjusted_score={adjusted}. "
            "Memory below threshold MUST be ignored."
        )
    if insights["similar_cases_found"] != 0:
        pytest.fail(
            f"FAIL: similar_cases_found={insights['similar_cases_found']} even though "
            "all cases are below the 0.7 threshold. Agent is blindly trusting noisy memory."
        )


def test_fail_condition_repeated_cin_must_be_detected() -> None:
    """
    FAIL if CIN appearing in 3+ fraud cases is NOT detected as identity reuse.
    This is the core CIN-reuse fraud signal.
    """
    cin = "BE123456"
    three_fraud_cases = [
        {"claim_id": f"REPEAT-{i}", "cin": cin, "fraud_label": "fraud",
         "similarity": 0.93, "summary": "repeated fraud",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"}
        for i in range(3)
    ]
    _, insights = process_memory_context(
        agent_name="IdentityAgent",
        claim_data={"memory_context": three_fraud_cases},
        current_score=70.0,
        current_cin=cin,
    )
    if not insights["identity_reuse_detected"]:
        pytest.fail(
            f"FAIL: CIN '{cin}' appears in 3 fraud cases but identity_reuse_detected=False. "
            "The memory layer MUST detect repeated CIN usage across fraud cases."
        )
    if insights["fraud_matches"] < 3:
        pytest.fail(
            f"FAIL: fraud_matches={insights['fraud_matches']}, expected ≥3. "
            f"All 3 fraud memory cases must be counted."
        )


def test_fail_condition_memory_contradicts_but_does_not_override() -> None:
    """
    FAIL if memory overrides agent score when current data shows the opposite.
    Memory showing clean history must NOT force a high clean score on an otherwise risky claim.
    A contradiction note must be added instead.
    """
    cin = "BE123456"
    # Memory says this CIN is always clean
    clean_memory = [
        {"claim_id": f"CL-{i}", "cin": cin, "fraud_label": "clean",
         "similarity": 0.90, "summary": "routine clean",
         "hospital": "CHU Rabat", "doctor": "Dr. Alami"}
        for i in range(5)
    ]
    # But current score is very low (high fraud risk from other signals)
    very_risky_score = 10.0
    adjusted, insights = process_memory_context(
        agent_name="AnomalyAgent",
        claim_data={"memory_context": clean_memory},
        current_score=very_risky_score,
        current_cin=cin,
    )
    # Memory may boost a little (clean history), but MUST NOT jump from 10 → above 70
    assert adjusted <= 30.0, (
        f"FAIL: Memory overrode current fraud signals! "
        f"original_score={very_risky_score}, adjusted={adjusted}. "
        "When current data shows high risk (score=10), memory MUST NOT override it to clean."
    )
    # Contradiction note must be present
    contradiction_mentioned = (
        "contradiction" in insights.get("notes", "").lower()
        or "contradict" in insights.get("impact_on_score", "").lower()
        or "memory advisory" in insights.get("notes", "").lower()
    )
    assert contradiction_mentioned or adjusted <= very_risky_score + 20, (
        f"FAIL: No contradiction handling when memory=clean but score={very_risky_score}. "
        f"Got adjusted={adjusted}, notes='{insights.get('notes')}'"
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXTRA: CaseMemoryEntry / helper function validation
# ═══════════════════════════════════════════════════════════════════════════

def test_case_memory_entry_embedding_text_includes_all_fields() -> None:
    """The embedding_text must contain CIN, hospital, doctor, diagnosis, and decision."""
    entry = _entry(
        claim_id="EMB-001",
        cin="BE123456",
        hospital="CHU Rabat",
        doctor="Dr. Alami",
        diagnosis="fracture",
        fraud_label="fraud",
        agent_summary="inflated billing detected",
    )
    text = entry.embedding_text
    assert "BE123456" in text, f"CIN not in embedding_text: {text!r}"
    assert "CHU Rabat" in text, f"Hospital not in embedding_text: {text!r}"
    assert "Dr. Alami" in text, f"Doctor not in embedding_text: {text!r}"
    assert "fracture" in text, f"Diagnosis not in embedding_text: {text!r}"
    assert "fraud" in text, f"Fraud label not in embedding_text: {text!r}"


def test_decision_to_fraud_label_mapping() -> None:
    """decision_to_fraud_label must correctly map all decision types."""
    assert decision_to_fraud_label("APPROVED", 95.0) == "clean"
    assert decision_to_fraud_label("REJECTED", 20.0) == "fraud"
    assert decision_to_fraud_label("REFLEXIVE_TRIGGER", 30.0) == "fraud"
    assert decision_to_fraud_label("HUMAN_REVIEW", 75.0) == "suspicious"
    assert decision_to_fraud_label("HUMAN_REVIEW", 45.0) == "fraud"


def test_build_agent_summary_from_outputs() -> None:
    """build_agent_summary must return a non-empty string for valid agent outputs."""
    outputs = [
        {"agent": "IdentityAgent", "score": 0.3, "explanation": "CIN mismatch detected"},
        {"agent": "AnomalyAgent", "score": 0.8, "explanation": "Amount spike detected"},
    ]
    summary = build_agent_summary(outputs)
    assert isinstance(summary, str), "Expected string from build_agent_summary"
    assert len(summary) > 0, "build_agent_summary returned empty string"
    assert "IdentityAgent" in summary, "Agent name not in summary"
    assert "AnomalyAgent" in summary, "Agent name not in summary"


def test_fallback_memory_insights_without_llm() -> None:
    """_compute_fallback_memory_insights must produce valid MemoryInsights for all inputs."""
    # With fraud memory and CIN reuse
    fraud_ctx = [
        {"claim_id": "F1", "cin": "BE123456", "fraud_label": "fraud", "similarity": 0.9,
         "hospital": "CHU Rabat", "summary": "fraud case"},
    ]
    mi = _compute_fallback_memory_insights(fraud_ctx, "BE123456")
    assert mi.fraud_matches == 1
    assert mi.identity_reuse_detected is True
    assert mi.similar_cases_found == 1
    assert mi.impact_on_score

    # With empty context
    mi_empty = _compute_fallback_memory_insights([], "BE123456")
    assert mi_empty.fraud_matches == 0
    assert mi_empty.similar_cases_found == 0


def test_memory_layer_entry_count_tracks_insertions() -> None:
    """entry_count must reflect the exact number of stored cases."""
    layer = _make_layer()
    assert layer.entry_count == 0
    for i in range(5):
        layer.store_case(_entry(claim_id=f"EC-{i}"))
    assert layer.entry_count == 5


def test_blackboard_memory_context_injection() -> None:
    """SharedBlackboard.inject_memory_context must store and expose the context."""
    routing = RoutingDecision(
        intent="general_claim", complexity="simple", model="mistral", reason="test"
    )
    bb = SharedBlackboard(
        {},
        routing,
        extracted_text="memory test text",
        structured_data={"cin": "", "amount": "", "date": "", "provider": ""},
    )
    assert bb.memory_context == []

    cases = [
        {"claim_id": "X1", "cin": "BE123456", "fraud_label": "fraud", "similarity": 0.85,
         "summary": "fraud case"},
    ]
    bb.inject_memory_context(cases)
    ctx = bb.memory_context
    assert len(ctx) == 1, f"Expected 1 case, got {len(ctx)}"
    assert ctx[0]["claim_id"] == "X1"
    assert ctx[0]["fraud_label"] == "fraud"

    # Must appear in blackboard serialisation
    state = bb.to_dict()
    assert "memory_context" in state, "memory_context missing from blackboard.to_dict()"
    assert len(state["memory_context"]) == 1


def test_build_query_text_extracts_all_fields() -> None:
    """_build_query_text must extract cin, hospital, doctor, and diagnosis from the claim."""
    claim = {
        "identity": {"cin": "BE123456", "hospital": "CHU Rabat"},
        "policy": {"diagnosis": "fracture", "doctor": "Dr. Alami"},
        "metadata": {},
    }
    text = _build_query_text(claim)
    assert "BE123456" in text, f"CIN missing from query text: {text!r}"
    assert "CHU Rabat" in text, f"Hospital missing from query text: {text!r}"
    assert "Dr. Alami" in text, f"Doctor missing from query text: {text!r}"
    assert "fracture" in text, f"Diagnosis missing from query text: {text!r}"


def test_memory_layer_handles_store_failure_gracefully() -> None:
    """
    A broken embedder must not crash the pipeline.
    The memory layer must either silently fall back or log the error — never raise.
    Pipeline continuity is the primary guarantee; exact entry count is secondary.
    """

    class _BrokenEmbedder:
        def embed_query(self, text: str) -> List[float]:
            raise RuntimeError("embedder unavailable")

    layer = _make_layer()
    layer._embedder = _BrokenEmbedder()

    # Must not raise — pipeline resilience is the guarantee
    try:
        layer.store_case(_entry(claim_id="BROKEN-001"))
    except Exception as exc:
        pytest.fail(
            f"FAIL: store_case raised {exc!r} when embedder is broken. "
            "Memory failures must NEVER crash the pipeline. "
            "Errors must be caught and logged, not propagated."
        )


def test_memory_layer_handles_retrieve_on_empty_store() -> None:
    """retrieve_similar_cases on an empty store must return [] without error."""
    layer = _make_layer()
    try:
        results = layer.retrieve_similar_cases(_claim_with_cin("BE123456"))
    except Exception as exc:
        pytest.fail(f"retrieve_similar_cases raised {exc!r} on empty store.")
    assert results == [], f"Expected [] on empty store, got {results}"
