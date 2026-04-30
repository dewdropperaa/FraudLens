from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.v2.consensus import ConsensusConfig, ConsensusEngine
from claimguard.v2.memory_health import (
    MemoryConfig,
    MemoryHealthReport,
    MemoryHealthStatus,
    get_memory_health,
)
from claimguard.v2.orchestrator import ClaimGuardV2Orchestrator
from claimguard.v2.trust_layer import (
    FirebaseTrustRecord,
    OnChainTrustPayload,
    SanitizedTrustDocument,
    TrustLayerService,
    is_trust_eligible,
)


def test_memory_health_healthy_path() -> None:
    class _Embedder:
        def embed_query(self, text: str) -> List[float]:
            return [0.1, 0.2, 0.3]

    class _Memory:
        _using_fake_embeddings = False
        _embedder = _Embedder()

        def retrieve_similar_cases(self, *_: Any, **__: Any) -> List[Dict[str, Any]]:
            return [{"claim_id": "probe", "similarity": 0.91}]

    report = get_memory_health(MemoryConfig(min_similarity=0.7), _Memory())
    assert report.status == MemoryHealthStatus.HEALTHY
    assert report.probe_result_count == 1
    assert report.failure_reason == ""


def test_memory_health_degraded_is_informational_no_confidence_penalty() -> None:
    engine = ConsensusEngine()
    entries = {
        "PatternAgent": {"score": 0.7, "confidence": 0.9, "explanation": "p"},
        "GraphRiskAgent": {"score": 0.7, "confidence": 0.9, "explanation": "g"},
    }
    result = engine.evaluate(
        claim_request={"identity": {}, "documents": [{"id": "1"}], "policy": {}, "metadata": {}},
        entries=entries,
        blackboard={"memory_degraded": True, "memory_status": "DEGRADED"},
        config=ConsensusConfig(degraded_memory_penalty=0.15, unavailable_memory_penalty=0.25),
    )
    assert result["entries"]["PatternAgent"]["confidence"] == 0.9
    assert result["entries"]["GraphRiskAgent"]["confidence"] == 0.9


def test_memory_degraded_blocks_auto_approve_below_threshold(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module

    class _MemoryUnavailable:
        _using_fake_embeddings = True
        _embedder = None

        def retrieve_similar_cases(self, *_: Any, **__: Any) -> List[Dict[str, Any]]:
            return []

        def store_case(self, *_: Any, **__: Any) -> None:
            return None

    class _NoOpTrust:
        def process_if_applicable(self, **_: Any):
            return None

    class _FakeAgent:
        def __init__(self, role: str, **_: Any) -> None:
            self.role = role

    class _FakeTask:
        def __init__(self, description: str, agent: _FakeAgent, **_: Any) -> None:
            self.description = description
            self.agent = agent

    class _Crew:
        def __init__(self, tasks: List[_FakeTask], **_: Any) -> None:
            self._task = tasks[0]

        def kickoff(self) -> str:
            return json.dumps(
                {
                    "score": 0.95,
                    "confidence": 0.99,
                    "explanation": "high confidence",
                    "claims": [{"statement": "ok", "evidence": "ok", "verified": True}],
                    "hallucination_flags": [],
                }
            )

    original_eval = module.ConsensusEngine.evaluate

    def _forced_eval(self, **kwargs):
        out = original_eval(self, **kwargs)
        out["decision"] = "APPROVED"
        out["Ts"] = 92.0
        return out

    monkeypatch.setattr(module, "assert_ollama_connection", lambda: None)
    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _Crew)
    monkeypatch.setattr(module.ConsensusEngine, "evaluate", _forced_eval)
    orchestrator = ClaimGuardV2Orchestrator(
        trust_layer_service=_NoOpTrust(),
        memory_layer=_MemoryUnavailable(),
    )
    rich_text = (
        "Facture medicale consultation. Patient Sara Benali CIN AB123456 "
        "Hopital Atlas Docteur Dr A. Date 2026-01-10. Montant total 2000 MAD."
    )
    response = orchestrator.run(
        {
            "identity": {"cin": "AB123456", "hospital": "Hopital Atlas", "doctor": "Dr A"},
            "documents": [{"id": "d1", "document_type": "invoice", "text": rich_text}],
            "document_extractions": [{"file_name": "d1", "extracted_text": rich_text}],
            "policy": {"amount": 2000, "diagnosis": "consult", "hospital": "Hopital Atlas"},
            "metadata": {"claim_id": "memory-block", "service_date": "2026-01-10", "hospital": "Hopital Atlas"},
            "patient_id": "AB123456",
            "amount": 2000,
        }
    )
    assert response.decision != "APPROVED"


def test_consecutive_degraded_claims_triggers_alert(monkeypatch) -> None:
    monkeypatch.setattr("claimguard.v2.orchestrator.assert_ollama_connection", lambda: None)
    orchestrator = ClaimGuardV2Orchestrator(memory_layer=type("M", (), {"_using_fake_embeddings": True})())
    calls: List[str] = []
    monkeypatch.setattr(
        orchestrator,
        "_emit_memory_degraded_alert",
        lambda **kwargs: calls.append(kwargs["claim_id"]),
    )
    degraded = MemoryHealthReport(
        status=MemoryHealthStatus.DEGRADED,
        latency_ms=10,
        probe_result_count=0,
        failure_reason="x",
    )
    orchestrator._track_memory_health(claim_id="c1", report=degraded)
    orchestrator._track_memory_health(claim_id="c2", report=degraded)
    assert calls == ["c2"]


def test_all_decisions_produce_firebase_trust_hash() -> None:
    class _FakeIPFS:
        def upload_documents(self, claim_id: str, documents: list[SanitizedTrustDocument]) -> str:
            return "QmCID"

    class _FakeChain:
        def store_claim(self, *, cid: str, metadata: dict) -> str:
            return "0xtx"

        def store_record(self, payload: OnChainTrustPayload) -> str:
            return "0xtx"

    class _FakeFirebase:
        def __init__(self) -> None:
            self.rows: list[FirebaseTrustRecord] = []

        def store_record(self, payload: FirebaseTrustRecord) -> str:
            self.rows.append(payload)
            return f"f-{len(self.rows)}"

    fb = _FakeFirebase()
    service = TrustLayerService(
        ipfs_client=_FakeIPFS(),
        blockchain_client=_FakeChain(),
        firebase_client=fb,
    )
    for decision in ("APPROVED", "HUMAN_REVIEW", "REJECTED"):
        service.process_if_applicable(
            claim_id=f"id-{decision}",
            decision=decision,
            ts_score=80.0,
            claim_request={"documents": [{"document_type": "invoice", "text": "ok"}]},
            agent_outputs=[{"agent": "A", "explanation": "ok"}],
            flags=["flag-a"],
        )
    assert len(fb.rows) == 3
    assert all(bool(row.trust_hash) for row in fb.rows)


def test_disputed_human_review_produces_ipfs_bundle() -> None:
    class _FakeIPFS:
        def __init__(self) -> None:
            self.calls = 0

        def upload_documents(self, claim_id: str, documents: list[SanitizedTrustDocument]) -> str:
            self.calls += 1
            return "QmEvidence"

    class _FakeChain:
        def store_claim(self, *, cid: str, metadata: dict) -> str:
            return "0xtx"

        def store_record(self, payload: OnChainTrustPayload) -> str:
            return "0xtx"

    class _FakeFirebase:
        def store_record(self, payload: FirebaseTrustRecord) -> str:
            return "f-1"

    ipfs = _FakeIPFS()
    service = TrustLayerService(
        ipfs_client=ipfs,
        blockchain_client=_FakeChain(),
        firebase_client=_FakeFirebase(),
    )
    result = service.process_if_applicable(
        claim_id="hr-1",
        decision="HUMAN_REVIEW",
        ts_score=88.0,
        claim_request={"documents": []},
        agent_outputs=[{"agent": "A", "explanation": "ok"}],
        flags=["memory_degraded"],
        score_evolution=[88.2, 88.0],
        dispute_risk=True,
    )
    assert result is not None
    assert result.evidence_cid == "QmEvidence"
    assert ipfs.calls == 1


def test_rejected_with_dispute_risk_produces_ipfs_bundle() -> None:
    class _FakeIPFS:
        def upload_documents(self, claim_id: str, documents: list[SanitizedTrustDocument]) -> str:
            return "QmRejEvidence"

    class _FakeChain:
        def store_claim(self, *, cid: str, metadata: dict) -> str:
            return "0xtx"

        def store_record(self, payload: OnChainTrustPayload) -> str:
            return "0xtx"

    class _FakeFirebase:
        def store_record(self, payload: FirebaseTrustRecord) -> str:
            return "f-2"

    service = TrustLayerService(
        ipfs_client=_FakeIPFS(),
        blockchain_client=_FakeChain(),
        firebase_client=_FakeFirebase(),
    )
    result = service.process_if_applicable(
        claim_id="rj-1",
        decision="REJECTED",
        ts_score=40.0,
        claim_request={"documents": []},
        agent_outputs=[{"agent": "A", "explanation": "ok"}],
        flags=["contradiction_high"],
        score_evolution=[45.0, 40.0],
        dispute_risk=True,
    )
    assert result is not None
    assert result.evidence_cid == "QmRejEvidence"


def test_is_trust_eligible_accepts_orchestrator_blackboard_shape() -> None:
    blackboard = {
        "identity": {"cin": "AB123456"},
        "verified_structured_data": {"amount": "2400"},
        "document_classification": {"label": "MEDICAL_CLAIM"},
        "extracted_text": "Facture medicale montant 2400 MAD",
    }
    assert is_trust_eligible(blackboard) is True


def test_trust_layer_fallback_document_when_claim_request_documents_not_eligible() -> None:
    class _FakeIPFS:
        def __init__(self) -> None:
            self.calls = 0
            self.last_docs: list[SanitizedTrustDocument] = []

        def upload_documents(self, claim_id: str, documents: list[SanitizedTrustDocument]) -> str:
            self.calls += 1
            self.last_docs = list(documents)
            return "QmFallbackCID"

    class _FakeChain:
        def store_claim(self, *, cid: str, metadata: dict) -> str:
            return "0xtx"

        def store_record(self, payload: OnChainTrustPayload) -> str:
            return "0xtx"

    class _FakeFirebase:
        def store_record(self, payload: FirebaseTrustRecord) -> str:
            return "f-fallback"

    ipfs = _FakeIPFS()
    service = TrustLayerService(
        ipfs_client=ipfs,
        blockchain_client=_FakeChain(),
        firebase_client=_FakeFirebase(),
    )
    result = service.process_approved_claim(
        {
            "claim_id": "fallback-1",
            "decision": "APPROVED",
            "ts_score": 91.0,
            "claim_request": {
                "documents": [
                    {
                        "id": "d-non-trust-type",
                        "document_type": "other",
                        "text": "short text",
                    }
                ]
            },
            "blackboard": {
                "identity": {"cin": "AB123456"},
                "verified_structured_data": {"amount": "2400"},
                "document_classification": {"label": "MEDICAL_CLAIM"},
                "extracted_text": "Facture medicale montant total 2400 MAD",
            },
            "agent_outputs": [{"agent": "A", "explanation": "ok"}],
            "flags": [],
        }
    )
    assert result["cid"] == "QmFallbackCID"
    assert ipfs.calls == 1
    assert len(ipfs.last_docs) >= 1

