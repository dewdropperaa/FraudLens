import sys
import base64
from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.main import create_app
from claimguard.v2.blackboard import BlackboardValidationError, SharedBlackboard
from claimguard.v2.concierge import build_routing_decision
from claimguard.v2.consensus import ConsensusEngine
from claimguard.v2.schemas import RoutingDecision
from claimguard.v2.trust_layer import (
    FirebaseTrustRecord,
    OnChainTrustPayload,
    SanitizedTrustDocument,
    TrustLayerIPFSFailure,
    TrustLayerService,
)


def test_concierge_routing_and_model_selection() -> None:
    decision = build_routing_decision(
        {
            "identity": {"claimant_type": "hospital"},
            "documents": [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}, {"id": "d4"}],
            "policy": {},
            "metadata": {},
        }
    )
    assert decision.intent == "claim_hospital"
    assert decision.complexity == "complex"
    assert decision.model == "deepseek-r1"


def test_blackboard_validation_guard_blocks_missing_context() -> None:
    bb = SharedBlackboard(
        request_payload={"identity": {}, "documents": [], "policy": {}, "metadata": {}},
        routing=RoutingDecision(
            intent="general_claim",
            complexity="simple",
            model="mistral",
            reason="test",
        ),
        extracted_text="stub text",
        structured_data={"cin": "", "amount": "", "date": "", "provider": ""},
    )
    try:
        bb.require(("IdentityAgent",))
    except BlackboardValidationError:
        return
    raise AssertionError("Expected BlackboardValidationError")


def test_v2_endpoint_returns_architecture_shape(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as orchestrator_module
    from claimguard.v2.schemas import ClaimGuardV2Response

    class FakeOrchestrator:
        def run(self, claim_request):
            return ClaimGuardV2Response(
                agent_outputs=[],
                blackboard={"entries": {}},
                routing_decision=RoutingDecision(
                    intent="general_claim",
                    complexity="simple",
                    model="mistral",
                    reason="fake",
                ),
                goa_used=False,
                Ts=82.5,
                decision="HUMAN_REVIEW",
                exit_reason="low_confidence",
                retry_count=0,
                mahic_breakdown={"billing": 10.0, "clinical": 8.0, "temporal": 5.0, "geo": 3.0},
                contradictions=[],
            )

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DOCUMENT_ENCRYPTION_KEY", "0" * 32)
    monkeypatch.setenv("CLAIMAGUARD_API_KEYS", "test-api-key-for-ci")
    monkeypatch.setattr(orchestrator_module, "_singleton", FakeOrchestrator())
    client = TestClient(create_app())
    response = client.post(
        "/v2/claim/analyze",
        headers={"x-api-key": "test-api-key-for-ci"},
        json={"identity": {}, "documents": [], "policy": {}, "metadata": {}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "agent_outputs" in payload
    assert "blackboard" in payload
    assert "routing_decision" in payload
    assert payload["goa_used"] is False
    assert "Ts" in payload
    assert "decision" in payload
    assert "retry_count" in payload
    assert "mahic_breakdown" in payload
    assert "contradictions" in payload


def test_v2_endpoint_builds_document_extractions_from_base64_pdf(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as orchestrator_module
    from claimguard.v2.schemas import ClaimGuardV2Response

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.last_claim_request = None

        def run(self, claim_request):
            self.last_claim_request = claim_request
            return ClaimGuardV2Response(
                agent_outputs=[],
                blackboard={"entries": {}},
                routing_decision=RoutingDecision(
                    intent="general_claim",
                    complexity="simple",
                    model="mistral",
                    reason="fake",
                ),
                goa_used=False,
                Ts=0.0,
                decision="REJECTED",
                exit_reason="low_confidence",
                retry_count=0,
                mahic_breakdown={"billing": 0.0, "clinical": 0.0, "temporal": 0.0, "geo": 0.0},
                contradictions=[],
            )

    fake = FakeOrchestrator()
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DOCUMENT_ENCRYPTION_KEY", "0" * 32)
    monkeypatch.setenv("CLAIMAGUARD_API_KEYS", "test-api-key-for-ci")
    monkeypatch.setattr(orchestrator_module, "_singleton", fake)
    client = TestClient(create_app())

    # Minimal PDF-like bytes to ensure the extraction path is exercised.
    pdf_b64 = base64.b64encode(b"%PDF-1.4\n%EOF").decode("ascii")
    response = client.post(
        "/v2/claim/analyze",
        headers={"x-api-key": "test-api-key-for-ci"},
        json={
            "identity": {},
            "documents": [],
            "documents_base64": [{"name": "claim.pdf", "content_base64": pdf_b64}],
            "policy": {},
            "metadata": {},
        },
    )
    assert response.status_code == 200
    assert fake.last_claim_request is not None
    assert "documents_base64" not in fake.last_claim_request
    assert "document_extractions" in fake.last_claim_request
    assert len(fake.last_claim_request["document_extractions"]) == 1
    extraction = fake.last_claim_request["document_extractions"][0]
    assert extraction["file_name"] == "claim.pdf"
    assert extraction["extraction_method"] in {
        "pypdf",
        "pdf_ocr",
        "pdf_ocr_failed",
        "pdf_ocr_unavailable",
        "pypdf_failed",
    }


def test_consensus_engine_ts_formula_and_decision_thresholds() -> None:
    engine = ConsensusEngine()
    entries = {
        "IdentityAgent": {"score": 0.8, "confidence": 0.9, "explanation": "ok"},
        "DocumentAgent": {"score": 0.3, "confidence": 0.8, "explanation": "weak"},
        "PolicyAgent": {"score": 0.9, "confidence": 0.95, "explanation": "ok"},
        "AnomalyAgent": {"score": 0.75, "confidence": 0.9, "explanation": "high anomaly"},
        "PatternAgent": {"score": 0.6, "confidence": 0.7, "explanation": "moderate"},
        "GraphRiskAgent": {"score": 0.8, "confidence": 0.85, "explanation": "network risk"},
    }
    result = engine.evaluate(
        claim_request={"identity": {}, "documents": [{"id": "1"}], "policy": {}, "metadata": {}},
        entries=entries,
    )
    # Calibration bonuses push this profile above the approve threshold.
    assert result["Ts"] >= 65.0
    assert result["decision"] == "APPROVED"


def test_trust_layer_tier1_hash_runs_for_all_decisions() -> None:
    class FakeIPFS:
        def __init__(self) -> None:
            self.called = False

        def upload_documents(self, claim_id: str, documents: list[SanitizedTrustDocument]) -> str:
            self.called = True
            return "QmCID123"

    class FakeChain:
        def __init__(self) -> None:
            self.called = False

        def store_record(self, payload: OnChainTrustPayload) -> str:
            self.called = True
            return "0xabc"

    class FakeFirebase:
        def __init__(self) -> None:
            self.called = False
            self.last_payload = None

        def store_record(self, payload: FirebaseTrustRecord) -> str:
            self.called = True
            self.last_payload = payload
            return "fire-1"

    class FakeFallback:
        def log_blockchain_failure(self, context):
            raise AssertionError("fallback should not be called")

    ipfs = FakeIPFS()
    chain = FakeChain()
    firebase = FakeFirebase()
    service = TrustLayerService(
        ipfs_client=ipfs,
        blockchain_client=chain,
        firebase_client=firebase,
        fallback_logger=FakeFallback(),
        validator_id="sys-1",
    )
    request = {
        "documents": [
            {"id": "d1", "document_type": "medical_report", "text": "report"},
        ]
    }
    tier1 = service.process_if_applicable(
        claim_id="c1",
        decision="HUMAN_REVIEW",
        ts_score=75.0,
        claim_request=request,
        agent_outputs=[],
    )
    assert tier1 is not None
    assert tier1.trust_hash
    assert ipfs.called is False
    assert chain.called is False
    assert firebase.called is True
    assert firebase.last_payload is not None
    assert firebase.last_payload.decision == "HUMAN_REVIEW"
    assert firebase.last_payload.trust_hash

    result = service.process_if_applicable(
        claim_id="c2",
        decision="APPROVED",
        ts_score=95.0,
        claim_request=request,
        agent_outputs=[{"agent": "IdentityAgent", "explanation": "ok"}],
    )
    assert result is not None
    assert result.status == "stored"
    assert ipfs.called is True
    assert chain.called is True
    assert firebase.called is True


def test_tier1_hash_is_deterministic_for_same_inputs() -> None:
    class FakeIPFS:
        def upload_documents(self, claim_id: str, documents: list[SanitizedTrustDocument]) -> str:
            return "QmCID123"

    class FakeChain:
        def store_record(self, payload: OnChainTrustPayload) -> str:
            return "0xabc"

    class FakeFirebase:
        def __init__(self) -> None:
            self.payloads: list[FirebaseTrustRecord] = []

        def store_record(self, payload: FirebaseTrustRecord) -> str:
            self.payloads.append(payload)
            return f"fire-{len(self.payloads)}"

    class FakeFallback:
        def log_blockchain_failure(self, context):
            return None

    service = TrustLayerService(
        ipfs_client=FakeIPFS(),
        blockchain_client=FakeChain(),
        firebase_client=FakeFirebase(),
        fallback_logger=FakeFallback(),
        validator_id="sys-1",
    )
    ts = "2026-01-01T00:00:00+00:00"
    h1 = service._build_tier1_hash(
        claim_id="c1",
        decision="REJECTED",
        ts_score=55.1,
        timestamp=ts,
        agent_output_summary="a|b|c",
        flags=["x", "y"],
    )
    h2 = service._build_tier1_hash(
        claim_id="c1",
        decision="REJECTED",
        ts_score=55.1,
        timestamp=ts,
        agent_output_summary="a|b|c",
        flags=["y", "x"],
    )
    assert h1 == h2


def test_trust_layer_degrades_when_ipfs_fails() -> None:
    class FailingIPFS:
        def upload_documents(self, claim_id: str, documents: list[SanitizedTrustDocument]) -> str:
            raise TrustLayerIPFSFailure("boom")

    class FakeChain:
        def __init__(self) -> None:
            self.called = False

        def store_record(self, payload: OnChainTrustPayload) -> str:
            self.called = True
            return "0xf00"

    class FakeFirebase:
        def __init__(self) -> None:
            self.called = False

        def store_record(self, payload: FirebaseTrustRecord) -> str:
            self.called = True
            return "fire-degraded"

    class FakeFallback:
        def log_blockchain_failure(self, context):
            raise AssertionError("fallback should not run after IPFS failure")

    chain = FakeChain()
    firebase = FakeFirebase()
    service = TrustLayerService(
        ipfs_client=FailingIPFS(),
        blockchain_client=chain,
        firebase_client=firebase,
        fallback_logger=FakeFallback(),
    )
    result = service.process_if_applicable(
        claim_id="c3",
        decision="APPROVED",
        ts_score=99.0,
        claim_request={"documents": [{"document_type": "invoice", "text": "A"}]},
        agent_outputs=[],
    )
    assert result is not None
    assert result.cid is None
    assert chain.called is True
    assert firebase.called is True
    assert result.tx_hash == "0xf00"


def test_trust_layer_blockchain_fallback_keeps_firebase_write() -> None:
    class FakeIPFS:
        def upload_documents(self, claim_id: str, documents: list[SanitizedTrustDocument]) -> str:
            return "QmCID456"

    class FailingChain:
        def store_record(self, payload: OnChainTrustPayload) -> str:
            raise RuntimeError("rpc down")

    class FakeFirebase:
        def __init__(self) -> None:
            self.writes = 0

        def store_record(self, payload: FirebaseTrustRecord) -> str:
            self.writes += 1
            return "fire-2"

    class FakeFallback:
        def __init__(self) -> None:
            self.logged = 0

        def log_blockchain_failure(self, context):
            self.logged += 1

    firebase = FakeFirebase()
    fallback = FakeFallback()
    service = TrustLayerService(
        ipfs_client=FakeIPFS(),
        blockchain_client=FailingChain(),
        firebase_client=firebase,
        fallback_logger=fallback,
    )
    result = service.process_if_applicable(
        claim_id="c4",
        decision="APPROVED",
        ts_score=93.0,
        claim_request={"documents": [{"document_type": "prescription", "text": "B"}]},
        agent_outputs=[],
    )
    assert result is not None
    assert result.tx_hash is None
    assert result.firebase_id == "fire-2"
    assert firebase.writes == 1
    assert fallback.logged == 1
