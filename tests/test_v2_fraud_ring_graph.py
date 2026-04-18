from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.v2.fraud_ring_graph import FraudRingGraph
from claimguard.v2.orchestrator import ClaimGuardV2Orchestrator, SEQUENTIAL_AGENT_CONTRACTS


class _NoOpTrustLayer:
    def process_if_applicable(self, **_: Any) -> None:
        return None


class _FakeAgent:
    def __init__(self, role: str, **_: Any) -> None:
        self.role = role


class _FakeTask:
    def __init__(self, description: str, agent: _FakeAgent, **_: Any) -> None:
        self.description = description
        self.agent = agent


class _FakeCrew:
    def __init__(self, tasks: List[_FakeTask], **_: Any) -> None:
        self._task = tasks[0]

    def kickoff(self) -> str:
        role = self._task.agent.role
        payload = {
            "score": 0.35 if role == "GraphRiskAgent" else 0.8,
            "confidence": 0.8,
            "explanation": f"{role} baseline explanation.",
        }
        return json.dumps(payload)


def _patch_orchestrator(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _FakeCrew)


def test_repeated_cin_is_detected() -> None:
    graph = FraudRingGraph()
    graph.add_claim(claim_id="C1", cin="AA111", hospital="H1", doctor="D1", anomaly_score=0.7)
    analysis = graph.add_claim(claim_id="C2", cin="AA111", hospital="H1", doctor="D2", anomaly_score=0.75)
    assert analysis["reuse_detection"]["cin_reuse_detected"] is True
    assert analysis["reuse_detection"]["cin_claim_count"] >= 2


def test_cluster_is_flagged_when_ring_criteria_match() -> None:
    graph = FraudRingGraph()
    graph.add_claim(claim_id="C1", cin="CIN-1", hospital="H-RING", doctor="D-1", anomaly_score=0.9)
    graph.add_claim(claim_id="C2", cin="CIN-2", hospital="H-RING", doctor="D-2", anomaly_score=0.8)
    graph.add_claim(claim_id="C3", cin="CIN-3", hospital="H-RING", doctor="D-2", anomaly_score=0.85)
    rings = graph.detect_fraud_rings(anomaly_threshold=0.6, min_claims=3)
    assert len(rings["fraud_rings"]) >= 1
    top = rings["fraud_rings"][0]
    assert set(top.keys()) == {"cluster_id", "nodes", "claims", "risk_score", "reason"}
    assert len(top["claims"]) >= 3


def test_graph_risk_agent_contains_cluster_reuse_and_network_score(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    # Isolate this test from process-wide singleton state.
    orchestrator._fraud_ring_graph = FraudRingGraph()

    claim_base: Dict[str, Any] = {
        "identity": {"cin": "XYZ123", "hospital": "H99", "doctor": "DOC-X"},
        "documents": [{"id": "doc-1", "document_type": "medical_report", "text": "suspicious pattern"}],
        "policy": {},
        "metadata": {"scenario": "fraud", "anomaly_score": 0.92},
    }

    for claim_id in ("ring-1", "ring-2", "ring-3"):
        req = dict(claim_base)
        req["metadata"] = dict(claim_base["metadata"], claim_id=claim_id)
        response = orchestrator.run(req)

    graph_output = next(o for o in response.agent_outputs if o.agent == "GraphRiskAgent")
    ctx = graph_output.input_snapshot["graph_context"]
    assert "cluster_membership" in ctx
    assert "reuse_detection" in ctx
    assert "network_risk_score" in ctx
    assert isinstance(ctx["network_risk_score"], float)
    assert ctx["reuse_detection"]["cin_reuse_detected"] is True
    assert response.blackboard["fraud_ring_analysis"]["fraud_rings"]
    assert response.blackboard["fraud_ring_analysis"]["fraud_rings"][0]["risk_score"] >= 0.0
    assert [c.name for c in SEQUENTIAL_AGENT_CONTRACTS][-1] == "GraphRiskAgent"
