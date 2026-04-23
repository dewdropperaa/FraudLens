from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.v2.orchestrator import ClaimGuardV2Orchestrator
from claimguard.v2.consensus import ConsensusConfig, should_force_human_review
from claimguard.v2.schemas import AgentOutput


class _NoOpTrustLayer:
    def process_if_applicable(self, **_: Any) -> None:
        return None


class _MemoryDisabledLayer:
    _using_fake_embeddings = True

    def retrieve_similar_cases(self, *_: Any, **__: Any) -> List[Dict[str, Any]]:
        return []

    def store_case(self, *_: Any, **__: Any) -> None:
        return None


class _FakeAgent:
    def __init__(self, role: str, **_: Any) -> None:
        self.role = role


class _FakeTask:
    def __init__(self, description: str, agent: _FakeAgent, **_: Any) -> None:
        self.description = description
        self.agent = agent


def _extract_json_block(prefix: str, prompt: str) -> Dict[str, Any]:
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :].strip())
    return {}


class _HallucinatingCrew:
    def __init__(self, tasks: List[_FakeTask], **_: Any) -> None:
        self._task = tasks[0]

    def kickoff(self) -> str:
        role = self._task.agent.role
        claim = _extract_json_block("Claim: ", self._task.description)
        text = str(claim.get("text", ""))
        grounded = {
            "score": 0.7,
            "confidence": 0.8,
            "explanation": "CIN AB123456 and amount 2000 are present in OCR text.",
            "claims": [
                {"statement": "Claim has amount 2000", "evidence": "2000", "verified": True},
            ],
            "hallucination_flags": [],
        }
        if role in {"IdentityAgent", "DocumentAgent"}:
            return json.dumps(
                {
                    "score": 0.92,
                    "confidence": 0.95,
                    "explanation": "Invoice contains amount 999999 and provider Moon Hospital.",
                    "claims": [
                        {"statement": "Amount is 999999", "evidence": "999999", "verified": True},
                        {"statement": "Provider is Moon Hospital", "evidence": "Moon Hospital", "verified": True},
                    ],
                    "hallucination_flags": [],
                }
            )
        grounded["claims"][0]["evidence"] = "2000" if "2000" in text else "UNVERIFIED"
        return json.dumps(grounded)


class _HighConfidencePatternCrew:
    def __init__(self, tasks: List[_FakeTask], **_: Any) -> None:
        self._task = tasks[0]

    def kickoff(self) -> str:
        role = self._task.agent.role
        return json.dumps(
            {
                "score": 0.95 if role in {"PatternAgent", "GraphRiskAgent"} else 0.75,
                "confidence": 0.98,
                "explanation": f"{role} high-confidence analysis.",
                "claims": [{"statement": f"{role} signal", "evidence": "2000", "verified": True}],
                "hallucination_flags": [],
            }
        )


class _SingleHallucinatingCrew:
    def __init__(self, tasks: List[_FakeTask], **_: Any) -> None:
        self._task = tasks[0]

    def kickoff(self) -> str:
        role = self._task.agent.role
        if role == "IdentityAgent":
            return json.dumps(
                {
                    "score": 0.82,
                    "confidence": 0.91,
                    "explanation": "Identity output includes a self-reported hallucination warning.",
                    "claims": [
                        {"statement": "Claim amount is 2000", "evidence": "2000", "verified": True},
                    ],
                    "hallucination_flags": ["self_reported_hallucination"],
                }
            )
        return json.dumps(
            {
                "score": 0.75,
                "confidence": 0.85,
                "explanation": f"{role} grounded analysis from OCR.",
                "claims": [{"statement": f"{role} evidence", "evidence": "2000", "verified": True}],
                "hallucination_flags": [],
            }
        )


class _LowConfidenceUngroundedCrew:
    def __init__(self, tasks: List[_FakeTask], **_: Any) -> None:
        self._task = tasks[0]

    def kickoff(self) -> str:
        role = self._task.agent.role
        if role == "IdentityAgent":
            return json.dumps(
                {
                    "score": 0.62,
                    "confidence": 0.35,
                    "explanation": "Identity signal is weak and lacks grounded evidence.",
                    "claims": [{"statement": "Unknown ID", "evidence": "", "verified": False}],
                    "hallucination_flags": [],
                }
            )
        return json.dumps(
            {
                "score": 0.74,
                "confidence": 0.82,
                "explanation": f"{role} grounded analysis.",
                "claims": [{"statement": f"{role} evidence", "evidence": "2000", "verified": True}],
                "hallucination_flags": [],
            }
        )


def _patch_orchestrator(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _HallucinatingCrew)
    monkeypatch.setattr(module, "assert_ollama_connection", lambda: None)


def _patch_orchestrator_high_confidence(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _HighConfidencePatternCrew)
    monkeypatch.setattr(module, "assert_ollama_connection", lambda: None)


def _patch_orchestrator_single_hallucination(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _SingleHallucinatingCrew)
    monkeypatch.setattr(module, "assert_ollama_connection", lambda: None)


def _patch_orchestrator_low_confidence_ungrounded(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _LowConfidenceUngroundedCrew)
    monkeypatch.setattr(module, "assert_ollama_connection", lambda: None)


def _make_claim() -> Dict[str, Any]:
    text = (
        "Facture medicale consultation. Patient Sara Benali CIN AB123456 "
        "Hopital Atlas. Date 2026-01-10. Montant total 2000 MAD."
    )
    return {
        "identity": {"cin": "AB123456", "name": "Sara Benali", "hospital": "Hopital Atlas", "doctor": "Dr Amrani"},
        "documents": [{"id": "doc-1", "document_type": "invoice", "text": text}],
        "document_extractions": [{"file_name": "claim.pdf", "extracted_text": text}],
        "policy": {"amount": 2000, "hospital": "Hopital Atlas", "diagnosis": "consultation"},
        "metadata": {"claim_id": "hallucination-check", "service_date": "2026-01-10", "hospital": "Hopital Atlas", "doctor": "Dr Amrani"},
        "patient_id": "AB123456",
        "amount": 2000,
    }


def test_hallucinated_values_are_flagged_and_penalized(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    response = orchestrator.run(_make_claim())

    identity = next(a for a in response.agent_outputs if a.agent == "IdentityAgent")
    assert "unsupported_claim" in identity.hallucination_flags
    assert "ocr_value_not_found" in identity.hallucination_flags
    assert identity.hallucination_penalty >= 0.3
    assert identity.confidence < 0.95


def test_missing_or_unverified_claim_evidence_is_not_silent(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    response = orchestrator.run(_make_claim())

    doc = next(a for a in response.agent_outputs if a.agent == "DocumentAgent")
    assert any(claim.verified is False for claim in doc.claims)
    assert any(claim.evidence == "UNVERIFIED" or claim.evidence for claim in doc.claims)


def test_two_hallucinating_agents_force_human_review(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    response = orchestrator.run(_make_claim())
    result = {
        "decision": response.decision,
        "flags": response.blackboard.get("flags", {}),
        "Ts": response.Ts,
    }
    config = orchestrator._consensus_config

    assert response.blackboard.get("force_human_review") is True
    assert result["decision"] == "HUMAN_REVIEW"
    assert "hallucination_force_reason" in result["flags"]
    assert result["Ts"] < config.auto_approve_threshold


def test_one_hallucinating_agent_does_not_force() -> None:
    config = ConsensusConfig()
    outputs = [
        AgentOutput(
            agent="IdentityAgent",
            score=0.8,
            confidence=0.9,
            claims=[{"statement": "Grounded claim", "evidence": "2000", "verified": True}],
            hallucination_flags=["self_reported_hallucination"],
            explanation="Identity issued one hallucination warning.",
            hallucination_penalty=0.0,
            elapsed_ms=5,
            input_snapshot={},
            output_snapshot={},
        ),
        AgentOutput(
            agent="DocumentAgent",
            score=0.76,
            confidence=0.88,
            claims=[{"statement": "Grounded claim", "evidence": "AB123456", "verified": True}],
            hallucination_flags=[],
            explanation="Document evidence is grounded.",
            hallucination_penalty=0.0,
            elapsed_ms=5,
            input_snapshot={},
            output_snapshot={},
        ),
    ]

    force, reason = should_force_human_review(
        agent_outputs=outputs,
        blackboard={"contradictions": []},
        config=config,
    )
    assert force is False
    assert "no force condition met" in reason.lower()


def test_high_contradiction_penalty_forces_review(monkeypatch) -> None:
    _patch_orchestrator_high_confidence(monkeypatch)
    from claimguard.v2 import orchestrator as module

    original_evaluate = module.ConsensusEngine.evaluate

    def _high_contradiction_evaluate(self, *, claim_request: Dict[str, Any], entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        result = original_evaluate(self, claim_request=claim_request, entries=entries)
        result["decision"] = "APPROVED"
        result["Ts"] = 95.0
        result["contradictions"] = [
            {
                "agents": ["IdentityAgent", "DocumentAgent"],
                "H_penalty": 0.45,
                "reason": "Injected contradiction for regression coverage.",
            }
        ]
        return result

    monkeypatch.setattr(module.ConsensusEngine, "evaluate", _high_contradiction_evaluate)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    response = orchestrator.run(_make_claim())

    assert response.blackboard.get("force_human_review") is True
    assert response.decision == "HUMAN_REVIEW"
    assert "hallucination_force_reason" in response.blackboard.get("flags", {})


def test_low_confidence_ungrounded_agent_forces_review(monkeypatch) -> None:
    _patch_orchestrator_low_confidence_ungrounded(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    response = orchestrator.run(_make_claim())

    assert response.blackboard.get("force_human_review") is True
    assert response.decision == "HUMAN_REVIEW"
    assert "hallucination_force_reason" in response.blackboard.get("flags", {})


def test_memory_degraded_forces_human_review_and_caps_pattern_confidence(monkeypatch) -> None:
    _patch_orchestrator_high_confidence(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(
        trust_layer_service=_NoOpTrustLayer(),
        memory_layer=_MemoryDisabledLayer(),
    )
    response = orchestrator.run(_make_claim())

    pattern = next(a for a in response.agent_outputs if a.agent == "PatternAgent")
    graph = next(a for a in response.agent_outputs if a.agent == "GraphRiskAgent")
    assert response.blackboard.get("memory_status") in {"DEGRADED", "DISABLED"}
    assert "MEMORY_DEGRADED" in response.system_flags
    assert response.decision == "HUMAN_REVIEW"
    assert pattern.confidence <= 0.45
    assert graph.confidence <= 0.45
