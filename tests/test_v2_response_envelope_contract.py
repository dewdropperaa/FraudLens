from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.v2.orchestrator import (
    ClaimGuardV2Orchestrator,
    ContractViolationError,
    build_response_envelope,
)


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
        return json.dumps(
            {
                "score": 0.72,
                "confidence": 0.88,
                "explanation": "Grounded response.",
                "claims": [{"statement": "grounded", "evidence": "AB123456", "verified": True}],
                "hallucination_flags": [],
                "status": "VERIFIED",
            }
        )


def _patch_orchestrator(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _FakeCrew)
    monkeypatch.setattr(module, "assert_ollama_connection", lambda: None)


def _make_claim(claim_id: str = "env-contract") -> Dict[str, Any]:
    text = "Facture medicale. Patient Sara Benali CIN AB123456. Montant 2000 MAD."
    return {
        "identity": {"cin": "AB123456", "name": "Sara Benali", "hospital": "Hopital Atlas", "doctor": "Dr Amrani"},
        "documents": [{"id": "doc-1", "document_type": "invoice", "text": text}],
        "document_extractions": [{"file_name": "claim.pdf", "extracted_text": text}],
        "policy": {"amount": 2000, "hospital": "Hopital Atlas", "diagnosis": "consultation"},
        "metadata": {"claim_id": claim_id, "service_date": "2026-01-10", "hospital": "Hopital Atlas", "doctor": "Dr Amrani"},
        "patient_id": "AB123456",
        "amount": 2000,
    }


def _hard_gate_validation(*_: Any, **__: Any) -> Dict[str, Any]:
    return {
        "validation_status": "INVALID",
        "validation_score": 0,
        "document_type": "unknown",
        "missing_fields": ["cin"],
        "found_fields": [],
        "reason": "hard gate",
        "should_stop_pipeline": True,
        "details": {},
    }


@pytest.mark.parametrize(
    ("exit_case", "expected_reason"),
    [
        ("non_claim_detected", "non_claim_detected"),
        ("ocr_unreadable", "ocr_unreadable"),
        ("critical_fields_unverified", "critical_fields_unverified"),
        ("claim_validation_hard_gate", "claim_validation_hard_gate"),
    ],
)
def test_all_exit_paths_return_valid_envelope(monkeypatch, exit_case: str, expected_reason: str) -> None:
    _patch_orchestrator(monkeypatch)
    from claimguard.v2 import orchestrator as module

    if exit_case == "non_claim_detected":
        monkeypatch.setattr(module.ClaimGuardV2Orchestrator, "_collect_critical_failures", lambda *a, **k: [])
        monkeypatch.setattr(module, "classify_document", lambda *a, **k: {"label": "NON_CLAIM", "confidence": 60})
    elif exit_case == "ocr_unreadable":
        monkeypatch.setattr(module.ClaimGuardV2Orchestrator, "_collect_critical_failures", lambda *a, **k: [])
        monkeypatch.setattr(module, "classify_document", lambda *a, **k: {"label": "CLAIM", "confidence": 90})
        monkeypatch.setattr(module.ClaimGuardV2Orchestrator, "_run_pre_validation_guard", lambda *a, **k: {"failed": False, "flags": []})
    elif exit_case == "critical_fields_unverified":
        monkeypatch.setattr(module.ClaimGuardV2Orchestrator, "_collect_critical_failures", lambda *a, **k: ["CRITICAL_FIELD_CIN_NOT_FOUND"])
        monkeypatch.setattr(module, "classify_document", lambda *a, **k: {"label": "CLAIM", "confidence": 90})
        monkeypatch.setattr(module.ClaimGuardV2Orchestrator, "_run_pre_validation_guard", lambda *a, **k: {"failed": False, "flags": []})
    elif exit_case == "claim_validation_hard_gate":
        monkeypatch.setattr(module.ClaimGuardV2Orchestrator, "_collect_critical_failures", lambda *a, **k: [])
        monkeypatch.setattr(module, "classify_document", lambda *a, **k: {"label": "CLAIM", "confidence": 90})
        monkeypatch.setattr(module.ClaimGuardV2Orchestrator, "_run_pre_validation_guard", lambda *a, **k: {"failed": False, "flags": []})
        monkeypatch.setattr(module.ClaimValidationAgent, "analyze", _hard_gate_validation)

    claim = _make_claim(f"env-{exit_case}")
    if exit_case == "ocr_unreadable":
        claim["documents"] = [{"id": "doc-1", "document_type": "invoice", "text": ""}]
        claim["document_extractions"] = [{"file_name": "claim.pdf", "extracted_text": ""}]

    response = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer()).run(claim)
    envelope = response.response_envelope or {}

    assert envelope.get("decision") in {"APPROVED", "HUMAN_REVIEW", "REJECTED"}
    assert "Ts" in envelope
    assert envelope.get("exit_reason") == expected_reason
    assert isinstance(envelope.get("blackboard_snapshot"), dict)
    assert isinstance(envelope.get("timestamp_utc"), str)
    assert envelope.get("claim_id") == claim["metadata"]["claim_id"]
    assert isinstance(envelope.get("score_evolution"), list)
    assert isinstance(envelope.get("reflexive_retry_logs"), list)
    assert isinstance(envelope.get("flags"), dict)


def test_contract_violation_on_invalid_decision_string() -> None:
    with pytest.raises(ContractViolationError):
        build_response_envelope(
            decision="INVALID_DECISION",
            Ts=0.0,
            blackboard={"entries": {}},
            exit_reason="bad",
            claim_id="bad-1",
        )


def test_early_exit_envelope_has_empty_not_null_lists(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(module.ClaimGuardV2Orchestrator, "_collect_critical_failures", lambda *a, **k: ["CRITICAL_FIELD_CIN_NOT_FOUND"])
    response = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer()).run(_make_claim("env-empty-lists"))
    envelope = response.response_envelope or {}

    assert envelope.get("score_evolution") == []
    assert envelope.get("reflexive_retry_logs") == []
    assert envelope.get("flags") is not None


def test_response_envelope_schema_snapshot(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    response = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer()).run(_make_claim("env-snapshot"))
    envelope = response.response_envelope or {}
    assert set(envelope.keys()) == {
        "decision",
        "Ts",
        "exit_reason",
        "blackboard_snapshot",
        "timestamp_utc",
        "claim_id",
        "score_evolution",
        "reflexive_retry_logs",
        "flags",
        "agent_outputs",
    }
