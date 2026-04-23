from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.v2.orchestrator import ClaimGuardV2Orchestrator


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


def _extract_json_block(prefix: str, prompt: str) -> Dict[str, Any]:
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :].strip())
    return {}


class _FieldAwareCrew:
    def __init__(self, tasks: List[_FakeTask], **_: Any) -> None:
        self._task = tasks[0]

    def kickoff(self) -> str:
        claim = _extract_json_block("Claim: ", self._task.description)
        data = claim.get("data", {}) if isinstance(claim, dict) else {}
        has_verified = bool(data)
        payload = {
            "score": 0.86 if has_verified else 0.42,
            "confidence": 0.88 if has_verified else 0.36,
            "status": "VERIFIED" if has_verified else "INSUFFICIENT_DATA",
            "claims": [
                {
                    "statement": "Used verified claim fields only",
                    "evidence": next(iter(data.values()), "UNVERIFIED"),
                    "verified": has_verified,
                }
            ],
            "hallucination_flags": [],
            "explanation": (
                f"Verified fields={list(data.keys())}; unverified fields treated as missing."
                if has_verified
                else "No verified structured fields available; treated as missing."
            ),
        }
        return json.dumps(payload)


def _patch_orchestrator(monkeypatch) -> None:
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(module, "Agent", _FakeAgent)
    monkeypatch.setattr(module, "Task", _FakeTask)
    monkeypatch.setattr(module, "Crew", _FieldAwareCrew)
    monkeypatch.setattr(module, "assert_ollama_connection", lambda: None)


def _base_claim() -> Dict[str, Any]:
    return {
        "identity": {
            "cin": "AB123456",
            "name": "Sara Benali",
            "hospital": "Hopital Atlas",
            "doctor": "Dr Amrani",
        },
        "policy": {
            "amount": 1200,
            "hospital": "Hopital Atlas",
            "diagnosis": "consultation",
        },
        "metadata": {
            "claim_id": "field-verification-test",
            "service_date": "2026-01-10",
            "amount": 1200,
            "hospital": "Hopital Atlas",
            "doctor": "Dr Amrani",
        },
        "patient_id": "AB123456",
        "amount": 1200,
    }


def test_valid_document_matching_fields_passes(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    claim = _base_claim()
    text = (
        "Facture medicale. CIN AB123456. Montant 1200 MAD. "
        "Date 2026-01-10. Hopital Atlas. Consultation."
    )
    claim["documents"] = [{"id": "doc-1", "document_type": "invoice", "text": text}]
    claim["document_extractions"] = [{"file_name": "invoice.pdf", "extracted_text": text}]
    response = orchestrator.run(claim)
    assert response.decision in {"APPROVED", "HUMAN_REVIEW"}
    assert any(row.get("verified") for row in response.blackboard.get("field_verification", []))


def test_random_document_with_fake_fields_is_rejected(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    claim = _base_claim()
    text = "shopping receipt electronics 88.90 usd order id 4459"
    claim["documents"] = [{"id": "doc-1", "document_type": "txt", "text": text}]
    claim["document_extractions"] = [{"file_name": "receipt.txt", "extracted_text": text}]
    response = orchestrator.run(claim)
    assert response.decision == "REJECTED"
    assert response.Ts == 0.0
    assert len(response.agent_outputs) == 0
    assert response.decision_trace is not None
    assert "critical_failures" in response.decision_trace
    assert any(
        flag in response.decision_trace["critical_failures"]
        for flag in ("CRITICAL_FIELD_CIN_NOT_FOUND", "DOCUMENT_CLASSIFIED_NON_CLAIM")
    )
    assert "CRITICAL_FIELD_AMOUNT_NOT_FOUND" in response.blackboard.get("critical_failures", [])


def test_partial_field_match_forces_human_review(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    claim = _base_claim()
    # CIN + provider are present; amount/date are missing from OCR.
    text = "Facture medicale. CIN AB123456. Hopital Atlas. Consultation."
    claim["documents"] = [{"id": "doc-1", "document_type": "invoice", "text": text}]
    claim["document_extractions"] = [{"file_name": "invoice.pdf", "extracted_text": text}]
    response = orchestrator.run(claim)
    assert response.decision in {"HUMAN_REVIEW", "REJECTED"}
    assert response.decision != "APPROVED"


def test_missing_input_fields_do_not_raise_critical_not_found(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    claim = _base_claim()
    claim["identity"]["cin"] = ""
    claim["patient_id"] = ""
    claim["amount"] = ""
    claim["metadata"]["amount"] = ""
    claim["policy"]["amount"] = ""
    text = "Facture medicale. Hopital Atlas. Date 2026-01-10."
    claim["documents"] = [{"id": "doc-1", "document_type": "invoice", "text": text}]
    claim["document_extractions"] = [{"file_name": "invoice.pdf", "extracted_text": text}]
    response = orchestrator.run(claim)

    critical_failures = response.blackboard.get("critical_failures", [])
    assert "CRITICAL_FIELD_CIN_NOT_FOUND" not in critical_failures
    assert "CRITICAL_FIELD_AMOUNT_NOT_FOUND" not in critical_failures

