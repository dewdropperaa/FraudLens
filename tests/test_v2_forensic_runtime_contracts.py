from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.v2.orchestrator import ClaimGuardV2Orchestrator, SEQUENTIAL_AGENT_CONTRACTS


SNAPSHOT_VERSION = "v1"
SNAPSHOT_PATH = _ROOT / "tests" / "artifacts" / f"v2_contract_snapshots.{SNAPSHOT_VERSION}.json"


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
                "explanation": f"{self._task.agent.role} grounded evidence from OCR and claim fields.",
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


def _make_claim(*, claim_id: str, forensic_debug: bool = False) -> Dict[str, Any]:
    text = (
        "Facture medicale consultation. Patient Sara Benali CIN AB123456. "
        "Hopital Atlas. Date 2026-01-10. Montant total 2000 MAD."
    )
    metadata: Dict[str, Any] = {
        "claim_id": claim_id,
        "service_date": "2026-01-10",
        "hospital": "Hopital Atlas",
        "doctor": "Dr Amrani",
    }
    if forensic_debug:
        metadata["forensic_debug"] = True
        metadata["forensic_input_id"] = f"{claim_id}-forensic"
    return {
        "identity": {"cin": "AB123456", "name": "Sara Benali", "hospital": "Hopital Atlas", "doctor": "Dr Amrani"},
        "documents": [{"id": "doc-1", "document_type": "invoice", "text": text}],
        "document_extractions": [{"file_name": "claim.pdf", "extracted_text": text}],
        "policy": {"amount": 2000, "hospital": "Hopital Atlas", "diagnosis": "consultation"},
        "metadata": metadata,
        "patient_id": "AB123456",
        "amount": 2000,
    }


def _load_snapshot() -> Dict[str, Any]:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _runtime_projection(response) -> Dict[str, Any]:
    blackboard = response.blackboard
    return {
        "decision": response.decision,
        "forensic_trace_present": response.forensic_trace is not None,
        "system_flags_contains_broken": "BROKEN" in [str(flag).upper() for flag in response.system_flags],
        "entries_count": len(blackboard.get("entries", {})),
        "entry_agents": sorted(list(blackboard.get("entries", {}).keys())),
        "decision_trace_present": bool(response.decision_trace),
    }


def _early_rejection_projection(response) -> Dict[str, Any]:
    blackboard = response.blackboard
    return {
        "decision": response.decision,
        "ts": response.Ts,
        "pre_validation_result_present": response.pre_validation_result is not None,
        "agent_outputs_count": len(response.agent_outputs),
        "forensic_trace_present": response.forensic_trace is not None,
        "blackboard_keys": sorted(list(blackboard.keys())),
    }


def test_forensic_mode_is_decoupled_from_runtime(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    response = orchestrator.run(_make_claim(claim_id="runtime-normal-001", forensic_debug=False))

    assert response.forensic_trace is None
    assert "BROKEN" not in [str(flag).upper() for flag in response.system_flags]
    assert set(response.blackboard.get("entries", {}).keys()) == {
        contract.name for contract in SEQUENTIAL_AGENT_CONTRACTS
    }

    snapshots = _load_snapshot()
    assert _runtime_projection(response) == snapshots["runtime_normal"]


def test_forensic_mode_can_be_explicitly_enabled(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    response = orchestrator.run(_make_claim(claim_id="runtime-forensic-001", forensic_debug=True))

    assert response.forensic_trace is not None
    assert response.forensic_trace.get("input_differentiation_test", {}).get("executed") is True
    assert set(response.blackboard.get("entries", {}).keys()) == {
        contract.name for contract in SEQUENTIAL_AGENT_CONTRACTS
    }

    snapshots = _load_snapshot()
    assert _runtime_projection(response) == snapshots["runtime_forensic_enabled"]


def test_early_rejection_contract_snapshot(monkeypatch) -> None:
    _patch_orchestrator(monkeypatch)
    from claimguard.v2 import orchestrator as module

    monkeypatch.setattr(
        module,
        "classify_document",
        lambda *_args, **_kwargs: {
            "label": "NON_CLAIM",
            "confidence": 33,
            "source": "test",
            "features": {},
            "raw_prediction": "NON_CLAIM",
        },
    )
    orchestrator = ClaimGuardV2Orchestrator(trust_layer_service=_NoOpTrustLayer())
    response = orchestrator.run(_make_claim(claim_id="runtime-early-reject-001"))

    assert response.decision == "REJECTED"
    assert response.pre_validation_result is not None
    assert len(response.agent_outputs) == 0

    snapshots = _load_snapshot()
    assert _early_rejection_projection(response) == snapshots["early_rejection_non_claim"]
