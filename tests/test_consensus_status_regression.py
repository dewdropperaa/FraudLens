from __future__ import annotations

from claimguard.v2.consensus import ConsensusConfig, ConsensusEngine


def test_consensus_done_status_entries_do_not_trigger_error_mode() -> None:
    engine = ConsensusEngine()
    entries = {
        "IdentityAgent": {"status": "DONE", "score": 0.91, "confidence": 0.9, "explanation": "ok"},
        "DocumentAgent": {"status": "DONE", "score": 0.86, "confidence": 0.88, "explanation": "ok"},
        "PolicyAgent": {"status": "DONE", "score": 0.84, "confidence": 0.86, "explanation": "ok"},
        "AnomalyAgent": {"status": "DONE", "score": 0.42, "confidence": 0.8, "explanation": "normal"},
        "PatternAgent": {"status": "DONE", "score": 0.45, "confidence": 0.78, "explanation": "normal"},
        "GraphRiskAgent": {"status": "DONE", "score": 0.40, "confidence": 0.8, "explanation": "normal"},
    }
    result = engine.evaluate(
        claim_request={"metadata": {}, "policy": {}, "identity": {}},
        entries=entries,
        blackboard={"memory_degraded": False, "memory_status": "OK"},
        config=ConsensusConfig(),
    )
    assert result["too_many_error_agents"] is False
    assert result["Ts"] > 0.0
    assert result["decision"] in {"APPROVED", "HUMAN_REVIEW", "REJECTED"}
