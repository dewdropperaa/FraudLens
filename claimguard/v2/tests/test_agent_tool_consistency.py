from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from claimguard.agents.anomaly_agent import AnomalyAgent
from claimguard.agents.base_agent import BaseAgent
from claimguard.agents.document_agent import DocumentAgent
from claimguard.agents.graph_agent import GraphAgent
from claimguard.agents.identity_agent import IdentityAgent
from claimguard.agents.pattern_agent import PatternAgent
from claimguard.agents.policy_agent import PolicyAgent
from claimguard.agents.validation_agent import ClaimValidationAgent


def _sample_claim() -> Dict[str, Any]:
    return {
        "patient_id": "AB123456",
        "amount": 1200,
        "insurance": "CNSS",
        "documents": ["medical_report.pdf", "invoice.pdf", "prescription.pdf"],
        "document_extractions": [
            {
                "file_name": "invoice.pdf",
                "extracted_text": "Nom complet: Test User\nCIN: AB123456\nDate de naissance: 01/01/1990",
            }
        ],
        "history": [{"amount": 1000, "date": "2025-10-10"}],
        "identity": {"cin": "AB123456"},
        "policy": {"diagnosis": "consultation"},
    }


def _tool_stub(tool_name: str, _input_data: Dict[str, Any]) -> Dict[str, Any]:
    outputs: Dict[str, Dict[str, Any]] = {
        "ocr_extractor": {
            "extractions": [{"file_name": "invoice.pdf", "extracted_text": "ok"}],
            "extraction_count": 1,
            "empty_extractions": 0,
            "total_text_len": 120,
        },
        "document_classifier": {
            "document_type": "medical_claim_bundle",
            "missing_docs": [],
            "found_docs": ["medical_report", "invoice", "prescription"],
            "insurance_doc_only": False,
        },
        "fraud_detector": {
            "fraud_text_signals": [],
            "suspicious_doc_names": [],
            "high_amount_doc_requirement": False,
            "risk_indicators": 0,
        },
        "identity_extractor": {
            "name": "Test User",
            "cin": "AB123456",
            "dob": "01/01/1990",
        },
    }
    return {
        "tool": tool_name,
        "status": "DONE",
        "output": outputs.get(tool_name, {"ok": True}),
        "confidence": 0.9,
    }


@pytest.mark.parametrize(
    "agent_cls",
    [
        IdentityAgent,
        PatternAgent,
        AnomalyAgent,
        PolicyAgent,
        ClaimValidationAgent,
        GraphAgent,
        DocumentAgent,
    ],
)
def test_tool_first_contract_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    agent_cls: type[BaseAgent],
) -> None:
    tool_calls: List[str] = []
    llm_calls: List[Tuple[str, str]] = []

    def _execute(self: BaseAgent, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        tool_calls.append(tool_name)
        return _tool_stub(tool_name, input_data)

    def _llm_stub(*, agent_name: str, claim_data: Dict[str, Any], draft_reasoning: str) -> Tuple[str, Dict[str, Any]]:
        llm_calls.append((agent_name, draft_reasoning))
        return draft_reasoning, {"agent_type": "LLM_AGENT", "llm_calls": 1}

    monkeypatch.setattr(BaseAgent, "execute_tool", _execute)
    monkeypatch.setattr("claimguard.agents.document_agent.run_agent_consistency_check", _llm_stub)
    monkeypatch.setattr("claimguard.agents.pattern_agent.run_agent_consistency_check", _llm_stub)
    monkeypatch.setattr("claimguard.agents.anomaly_agent.run_agent_consistency_check", _llm_stub)
    monkeypatch.setattr("claimguard.agents.policy_agent.run_agent_consistency_check", _llm_stub)
    monkeypatch.setattr("claimguard.agents.graph_agent.run_agent_consistency_check", _llm_stub)

    agent = agent_cls()
    result = agent.safe_run(_sample_claim())

    assert result["status"] == "DONE"
    assert isinstance(result["output"], dict) and result["output"]
    assert 0.0 <= float(result["score"]) <= 100.0

    assert tool_calls, "agent does not call any tool"
    assert isinstance(result["tools_used"], list) and result["tools_used"], "DONE without tools_used"
    assert result["tool_policy"]["tool_first"] is True
    assert result["tool_policy"]["llm_fallback_used"] is False
    assert not llm_calls, "LLM used without fallback trigger"


def test_llm_fallback_requires_tool_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_calls: List[str] = []
    llm_calls: List[str] = []

    def _execute(self: BaseAgent, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        tool_calls.append(tool_name)
        result = _tool_stub(tool_name, input_data)
        result["confidence"] = 0.2
        return result

    def _llm_stub(*, agent_name: str, claim_data: Dict[str, Any], draft_reasoning: str) -> Tuple[str, Dict[str, Any]]:
        llm_calls.append(agent_name)
        return draft_reasoning, {"agent_type": "LLM_AGENT", "llm_calls": 1}

    monkeypatch.setattr(BaseAgent, "execute_tool", _execute)
    monkeypatch.setattr("claimguard.agents.document_agent.run_agent_consistency_check", _llm_stub)

    agent = DocumentAgent()
    result = agent.safe_run(_sample_claim())

    assert tool_calls, "fallback path skipped tool attempt"
    assert llm_calls, "fallback was not triggered when tool confidence was low"
    assert result["tool_policy"]["llm_fallback_used"] is True
