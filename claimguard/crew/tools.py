"""
Deterministic tools wrapping existing rule-based / graph agents.

These tools are the single source of truth for claim scoring — LLM agents must
call them (they must not invent scores).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict

from crewai.tools import BaseTool

from claimguard.agents.anomaly_agent import AnomalyAgent
from claimguard.agents.document_agent import DocumentAgent
from claimguard.agents.graph_agent import GraphAgent
from claimguard.agents.identity_agent import IdentityAgent
from claimguard.agents.pattern_agent import PatternAgent
from claimguard.agents.policy_agent import PolicyAgent


def _canonical_json(payload: Dict[str, Any]) -> str:
    """Stable serialization for tool outputs (deterministic ordering)."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _run_analyzer(claim_json: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> str:
    data = json.loads(claim_json)
    return _canonical_json(fn(data))


class AnomalyClaimTool(BaseTool):
    name: str = "run_anomaly_claim_analysis"
    description: str = (
        "Run deterministic anomaly detection on a health claim. "
        "Input must be the full claim JSON string. Returns JSON with scores and reasoning."
    )

    def _run(self, claim_json: str) -> str:
        return _run_analyzer(claim_json, AnomalyAgent().analyze)


class PatternClaimTool(BaseTool):
    name: str = "run_pattern_claim_analysis"
    description: str = (
        "Run statistical pattern detection on a claim. "
        "Input must be the full claim JSON string. Returns JSON with scores and reasoning."
    )

    def _run(self, claim_json: str) -> str:
        return _run_analyzer(claim_json, PatternAgent().analyze)


class IdentityClaimTool(BaseTool):
    name: str = "run_identity_claim_analysis"
    description: str = (
        "Verify identity consistency for a claim. "
        "Input must be the full claim JSON string. Returns JSON with scores and reasoning."
    )

    def _run(self, claim_json: str) -> str:
        return _run_analyzer(claim_json, IdentityAgent().analyze)


class DocumentClaimTool(BaseTool):
    name: str = "run_document_claim_analysis"
    description: str = (
        "Assess document completeness and authenticity signals. "
        "Input must be the full claim JSON string. Returns JSON with scores and reasoning."
    )

    def _run(self, claim_json: str) -> str:
        return _run_analyzer(claim_json, DocumentAgent().analyze)


class PolicyClaimTool(BaseTool):
    name: str = "run_policy_claim_analysis"
    description: str = (
        "Validate policy / coverage rules for CNSS vs CNOPS. "
        "Input must be the full claim JSON string. Returns JSON with scores and reasoning."
    )

    def _run(self, claim_json: str) -> str:
        return _run_analyzer(claim_json, PolicyAgent().analyze)


class GraphClaimTool(BaseTool):
    name: str = "run_graph_claim_analysis"
    description: str = (
        "Run graph-based fraud scoring (patient–provider–claim relationships). "
        "Input must be the full claim JSON string. Returns JSON with scores and reasoning."
    )

    def _run(self, claim_json: str) -> str:
        return _run_analyzer(claim_json, GraphAgent().analyze)


ALL_CLAIM_TOOLS: tuple[BaseTool, ...] = (
    AnomalyClaimTool(),
    PatternClaimTool(),
    IdentityClaimTool(),
    DocumentClaimTool(),
    PolicyClaimTool(),
    GraphClaimTool(),
)
