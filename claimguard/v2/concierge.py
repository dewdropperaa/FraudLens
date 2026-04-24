from __future__ import annotations

from typing import Any, Dict, List

from claimguard.v2.schemas import RoutingDecision


def classify_intent(identity: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    if metadata.get("intent"):
        return str(metadata["intent"]).strip().lower()
    if identity.get("claimant_type"):
        return f"claim_{str(identity['claimant_type']).strip().lower()}"
    return "general_claim"


def score_document_complexity(documents: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    doc_count = len(documents)
    flagged = bool(metadata.get("manual_review")) or bool(metadata.get("fraud_signal"))
    has_long_doc = any(len(str(d.get("text", ""))) > 2_000 for d in documents)
    if flagged:
        return "high_risk"
    if doc_count <= 1 and not has_long_doc:
        return "simple"
    if doc_count >= 4 or has_long_doc:
        return "complex"
    return "simple"


def choose_model(complexity: str) -> str:
    mapping = {
        "simple": "llama3",
        "complex": "deepseek-r1",
        "high_risk": "deepseek-r1",
    }
    return mapping[complexity]


def build_routing_decision(claim_request: Dict[str, Any]) -> RoutingDecision:
    identity = claim_request.get("identity", {})
    documents = claim_request.get("documents", [])
    metadata = claim_request.get("metadata", {})
    intent = classify_intent(identity, metadata)
    complexity = score_document_complexity(documents, metadata)
    model = choose_model(complexity)
    return RoutingDecision(
        intent=intent,
        complexity=complexity,
        model=model,
        reason=f"intent={intent}, complexity={complexity}",
        metadata={"document_count": len(documents)},
    )
