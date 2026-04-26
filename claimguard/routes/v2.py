from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from claimguard.security import AuthContext, verify_request_auth
from claimguard.services.document_extraction import build_extractions_from_base64_parts
from claimguard.v2 import get_v2_orchestrator
from claimguard.v2.blackboard import BlackboardValidationError
from claimguard.v2.redteam import StrictModeConfig, run_red_teaming
from claimguard.v2.schemas import ClaimGuardV2Response, ClaimRequestV2
from claimguard.v2.reliability import get_reliability_store
from claimguard.v2.trust_layer import TrustLayerIPFSFailure

from claimguard.v2.flow_tracker import get_tracker

router = APIRouter(prefix="/v2", tags=["claimguard-v2"])
LOGGER = logging.getLogger("claimguard.routes.v2")


class HumanFeedbackPayload(BaseModel):
    claim_id: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    cin: str = Field(default="")
    provider: str = Field(default="")


class HumanDecisionPayload(BaseModel):
    claim_id: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    notes: str = Field(default="")


def _require_investigator(auth: AuthContext) -> None:
    role = str(auth.role or "").lower()
    if role not in {"admin", "investigator"}:
        raise HTTPException(status_code=403, detail="Investigator privileges required")


def _raise_access_denied(*, claim_id: str, auth: AuthContext, reason: str) -> None:
    # PROD-FIX: structured 403 payload with audit-friendly auth log.
    LOGGER.warning(
        "[AUTH BLOCK] claim_id=%s user=%s role=%s",
        claim_id,
        str(auth.user_id or ""),
        str(auth.role or ""),
    )
    raise HTTPException(
        status_code=403,
        detail={"error": "ACCESS_DENIED", "reason": reason, "claim_id": claim_id},
    )


def _prepare_v2_claim_payload(claim: ClaimRequestV2) -> dict:
    """
    Merge optional inline base64 documents into `document_extractions` before analysis.
    This ensures PDF extraction/OCR runs in v2 the same way as legacy routes.
    """
    payload = claim.model_dump()
    parts = payload.pop("documents_base64", None) or []
    if not parts:
        return payload

    extractions = build_extractions_from_base64_parts(parts)
    payload["document_extractions"] = extractions

    documents = payload.get("documents")
    if not isinstance(documents, list):
        documents = []
    payload["documents"] = documents
    existing_ids = {
        str(item.get("id"))
        for item in documents
        if isinstance(item, dict) and item.get("id")
    }
    for ex in extractions:
        file_name = str(ex.get("file_name") or "").strip()
        if not file_name or file_name in existing_ids:
            continue
        documents.append({"id": file_name, "document_type": "uploaded_file"})
        existing_ids.add(file_name)

    return payload

@router.get("/claim/{claim_id}/flow", dependencies=[Depends(verify_request_auth)])
async def get_claim_flow(claim_id: str) -> dict:
    tracker = get_tracker(claim_id)
    return tracker.get_state()


@router.get("/claim/{claim_id}/proof")
async def get_claim_proof_trace(
    claim_id: str,
    auth: AuthContext = Depends(verify_request_auth),
) -> dict:
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    orchestrator = get_v2_orchestrator()
    payload = orchestrator.get_proof_trace(claim_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No proof trace found for claim {claim_id}")
    return payload


@router.get("/claim/{claim_id}/human-review-context")
async def get_human_review_context(
    claim_id: str,
    auth: AuthContext = Depends(verify_request_auth),
) -> dict:
    orchestrator = get_v2_orchestrator()
    role = str(auth.role or "").lower()
    if role not in {"admin", "investigator"}:
        # PROD-FIX: fallback allows any authenticated role when claim exists.
        if not auth.user_id:
            _raise_access_denied(claim_id=claim_id, auth=auth, reason="Authentication required")
        existing = orchestrator.get_human_review_context(claim_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"No human review payload found for claim {claim_id}")
        return existing
    payload = orchestrator.get_human_review_context(claim_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No human review payload found for claim {claim_id}")
    return payload


@router.get("/claim/{claim_id}/review-document")
async def get_human_review_document(
    claim_id: str,
    file: str = Query(min_length=1),
    token: str = Query(min_length=1),
    expires: int = Query(gt=0),
    auth: AuthContext = Depends(verify_request_auth),
):
    _require_investigator(auth)
    orchestrator = get_v2_orchestrator()
    file_path = orchestrator.resolve_human_review_document(
        claim_id=claim_id,
        file_name=file,
        token=token,
        expires=expires,
    )
    if not file_path:
        raise HTTPException(status_code=403, detail="Invalid review document token or file reference")
    return FileResponse(path=file_path, filename=file)


@router.post("/claim/analyze", response_model=ClaimGuardV2Response, dependencies=[Depends(verify_request_auth)])
async def analyze_claim_v2(claim: ClaimRequestV2) -> ClaimGuardV2Response:
    orchestrator = get_v2_orchestrator()
    try:
        return orchestrator.run(_prepare_v2_claim_payload(claim))
    except BlackboardValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TrustLayerIPFSFailure as exc:
        raise HTTPException(status_code=503, detail=f"Trust layer IPFS upload failed: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("v2 claim analyze failed")
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 analysis failed: {exc}") from exc


@router.get("/debug/fraud-graph", dependencies=[Depends(verify_request_auth)])
async def debug_fraud_graph_v2(render_png: bool = False) -> dict:
    orchestrator = get_v2_orchestrator()
    try:
        return orchestrator.get_fraud_graph_debug(render_png=render_png)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 fraud-graph debug failed: {exc}") from exc


@router.get("/debug/trust-layer-health", dependencies=[Depends(verify_request_auth)])
async def debug_trust_layer_health_v2() -> dict:
    orchestrator = get_v2_orchestrator()
    try:
        return orchestrator.get_trust_layer_health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 trust-layer healthcheck failed: {exc}") from exc


@router.post("/redteam/run", dependencies=[Depends(verify_request_auth)])
async def run_redteam_v2(
    claims: int = 100,
    seed: int = 42,
    artifact_dir: str = "tests/artifacts",
    plots: bool = True,
    simulated_agents: bool = False,
    critical_threshold: int = 5,
    hallucination_max: float = 0.2,
    fraud_target: float = 0.7,
) -> dict:
    """
    Run the full ClaimGuard red teaming engine (with/without memory) and return the report JSON.
    Intended for local stress-testing; in CI prefer `python tests/run_red_teaming.py`.
    """
    strict = StrictModeConfig(
        critical_failures_threshold=critical_threshold,
        hallucination_rate_max=hallucination_max,
        fraud_detection_target=fraud_target,
    )
    try:
        return run_red_teaming(
            claim_count=claims,
            random_seed=seed,
            artifact_dir=artifact_dir,
            generate_visualizations=plots,
            use_simulated_agents=simulated_agents,
            strict_mode=strict,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 redteam run failed: {exc}") from exc


@router.get("/claim/{claim_id}/replay", dependencies=[Depends(verify_request_auth)])
async def replay_claim_v2(claim_id: str) -> dict:
    orchestrator = get_v2_orchestrator()
    try:
        return orchestrator.replay(claim_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 replay failed: {exc}") from exc


@router.post("/claim/human-feedback")
async def submit_human_feedback_v2(
    payload: HumanFeedbackPayload,
    request: Request,
    auth: AuthContext = Depends(verify_request_auth),
) -> dict:
    role = str(auth.role or "").upper()
    if role not in {"INVESTIGATOR", "ADMIN"}:
        raise HTTPException(status_code=403, detail="Unauthorized feedback submission")
    if payload.reviewer_id != str(auth.user_id or ""):
        raise HTTPException(status_code=403, detail="reviewer_id does not match authenticated user")

    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    ip_address = forwarded_for.split(",")[0].strip() if forwarded_for else ""
    if not ip_address:
        ip_address = request.client.host if request.client else ""

    orchestrator = get_v2_orchestrator()
    try:
        return orchestrator.record_human_feedback(
            claim_id=payload.claim_id,
            outcome=payload.outcome,
            reviewer_id=payload.reviewer_id,
            cin=payload.cin,
            provider=payload.provider,
            ip_address=ip_address,
        )
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 feedback failed: {exc}") from exc


@router.post("/claim/human-decision")
async def submit_human_decision_v2(
    payload: HumanDecisionPayload,
    auth: AuthContext = Depends(verify_request_auth),
) -> dict:
    _require_investigator(auth)
    if payload.reviewer_id != (auth.user_id or payload.reviewer_id):
        raise HTTPException(status_code=403, detail="reviewer_id does not match authenticated user")
    orchestrator = get_v2_orchestrator()
    try:
        return orchestrator.apply_human_decision(
            claim_id=payload.claim_id,
            decision=payload.decision,
            reviewer_id=payload.reviewer_id,
            notes=payload.notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TrustLayerIPFSFailure as exc:
        raise HTTPException(status_code=503, detail=f"Trust layer IPFS upload failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 human decision failed: {exc}") from exc


@router.get("/investigator-analytics")
async def get_investigator_analytics_v2(
    investigator_id: str | None = None,
    auth: AuthContext = Depends(verify_request_auth),
) -> dict:
    if auth.role != "admin":
        raise HTTPException(status_code=403, detail="Investigator analytics are restricted to admin users")
    analytics = get_reliability_store().get_investigator_analytics()
    if not investigator_id:
        return analytics
    target = str(investigator_id).strip()
    selected = next(
        (row for row in analytics.get("leaderboard", []) if row.get("investigator_id") == target),
        None,
    )
    if selected is None:
        raise HTTPException(status_code=404, detail=f"No analytics found for investigator {target}")
    return {
        "investigator_id": target,
        "profile": selected.get("profile", {}),
        "metrics": selected.get("metrics", {}),
        "alerts": selected.get("alerts", []),
        "total_reviews": analytics.get("total_reviews", 0),
        "fairness_rules": analytics.get("fairness_rules", {}),
        "feedback_loop_signals": analytics.get("feedback_loop_signals", {}),
    }
