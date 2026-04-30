from __future__ import annotations

import logging
from datetime import datetime, timezone

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
from claimguard.models import AgentResult, ClaimResult
from claimguard.services.storage import get_claim_store
from claimguard.services import doc_store

from claimguard.v2.flow_tracker import get_tracker

router = APIRouter(prefix="/v2", tags=["claimguard-v2"])
LOGGER = logging.getLogger("claimguard.routes.v2")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if role != "admin":
        if not auth.user_id:
            _raise_access_denied(claim_id=claim_id, auth=auth, reason="Authentication required")
        existing = orchestrator.get_human_review_context(claim_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"No human review payload found for claim {claim_id}")
        if not existing.get("document_url"):
            if doc_store.get_document_path(claim_id):
                existing = dict(existing)
                existing["document_url"] = f"/api/v2/claim/{claim_id}/local-document"
        return existing

    payload = orchestrator.get_human_review_context(claim_id)

    # When orchestrator context is lost (server restart, missing Firestore)
    # but the document is still on disk, synthesize a minimal context so the
    # admin can still view and decide on the claim.
    if payload is None:
        local_path = doc_store.get_document_path(claim_id)
        if local_path:
            LOGGER.info(
                "human_review_context_rebuilt_from_doc_store claim_id=%s",
                claim_id,
            )
            payload = {
                "claim_id": claim_id,
                "ts": 0.0,
                "reason": "Review context recovered from local document store",
                "document_url": f"/api/v2/claim/{claim_id}/local-document",
                "extracted_data": {},
                "agent_breakdown": [],
                "heatmap": [],
                "heatmap_fallback": [],
                "heatmap_status": "missing_pdf",
                "pipeline_version": "v2",
                "ai_suggested_decision": "HUMAN_REVIEW",
                "risk_breakdown": {},
            }
        else:
            # Also check the claim store — the claim may have been stored
            # but the review context was lost.
            stored = get_claim_store().get(claim_id)
            if stored is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No human review payload found for claim {claim_id}",
                )
            payload = {
                "claim_id": claim_id,
                "ts": float(stored.Ts or stored.score or 0.0),
                "reason": "Review context recovered from claim store",
                "document_url": None,
                "extracted_data": {},
                "agent_breakdown": [],
                "heatmap": [],
                "heatmap_fallback": [],
                "heatmap_status": "missing_pdf",
                "pipeline_version": "v2",
                "ai_suggested_decision": "HUMAN_REVIEW",
                "risk_breakdown": {},
            }

    # _prepare_v2_claim_payload strips documents_base64 before the orchestrator
    # runs, so document_url is often None.  Fall back to the locally-stored
    # file saved at submission time.
    if not payload.get("document_url"):
        if doc_store.get_document_path(claim_id):
            payload = dict(payload)
            payload["document_url"] = f"/api/v2/claim/{claim_id}/local-document"

    LOGGER.debug(
        "human_review_context claim_id=%s document_url=%s",
        claim_id,
        payload.get("document_url"),
    )
    return payload


@router.get("/claim/{claim_id}/review-document")
async def get_human_review_document(
    claim_id: str,
    file: str = Query(min_length=1),
    token: str = Query(min_length=1),
    expires: int = Query(gt=0),
    auth: AuthContext = Depends(verify_request_auth),
):
    if str(auth.role or "").lower() != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
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


@router.get("/claim/{claim_id}/local-document")
async def get_local_review_document(
    claim_id: str,
):
    """Serve a locally-stored document for a Human Review claim (temporarily public)."""
    file_path = doc_store.get_document_path(claim_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="No document stored for this claim")
    filename = doc_store.get_document_name(claim_id) or "document"
    media_type = "application/pdf" if filename.lower().endswith(".pdf") else None
    return FileResponse(path=file_path, filename=filename, media_type=media_type)


@router.post("/claim/analyze", response_model=ClaimGuardV2Response, dependencies=[Depends(verify_request_auth)])
async def analyze_claim_v2(claim: ClaimRequestV2) -> ClaimGuardV2Response:
    orchestrator = get_v2_orchestrator()
    try:
        result: ClaimGuardV2Response = orchestrator.run(_prepare_v2_claim_payload(claim))

        # ── Persist document locally when claim needs human review ──────────
        # _prepare_v2_claim_payload pops documents_base64 from the dict it
        # passes to the orchestrator, so the orchestrator never sees the raw
        # files.  We save them here at the route layer instead.
        if result.decision == "HUMAN_REVIEW":
            cid = result.claim_id or str(claim.metadata.get("claim_id") or "")
            docs_b64 = claim.documents_base64 or []
            LOGGER.info(
                "human_review_doc_save claim_id=%s docs_count=%d",
                cid, len(docs_b64),
            )
            saved = False
            for doc_part in docs_b64:
                if not isinstance(doc_part, dict):
                    continue
                name = str(doc_part.get("name") or "document.pdf").strip()
                b64  = str(doc_part.get("content_base64") or "").strip()
                if b64:
                    path = doc_store.save_document(cid, name, b64)
                    if path:
                        LOGGER.info("human_review_doc_saved claim_id=%s path=%s", cid, path)
                        saved = True
                    else:
                        LOGGER.warning("human_review_doc_save_failed claim_id=%s name=%s", cid, name)
                    break  # store only the first document
            if not saved:
                LOGGER.warning("human_review_no_doc_saved claim_id=%s", cid)

        try:
            agent_results = [
                AgentResult(
                    agent_name=str(a.get("agent_name") or a.get("agent") or ""),
                    decision=bool(a.get("decision", False)),
                    score=float(a.get("score", 0.0)),
                    reasoning=str(a.get("explanation") or a.get("reasoning") or ""),
                    details={},
                )
                for a in (result.agent_results or [])
                if isinstance(a, dict) and (a.get("agent_name") or a.get("agent"))
            ]
            trust = result.trust_layer or {}
            claim_record = ClaimResult(
                claim_id=result.claim_id or "",
                decision=result.decision,
                id=result.claim_id or "",
                status=result.decision,
                score=float(result.Ts),
                agent_results=agent_results,
                consensus_decision=result.decision,
                Ts=float(result.Ts),
                retry_count=result.retry_count,
                mahic_breakdown=result.mahic_breakdown,
                contradictions=result.contradictions,
                tx_hash=str(trust.get("tx_hash") or result.tx_hash or ""),
                ipfs_hash=str(trust.get("cid") or result.ipfs_hash or ""),
                decision_source="AI",
                previous_status=None,
                review_trace=(
                    [{
                        "step": "AI_ANALYSIS",
                        "decision": "HUMAN_REVIEW",
                        "timestamp": _utc_now_iso(),
                    }]
                    if result.decision == "HUMAN_REVIEW"
                    else []
                ),
            )
            get_claim_store().put(claim_record)
        except Exception as store_exc:
            LOGGER.warning("claim_store_put_failed claim_id=%s error=%s", result.claim_id, store_exc)
        return result
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
    if role != "ADMIN":
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
    orchestrator = get_v2_orchestrator()
    try:
        normalized_decision = str(payload.decision or "").upper()
        result = orchestrator.apply_human_decision(
            claim_id=payload.claim_id,
            decision=normalized_decision,
            reviewer_id=payload.reviewer_id,
            notes=payload.notes,
        )
        try:
            store = get_claim_store()
            existing_claim = store.get(payload.claim_id)
            if existing_claim is not None and normalized_decision in {"APPROVED", "REJECTED"}:
                trace = list(existing_claim.review_trace or [])
                trace.append(
                    {
                        "step": "HUMAN_REVIEW",
                        "decision": normalized_decision,
                        "reviewer": payload.reviewer_id or "admin",
                        "timestamp": _utc_now_iso(),
                    }
                )
                updated_claim = existing_claim.model_copy(
                    update={
                        "decision": normalized_decision,
                        "status": normalized_decision,
                        "previous_status": "HUMAN_REVIEW",
                        "decision_source": "HUMAN",
                        "review_trace": trace,
                    }
                )
                store.put(updated_claim)
        except Exception as store_exc:
            LOGGER.warning(
                "claim_store_human_decision_update_failed claim_id=%s error=%s",
                payload.claim_id,
                store_exc,
            )
        # Delete locally-stored review documents now that the claim is decided.
        # This runs regardless of APPROVED / REJECTED to keep no files on disk.
        try:
            doc_store.delete_documents(payload.claim_id)
        except Exception as del_exc:
            LOGGER.warning(
                "doc_store.delete_failed claim_id=%s error=%s",
                payload.claim_id, del_exc,
            )
        return result
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
