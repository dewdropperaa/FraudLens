import asyncio
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from claimguard.rate_limiting import limiter
from claimguard.models import ClaimInput, ClaimListResponse, ClaimResult
from claimguard.security import verify_request_auth, AuthContext
from claimguard.services import get_consensus_system
from claimguard.services.storage import get_claim_store
from claimguard.services.document_extraction import (
    build_extractions_from_base64_parts,
    build_extractions_from_upload_files,
)
from claimguard.v2.flow_tracker import get_tracker

router = APIRouter()
_log = logging.getLogger("claimguard.review")


class ReviewBody(BaseModel):
    decision: str  # "APPROVED" or "REJECTED"


@router.patch("/claim/{claim_id}/review")
@limiter.limit("120/minute")
async def review_claim(
    request: Request,
    claim_id: str,
    body: ReviewBody,
    auth: AuthContext = Depends(verify_request_auth),
) -> dict:
    if len(claim_id) > 128:
        raise HTTPException(status_code=400, detail="claim_id too long")
    decision = body.decision.strip().upper()
    if decision not in ("APPROVED", "REJECTED"):
        raise HTTPException(status_code=400, detail="decision must be APPROVED or REJECTED")

    store = get_claim_store()
    claim = store.get(claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail="Claim not found")

    tx_hash: str | None = None
    ipfs_hash: str | None = None

    # Admin approves a PENDING claim → upload IPFS then blockchain
    if decision == "APPROVED" and claim.decision == "PENDING":
        # 1. Upload to IPFS
        try:
            from claimguard.services.consensus import ConsensusSystem
            from claimguard.services.ipfs import get_ipfs_service
            ipfs_service = get_ipfs_service()
            result = await ipfs_service.upload_json(
                {"claim_id": claim_id, "decision": "APPROVED", "score": claim.score},
                f"{claim_id}_approved.json",
                encrypt=True,
            )
            ipfs_hash = result.ipfs_hash
            _log.info("human_review_ipfs claim_id=%s hash=%s", claim_id, ipfs_hash)
        except Exception as exc:
            _log.warning("ipfs_on_review_failed claim_id=%s error=%s", claim_id, exc)

        # 2. Blockchain
        try:
            from claimguard.services.blockchain import get_blockchain_service
            svc = get_blockchain_service()
            if svc.contract:
                chain_result = await asyncio.to_thread(
                    lambda: svc.validate_claim_on_chain(
                        claim_id=claim_id,
                        score=float(claim.score),
                        decision="APPROVED",
                        ipfs_hashes=[ipfs_hash] if ipfs_hash else [],
                        patient_id="",
                    )
                )
                tx_hash = chain_result.get("tx_hash")
                _log.info("human_review_blockchain claim_id=%s tx=%s", claim_id, tx_hash)
        except Exception as exc:
            _log.warning("blockchain_on_review_failed claim_id=%s error=%s", claim_id, exc)

    store.update_decision(claim_id, decision, reviewer_id=auth.user_id or "unknown")
    return {"claim_id": claim_id, "decision": decision, "status": "updated", "tx_hash": tx_hash, "ipfs_hash": ipfs_hash}


@router.get("/claim/{claim_id}/flow", dependencies=[Depends(verify_request_auth)])
async def get_claim_flow(claim_id: str) -> dict:
    tracker = get_tracker(claim_id)
    return tracker.get_state()


def _claim_payload_from_json_body(claim: ClaimInput) -> dict:
    """Merge optional base64 inline files into ``document_extractions`` and document names."""
    data = claim.model_dump()
    parts = data.pop("documents_base64", None) or []
    if parts:
        extractions = build_extractions_from_base64_parts(parts)
        data["document_extractions"] = extractions
        for ex in extractions:
            fn = ex.get("file_name") or ""
            if fn and fn not in data["documents"]:
                data["documents"].append(fn)
    return data


@router.post(
    "/claim",
    response_model=ClaimResult,
    dependencies=[Depends(verify_request_auth)],
)
@limiter.limit("60/minute")
async def submit_claim(request: Request, claim: ClaimInput) -> ClaimResult:
    consensus_system = get_consensus_system()
    return await consensus_system.process_claim(_claim_payload_from_json_body(claim))


@router.post(
    "/claim/upload",
    response_model=ClaimResult,
    dependencies=[Depends(verify_request_auth)],
)
@limiter.limit("30/minute")
async def submit_claim_upload(
    request: Request,
    patient_id: str = Form(...),
    provider_id: str = Form(...),
    amount: float = Form(...),
    insurance: str = Form(...),
    claim_id: Optional[str] = Form(None),
    history_json: str = Form("[]"),
    files: Optional[List[UploadFile]] = File(None),
) -> ClaimResult:
    if insurance not in ("CNSS", "CNOPS"):
        raise HTTPException(status_code=422, detail="insurance must be CNSS or CNOPS")
    try:
        history = json.loads(history_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"history_json must be valid JSON: {e}") from e
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history_json must be a JSON array")

    file_list = files or []
    extractions = await build_extractions_from_upload_files(file_list) if file_list else []
    doc_names = [e["file_name"] for e in extractions]

    claim_data: dict = {
        "patient_id": patient_id.strip(),
        "provider_id": provider_id.strip(),
        "amount": amount,
        "insurance": insurance,
        "documents": doc_names,
        "history": history,
    }
    if extractions:
        claim_data["document_extractions"] = extractions
    stripped = (claim_id or "").strip()
    if stripped:
        claim_data["claim_id"] = stripped

    consensus_system = get_consensus_system()
    return await consensus_system.process_claim(claim_data)


@router.get(
    "/claim/{claim_id}",
    response_model=ClaimResult,
    dependencies=[Depends(verify_request_auth)],
)
@limiter.limit("120/minute")
async def get_claim(request: Request, claim_id: str) -> ClaimResult:
    if len(claim_id) > 128:
        raise HTTPException(status_code=400, detail="claim_id too long")
    consensus_system = get_consensus_system()
    result = consensus_system.get_claim(claim_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Claim not found")
    return result


@router.get(
    "/claims",
    response_model=ClaimListResponse,
    dependencies=[Depends(verify_request_auth)],
)
@limiter.limit("120/minute")
async def list_claims(
    request: Request,
    filter: str = "all",
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
) -> ClaimListResponse:
    consensus_system = get_consensus_system()
    filter_normalized = filter.strip().lower()
    if filter_normalized == "all":
        decision = None
    elif filter_normalized == "fraud":
        decision = "REJECTED"
    elif filter_normalized == "valid":
        decision = "APPROVED"
    elif filter_normalized in ("pending", "human_review"):
        decision = "PENDING"
    else:
        raise HTTPException(status_code=400, detail="filter must be one of: all, fraud, valid, pending")
    offset = (page - 1) * page_size
    items, total = consensus_system.list_claims(decision, offset=offset, limit=page_size)
    return ClaimListResponse(items=items, total=total, page=page, page_size=page_size)
