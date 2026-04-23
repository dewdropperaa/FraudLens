import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from claimguard.rate_limiting import limiter
from claimguard.models import ClaimInput, ClaimListResponse, ClaimResult
from claimguard.security import verify_request_auth, AuthContext
from claimguard.services.storage import get_claim_store
from claimguard.v2.flow_tracker import get_tracker

router = APIRouter()
_log = logging.getLogger("claimguard.review")


class ReviewBody(BaseModel):
    decision: str  # "APPROVED" or "REJECTED"
    notes: str = ""
    review_time_seconds: float = 0.0


@router.patch("/claim/{claim_id}/review")
@limiter.limit("120/minute")
async def review_claim(
    request: Request,
    claim_id: str,
    body: ReviewBody,
    auth: AuthContext = Depends(verify_request_auth),
) -> dict:
    raise Exception("LEGACY REVIEW DISABLED — USE V2 HUMAN REVIEW")


@router.get("/claim/{claim_id}/flow", dependencies=[Depends(verify_request_auth)])
async def get_claim_flow(claim_id: str) -> dict:
    tracker = get_tracker(claim_id)
    return tracker.get_state()


def _claim_payload_from_json_body(claim: ClaimInput) -> dict:
    """Legacy helper kept for compatibility while /claim is disabled."""
    return claim.model_dump()


@router.post(
    "/claim",
    response_model=ClaimResult,
    dependencies=[Depends(verify_request_auth)],
)
@limiter.limit("60/minute")
async def submit_claim(request: Request, claim: ClaimInput) -> ClaimResult:
    raise Exception("LEGACY ROUTE DISABLED — USE V2")


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
    claim_id: str | None = Form(None),
    history_json: str = Form("[]"),
    files: list[UploadFile] | None = File(None),
) -> ClaimResult:
    raise Exception("LEGACY ROUTE DISABLED — USE V2")


@router.get(
    "/claim/{claim_id}",
    response_model=ClaimResult,
    dependencies=[Depends(verify_request_auth)],
)
@limiter.limit("120/minute")
async def get_claim(request: Request, claim_id: str) -> ClaimResult:
    if len(claim_id) > 128:
        raise HTTPException(status_code=400, detail="claim_id too long")
    result = get_claim_store().get(claim_id)
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
    filter_normalized = filter.strip().lower()
    if filter_normalized == "all":
        decision = None
    elif filter_normalized == "fraud":
        decision = "REJECTED"
    elif filter_normalized == "valid":
        decision = "APPROVED"
    elif filter_normalized in ("pending", "human_review"):
        decision = "HUMAN_REVIEW"
    else:
        raise HTTPException(status_code=400, detail="filter must be one of: all, fraud, valid, pending")
    offset = (page - 1) * page_size
    items, total = get_claim_store().list_page(decision, offset=offset, limit=page_size)
    return ClaimListResponse(items=items, total=total, page=page, page_size=page_size)
