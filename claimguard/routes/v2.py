from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from claimguard.security import verify_request_auth
from claimguard.v2 import get_v2_orchestrator
from claimguard.v2.blackboard import BlackboardValidationError
from claimguard.v2.redteam import StrictModeConfig, run_red_teaming
from claimguard.v2.schemas import ClaimGuardV2Response, ClaimRequestV2
from claimguard.v2.trust_layer import TrustLayerIPFSFailure

from claimguard.v2.flow_tracker import get_tracker

router = APIRouter(prefix="/v2", tags=["claimguard-v2"])

@router.get("/claim/{claim_id}/flow", dependencies=[Depends(verify_request_auth)])
async def get_claim_flow(claim_id: str) -> dict:
    tracker = get_tracker(claim_id)
    return tracker.get_state()


@router.post("/claim/analyze", response_model=ClaimGuardV2Response, dependencies=[Depends(verify_request_auth)])
async def analyze_claim_v2(claim: ClaimRequestV2) -> ClaimGuardV2Response:
    orchestrator = get_v2_orchestrator()
    try:
        return orchestrator.run(claim.model_dump())
    except BlackboardValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TrustLayerIPFSFailure as exc:
        raise HTTPException(status_code=503, detail=f"Trust layer IPFS upload failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 analysis failed: {exc}") from exc


@router.get("/debug/fraud-graph", dependencies=[Depends(verify_request_auth)])
async def debug_fraud_graph_v2(render_png: bool = False) -> dict:
    orchestrator = get_v2_orchestrator()
    try:
        return orchestrator.get_fraud_graph_debug(render_png=render_png)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ClaimGuard v2 fraud-graph debug failed: {exc}") from exc


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
