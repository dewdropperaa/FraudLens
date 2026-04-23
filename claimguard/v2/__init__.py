__all__ = [
    "ClaimGuardV2Orchestrator",
    "get_v2_orchestrator",
    "run_pipeline_v2",
    "ClaimGuardRedTeamEngine",
    "run_red_teaming",
]


def __getattr__(name: str):
    if name in {"ClaimGuardV2Orchestrator", "get_v2_orchestrator", "run_pipeline_v2"}:
        from claimguard.v2.orchestrator import (
            ClaimGuardV2Orchestrator,
            get_v2_orchestrator,
            run_pipeline_v2,
        )

        return {
            "ClaimGuardV2Orchestrator": ClaimGuardV2Orchestrator,
            "get_v2_orchestrator": get_v2_orchestrator,
            "run_pipeline_v2": run_pipeline_v2,
        }[name]
    if name in {"ClaimGuardRedTeamEngine", "run_red_teaming"}:
        from claimguard.v2.redteam import ClaimGuardRedTeamEngine, run_red_teaming

        return {
            "ClaimGuardRedTeamEngine": ClaimGuardRedTeamEngine,
            "run_red_teaming": run_red_teaming,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
