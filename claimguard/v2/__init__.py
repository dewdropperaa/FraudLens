__all__ = [
    "ClaimGuardV2Orchestrator",
    "get_v2_orchestrator",
    "ClaimGuardRedTeamEngine",
    "run_red_teaming",
]


def __getattr__(name: str):
    if name in {"ClaimGuardV2Orchestrator", "get_v2_orchestrator"}:
        from .orchestrator import ClaimGuardV2Orchestrator, get_v2_orchestrator

        return {
            "ClaimGuardV2Orchestrator": ClaimGuardV2Orchestrator,
            "get_v2_orchestrator": get_v2_orchestrator,
        }[name]
    if name in {"ClaimGuardRedTeamEngine", "run_red_teaming"}:
        from .redteam import ClaimGuardRedTeamEngine, run_red_teaming

        return {
            "ClaimGuardRedTeamEngine": ClaimGuardRedTeamEngine,
            "run_red_teaming": run_red_teaming,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
