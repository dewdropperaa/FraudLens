__all__ = ["ConsensusSystem", "get_consensus_system"]


def __getattr__(name: str):
    if name in {"ConsensusSystem", "get_consensus_system"}:
        from claimguard.services.consensus import ConsensusSystem, get_consensus_system

        return {"ConsensusSystem": ConsensusSystem, "get_consensus_system": get_consensus_system}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
