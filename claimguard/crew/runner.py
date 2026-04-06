"""
Parallel fraud analysis entrypoints (sync + async).

* Default (``CREW_USE_LLM`` unset/0): deterministic tools only — no LLM latency/cost.
* Optional (``CREW_USE_LLM=1``): six independent single-task Crews kicked off in parallel.

CrewAI 1.x note: ``Process.parallel`` was removed; parallel execution is achieved via
thread-pooled ``kickoff`` of independent mini-crews (or thread-pooled deterministic tools).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from crewai.crews.crew_output import CrewOutput

from claimguard.agents.anomaly_agent import AnomalyAgent
from claimguard.agents.document_agent import DocumentAgent
from claimguard.agents.graph_agent import GraphAgent
from claimguard.agents.identity_agent import IdentityAgent
from claimguard.agents.pattern_agent import PatternAgent
from claimguard.agents.policy_agent import PolicyAgent
from claimguard.crew.consensus import enrich_legacy_with_audit, log_agent_decisions, sort_agent_dicts
from claimguard.crew.crew import build_mini_crews
from claimguard.crew.models import AgentDecisionOutput
from claimguard.crew.tools import ALL_CLAIM_TOOLS

logger = logging.getLogger("claimguard.crew.runner")

_MAX_WORKERS = len(ALL_CLAIM_TOOLS)

_AGENT_KEYS: tuple[str, ...] = (
    "anomaly",
    "pattern",
    "identity",
    "document",
    "policy",
    "graph",
)

_AGENT_RUNNERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "anomaly": lambda d: AnomalyAgent().analyze(d),
    "pattern": lambda d: PatternAgent().analyze(d),
    "identity": lambda d: IdentityAgent().analyze(d),
    "document": lambda d: DocumentAgent().analyze(d),
    "policy": lambda d: PolicyAgent().analyze(d),
    "graph": lambda d: GraphAgent().analyze(d),
}


def _use_llm_crew() -> bool:
    return os.getenv("CREW_USE_LLM", "").strip().lower() in {"1", "true", "yes"}


def _run_deterministic_parallel(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        raw = list(pool.map(lambda k: _AGENT_RUNNERS[k](claim_data), _AGENT_KEYS))
    enriched = [enrich_legacy_with_audit(dict(r)) for r in raw]
    log_agent_decisions(enriched)
    return sort_agent_dicts(enriched)


def _parse_structured_output(crew_out: CrewOutput) -> AgentDecisionOutput | None:
    if not crew_out.tasks_output:
        return None
    t0 = crew_out.tasks_output[0]
    if t0.pydantic is not None:
        try:
            return AgentDecisionOutput.model_validate(t0.pydantic.model_dump())
        except Exception:
            pass
    if t0.json_dict:
        try:
            return AgentDecisionOutput.model_validate(t0.json_dict)
        except Exception:
            pass
    raw = (t0.raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[-1]
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    try:
        return AgentDecisionOutput.model_validate(json.loads(raw))
    except Exception:
        return None


def _kickoff_one_crew(crew) -> CrewOutput:
    return crew.kickoff()


def _run_llm_mini_crews_parallel(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    claim_json = json.dumps(claim_data, sort_keys=True, ensure_ascii=False)
    crews = build_mini_crews(claim_json)
    keys = list(_AGENT_KEYS)
    results: list[Dict[str, Any] | None] = [None] * len(keys)

    def run_one(idx: int, crew) -> tuple[int, CrewOutput]:
        return idx, _kickoff_one_crew(crew)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = [pool.submit(run_one, i, c) for i, c in enumerate(crews)]
        for fut in as_completed(futs):
            idx, cout = fut.result()
            parsed = _parse_structured_output(cout)
            key = keys[idx]
            truth = _AGENT_RUNNERS[key](claim_data)
            if parsed is not None:
                det = dict(truth.get("details") or {})
                det["llm_structured_output"] = parsed.model_dump()
                det.setdefault(
                    "explainability",
                    parsed.explainability or "",
                )
                merged = {
                    **truth,
                    "reasoning": truth.get("reasoning", ""),
                    "details": det,
                }
            else:
                logger.warning(
                    "crew_kickoff_parse_failed agent_key=%s — using deterministic output only",
                    key,
                )
                merged = dict(truth)
            results[idx] = enrich_legacy_with_audit(merged)

    out = [r for r in results if r is not None]
    log_agent_decisions(out)
    return sort_agent_dicts(out)


def run_claim_agents(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync entry: parallel analysis, structured legacy dicts (AgentResult-compatible)."""
    if _use_llm_crew():
        return _run_llm_mini_crews_parallel(claim_data)
    return _run_deterministic_parallel(claim_data)


async def run_claim_agents_async(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Async entry: offload sync parallel work so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_claim_agents, claim_data)
