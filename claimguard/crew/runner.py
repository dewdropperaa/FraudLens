"""
Sequential fraud analysis entrypoints (sync + async).

* Default (``CREW_USE_LLM`` unset/0): deterministic tools only — no LLM latency/cost.
* Optional (``CREW_USE_LLM=1``): six single-task Crews executed in strict sequence.

Memory integration: before agents run, similar past cases are retrieved from the
Case Memory Layer and injected into claim_data under "memory_context" so every
agent can reason over historical evidence.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
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
from claimguard.v2.memory import get_memory_layer

logger = logging.getLogger("claimguard.crew.runner")

from claimguard.v2.flow_tracker import get_tracker

_AGENT_KEYS: tuple[str, ...] = (
    "identity",
    "document",
    "policy",
    "anomaly",
    "pattern",
    "graph",
)

_AGENT_NAMES = {
    "identity": "IdentityAgent",
    "document": "DocumentAgent",
    "policy": "PolicyAgent",
    "anomaly": "AnomalyAgent",
    "pattern": "PatternAgent",
    "graph": "GraphAgent",
}

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


def _inject_memory_context(claim_data: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve similar past cases and inject them into claim_data.

    Returns a shallow copy of claim_data with 'memory_context' added so the
    original dict is not mutated.  Failures are logged and silenced — memory
    is advisory and must never block the pipeline.
    """
    try:
        memory = get_memory_layer()
        similar_cases = memory.retrieve_similar_cases(claim_data)
        if similar_cases:
            logger.info(
                "legacy_memory_context_injected count=%d", len(similar_cases)
            )
        return {**claim_data, "memory_context": similar_cases}
    except Exception as exc:
        logger.warning("legacy_memory_inject_failed error=%s — running without memory", exc)
        return claim_data


def _run_deterministic_sequential(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    enriched_data = _inject_memory_context(claim_data)
    claim_id = enriched_data.get("claim_id", "unknown")
    tracker = get_tracker(claim_id)
    
    raw = []
    for key in _AGENT_KEYS:
        agent_name = _AGENT_NAMES[key]
        tracker.update(agent_name, "RUNNING")
        try:
            result = _AGENT_RUNNERS[key](enriched_data)
            tracker.update(agent_name, "COMPLETED")
            raw.append(result)
        except Exception as exc:
            tracker.update(agent_name, "FAILED")
            raise exc

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


def _run_llm_mini_crews_sequential(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    claim_data = _inject_memory_context(claim_data)
    claim_id = claim_data.get("claim_id", "unknown")
    tracker = get_tracker(claim_id)
    claim_json = json.dumps(claim_data, sort_keys=True, ensure_ascii=False)
    crews = build_mini_crews(claim_json)
    by_key = dict(zip(_AGENT_KEYS, crews))
    out: List[Dict[str, Any]] = []
    
    for key in _AGENT_KEYS:
        agent_name = _AGENT_NAMES[key]
        tracker.update(agent_name, "RUNNING")
        crew = by_key.get(key)
        if crew is None:
            tracker.update(agent_name, "FAILED")
            raise RuntimeError(f"CrewAI configuration error: missing crew for agent key '{key}'.")
        
        try:
            cout: CrewOutput = crew.kickoff()
            parsed = _parse_structured_output(cout)
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
                raise RuntimeError(
                    f"CrewAI/Ollama output parsing failed for agent key '{key}'. "
                    "No deterministic fallback is allowed in LLM mode."
                )
            tracker.update(agent_name, "COMPLETED")
            out.append(enrich_legacy_with_audit(merged))
        except Exception as exc:
            tracker.update(agent_name, "FAILED")
            raise exc

    log_agent_decisions(out)
    return sort_agent_dicts(out)


def run_claim_agents(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync entry: sequential analysis, structured legacy dicts (AgentResult-compatible)."""
    if _use_llm_crew():
        return _run_llm_mini_crews_sequential(claim_data)
    return _run_deterministic_sequential(claim_data)


async def run_claim_agents_async(claim_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Async entry: offload sync sequential work so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_claim_agents, claim_data)
