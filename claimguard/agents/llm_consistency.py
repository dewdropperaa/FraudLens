from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from claimguard.llm_factory import get_llm
from claimguard.llm_tracking import get_llm_tracking_records, tracked_agent_context

EXPECTED_LLM_CALLS_PER_AGENT = 1

AGENT_CLASSIFICATION: Dict[str, str] = {
    "Identity Agent": "LLM_AGENT",
    "Document Agent": "LLM_AGENT",
    "Policy Agent": "LLM_AGENT",
    "Anomaly Agent": "LLM_AGENT",
    "Pattern Agent": "LLM_AGENT",
    "Graph Agent": "LLM_AGENT",
    "GraphRiskAgent": "LLM_AGENT",
}


def classify_agent(agent_name: str) -> str:
    return AGENT_CLASSIFICATION.get(agent_name, "DETERMINISTIC_AGENT")


def _is_explicit_deterministic(claim_data: Dict[str, Any]) -> bool:
    if not isinstance(claim_data, dict):
        return False
    explicit = bool(claim_data.get("deterministic_agent", False))
    env_flag = os.getenv("CLAIMGUARD_EXPLICIT_DETERMINISTIC", "").strip().lower() in {"1", "true", "yes"}
    return explicit or env_flag


def _build_context_prompt(agent_name: str, claim_data: Dict[str, Any], draft_reasoning: str) -> str:
    documents = claim_data.get("documents") or []
    extractions = claim_data.get("document_extractions") or []
    ocr_bits = []
    for ex in extractions:
        if isinstance(ex, dict):
            ocr_bits.append(str(ex.get("extracted_text") or ""))
    if not ocr_bits:
        ocr_bits = [str(d) for d in documents]
    ocr_text = "\n".join([x for x in ocr_bits if x]).strip() or "NO_OCR_TEXT_AVAILABLE"

    verified_fields = {
        "patient_id": claim_data.get("patient_id"),
        "amount": claim_data.get("amount"),
        "insurance": claim_data.get("insurance"),
        "identity": claim_data.get("identity"),
        "policy": claim_data.get("policy"),
    }
    previous_outputs = claim_data.get("previous_agent_outputs", claim_data.get("history", []))

    prompt = (
        f"Current agent: {agent_name}\n"
        "MANDATORY RULE: You MUST base your reasoning on the provided OCR text and verified fields. "
        "You MUST produce DIFFERENT outputs for different inputs. Generic responses are forbidden.\n"
        "Blackboard:\n"
        f"{json.dumps({'entries': previous_outputs}, ensure_ascii=False, default=str)}\n"
        "Here is the extracted document content:\n"
        f"{ocr_text}\n"
        "Here are structured fields:\n"
        f"{json.dumps(verified_fields, ensure_ascii=False, default=str)}\n"
        "Improve this draft explanation while staying concise, specific, and uncertainty-aware:\n"
        f"{draft_reasoning}\n"
    )
    return prompt


def _validate_prompt_context(prompt: str) -> None:
    required_markers = (
        "Blackboard:",
        "Here is the extracted document content:",
        "Here are structured fields:",
    )
    missing = [marker for marker in required_markers if marker not in prompt]
    if missing:
        raise RuntimeError(f"Prompt context missing required markers: {missing}")


def run_agent_consistency_check(
    *,
    agent_name: str,
    claim_data: Dict[str, Any],
    draft_reasoning: str,
) -> Tuple[str, Dict[str, Any]]:
    print(f"[AGENT EXECUTION] {agent_name}")
    agent_type = classify_agent(agent_name)
    deterministic = _is_explicit_deterministic(claim_data)
    if deterministic:
        return draft_reasoning, {
            "agent_type": "DETERMINISTIC_AGENT",
            "llm_calls": 0,
            "deterministic_declared": True,
        }

    if agent_type != "LLM_AGENT":
        return draft_reasoning, {
            "agent_type": "DETERMINISTIC_AGENT",
            "llm_calls": 0,
            "deterministic_declared": True,
        }

    before = len(get_llm_tracking_records())
    prompt = _build_context_prompt(agent_name, claim_data, draft_reasoning)
    _validate_prompt_context(prompt)

    llm = get_llm("simple")
    with tracked_agent_context(agent_name):
        response = llm.invoke(prompt)
    after = len(get_llm_tracking_records())
    llm_calls = max(0, after - before)
    if llm_calls < EXPECTED_LLM_CALLS_PER_AGENT:
        raise Exception(f"{agent_name} produced output without LLM call")
    response_text = str(getattr(response, "content", response)).strip()
    explanation = response_text if response_text else draft_reasoning
    return explanation, {
        "agent_type": "LLM_AGENT",
        "llm_calls": llm_calls,
        "deterministic_declared": False,
    }
