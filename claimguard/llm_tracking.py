from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Generator, List

_CURRENT_AGENT: ContextVar[str] = ContextVar("claimguard_current_agent", default="unassigned_agent")
_CALL_LOG: List["LLMCallRecord"] = []


@dataclass
class LLMCallRecord:
    agent_name: str
    prompt_hash: str
    prompt_runtime_hash: int
    prompt_preview: str
    response_preview: str
    has_blackboard_context: bool
    has_previous_outputs: bool
    has_ocr_text: bool
    has_verified_fields: bool


def set_current_agent(agent_name: str) -> None:
    _CURRENT_AGENT.set(str(agent_name or "unassigned_agent"))


def get_current_agent() -> str:
    return _CURRENT_AGENT.get()


@contextmanager
def tracked_agent_context(agent_name: str) -> Generator[None, None, None]:
    token = _CURRENT_AGENT.set(str(agent_name or "unassigned_agent"))
    try:
        yield
    finally:
        _CURRENT_AGENT.reset(token)


def reset_llm_tracking() -> None:
    _CALL_LOG.clear()


def get_llm_tracking_records() -> List[LLMCallRecord]:
    return list(_CALL_LOG)


def tracked_llm_call(agent_name: str, prompt: str, actual_llm_call: Callable[[str], Any]) -> Any:
    safe_prompt = str(prompt or "")
    runtime_hash = hash(safe_prompt)
    prompt_hash = hashlib.sha256(safe_prompt.encode("utf-8", errors="replace")).hexdigest()
    prompt_preview = safe_prompt[:200]
    print(f"[LLM CALL] Agent={agent_name}")
    print(f"[PROMPT HASH]={runtime_hash}")
    print(f"[PROMPT PREVIEW]={prompt_preview}")
    response = actual_llm_call(safe_prompt)
    response_text = str(getattr(response, "content", response))
    response_preview = response_text[:200]
    print(f"[RESPONSE PREVIEW]={response_preview}")
    _CALL_LOG.append(
        LLMCallRecord(
            agent_name=str(agent_name or "unassigned_agent"),
            prompt_hash=prompt_hash,
            prompt_runtime_hash=runtime_hash,
            prompt_preview=prompt_preview,
            response_preview=response_preview,
            has_blackboard_context=("Blackboard:" in safe_prompt),
            has_previous_outputs=('"entries"' in safe_prompt or "entries" in safe_prompt),
            has_ocr_text=(
                "Here is the extracted document content:" in safe_prompt
                or "Document:" in safe_prompt
            ),
            has_verified_fields=("verified_structured_data" in safe_prompt or "Here are structured fields:" in safe_prompt),
        )
    )
    return response


def parse_llm_json(response: str) -> dict[str, Any]:
    raw_text = str(response or "")
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
        return {"error": "INVALID_JSON", "raw": raw_text}
    except Exception:
        pass
    json_candidate = re.search(r"{.*}", raw_text, re.DOTALL)
    if json_candidate:
        try:
            parsed = json.loads(json_candidate.group(0))
            if isinstance(parsed, dict):
                return parsed
            return {"error": "INVALID_JSON", "raw": raw_text}
        except Exception:
            pass
    return {"error": "INVALID_JSON", "raw": raw_text}


def safe_tracked_llm_call(agent_name: str, prompt: str, llm_callable: Callable[[str], Any]) -> dict[str, Any]:
    safe_prompt = str(prompt or "")
    print(f"[LLM CALL START] agent={agent_name}")
    print(f"[PROMPT LENGTH] {len(safe_prompt)}")
    raw = tracked_llm_call(agent_name, safe_prompt, llm_callable)
    if raw is None:
        raise RuntimeError("LLM_RESPONSE_LOST")
    raw_text = str(getattr(raw, "content", raw))
    if not raw_text.strip():
        raise RuntimeError("LLM_RESPONSE_LOST")
    print(f"[LLM RAW RESPONSE] {raw_text}")
    print("[LLM RESPONSE RECEIVED]")
    parsed = parse_llm_json(raw_text)
    print(f"[AGENT PARSED OUTPUT] {str(parsed)[:500]}")
    print("[LLM RESPONSE PARSED]")
    return {
        "response": raw_text,
        "parsed": parsed,
        "agent": str(agent_name or "unassigned_agent"),
    }


class TrackedLLMProxy:
    """Proxy that wraps all known completion entrypoints with tracking."""

    def __init__(self, inner_llm: Any) -> None:
        self._inner = inner_llm

    def _agent_name(self) -> str:
        return _CURRENT_AGENT.get()

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        prompt_text = str(prompt)
        return tracked_llm_call(
            self._agent_name(),
            prompt_text,
            lambda p: self._inner.invoke(p, *args, **kwargs),
        )

    def predict(self, text: str, *args: Any, **kwargs: Any) -> Any:
        return tracked_llm_call(
            self._agent_name(),
            str(text),
            lambda p: self._inner.predict(p, *args, **kwargs),
        )

    async def ainvoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        prompt_text = str(prompt)
        safe_prompt = str(prompt_text or "")
        runtime_hash = hash(safe_prompt)
        prompt_hash = hashlib.sha256(safe_prompt.encode("utf-8", errors="replace")).hexdigest()
        prompt_preview = safe_prompt[:200]
        print(f"[LLM CALL] Agent={self._agent_name()}")
        print(f"[PROMPT HASH]={runtime_hash}")
        print(f"[PROMPT PREVIEW]={prompt_preview}")
        response = await self._inner.ainvoke(prompt_text, *args, **kwargs)
        response_text = str(getattr(response, "content", response))
        response_preview = response_text[:200]
        print(f"[RESPONSE PREVIEW]={response_preview}")
        _CALL_LOG.append(
            LLMCallRecord(
                agent_name=self._agent_name(),
                prompt_hash=prompt_hash,
                prompt_runtime_hash=runtime_hash,
                prompt_preview=prompt_preview,
                response_preview=response_preview,
                has_blackboard_context=("Blackboard:" in safe_prompt),
                has_previous_outputs=('"entries"' in safe_prompt or "entries" in safe_prompt),
                has_ocr_text=(
                    "Here is the extracted document content:" in safe_prompt
                    or "Document:" in safe_prompt
                ),
                has_verified_fields=("verified_structured_data" in safe_prompt or "Here are structured fields:" in safe_prompt),
            )
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
