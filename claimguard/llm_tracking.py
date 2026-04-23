from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Generator, List

_CURRENT_AGENT: ContextVar[str] = ContextVar("claimguard_current_agent", default="unknown")
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
    _CURRENT_AGENT.set(str(agent_name or "unknown"))


def get_current_agent() -> str:
    return _CURRENT_AGENT.get()


@contextmanager
def tracked_agent_context(agent_name: str) -> Generator[None, None, None]:
    token = _CURRENT_AGENT.set(str(agent_name or "unknown"))
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
            agent_name=str(agent_name or "unknown"),
            prompt_hash=prompt_hash,
            prompt_runtime_hash=runtime_hash,
            prompt_preview=prompt_preview,
            response_preview=response_preview,
            has_blackboard_context=("Blackboard:" in safe_prompt),
            has_previous_outputs=('"entries"' in safe_prompt or "entries" in safe_prompt),
            has_ocr_text=("Here is the extracted document content:" in safe_prompt),
            has_verified_fields=("verified_structured_data" in safe_prompt or "Here are structured fields:" in safe_prompt),
        )
    )
    return response


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
                has_ocr_text=("Here is the extracted document content:" in safe_prompt),
                has_verified_fields=("verified_structured_data" in safe_prompt or "Here are structured fields:" in safe_prompt),
            )
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
