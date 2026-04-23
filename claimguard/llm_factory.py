from __future__ import annotations

import os
from typing import Any, Final

from langchain_community.llms import Ollama
from claimguard.llm_tracking import TrackedLLMProxy

_MODEL_ROUTES: Final[dict[str, str]] = {
    "simple": "mistral",
    "complex": "llama3",
    "high_risk": "deepseek-r1",
}


def resolve_model(model_name: str) -> str:
    key = (model_name or "").strip()
    if key in _MODEL_ROUTES:
        return _MODEL_ROUTES[key]
    if key in _MODEL_ROUTES.values():
        return key
    raise RuntimeError(
        f"Unsupported LLM route/model '{model_name}'. "
        f"Allowed routes: {sorted(_MODEL_ROUTES.keys())}; models: {sorted(_MODEL_ROUTES.values())}."
    )


def get_llm(model_name: str, *, tracked: bool = True) -> Any:
    base = Ollama(
        model=resolve_model(model_name),
        temperature=0.1,
    )
    if not tracked:
        return base
    return TrackedLLMProxy(base)


def get_crewai_llm(model_name: str) -> str:
    """Return CrewAI-compatible model route string."""
    return f"ollama/{resolve_model(model_name)}"


def assert_ollama_connection() -> None:
    if os.getenv("CLAIMGUARD_SKIP_OLLAMA_CHECK", "").strip().lower() in {"1", "true", "yes"}:
        return
    if os.getenv("ENVIRONMENT", "").strip().lower() == "test":
        return
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    llm = get_llm("mistral", tracked=False)
    response = llm.invoke("Say 'Ollama working'")
    if not str(response).strip():
        raise RuntimeError("Ollama connectivity test failed: empty response.")
