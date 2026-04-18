"""Centralized Ollama model provider for CrewAI."""
from __future__ import annotations
from claimguard.llm_factory import get_llm


def get_crew_llm(model_name: str = "simple"):
    return get_llm(model_name)
