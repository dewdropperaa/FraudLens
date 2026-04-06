"""LangChain chat models for CrewAI (optional LLM path via CREW_USE_LLM=1)."""
from __future__ import annotations

import os

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI


def get_crew_llm() -> ChatOpenAI | ChatOllama:
    """
    Resolve a LangChain chat model for CrewAI agents.

    CrewAI accepts LangChain-compatible models and normalizes them via ``create_llm``
    (LiteLLM-backed ``crewai.LLM``), preserving the ClaimGuard stack: **LangChain
    connects models to tools; CrewAI orchestrates agents.**

    Priority:
    1. ``OLLAMA_BASE_URL`` + ``CREW_OLLAMA_MODEL`` (local)
    2. ``OPENAI_API_KEY`` + ``CREW_OPENAI_MODEL`` (hosted)
    """
    ollama_base = os.getenv("OLLAMA_BASE_URL", "").strip()
    ollama_model = os.getenv("CREW_OLLAMA_MODEL", "llama3.1").strip()
    if ollama_base:
        return ChatOllama(
            model=ollama_model,
            base_url=ollama_base.rstrip("/"),
            temperature=0,
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "CREW_USE_LLM=1 requires OLLAMA_BASE_URL or OPENAI_API_KEY to be set."
        )
    model = os.getenv("CREW_OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, api_key=api_key, temperature=0)
