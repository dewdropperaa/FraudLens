"""
CrewAI Agent definitions for fraud analysis (paired 1:1 with deterministic tools).

LLM routing uses LangChain chat models from :mod:`claimguard.crew.llm`; CrewAI orchestrates
agents on top. When CREW_USE_LLM=0, orchestration bypasses LLM calls and invokes the same
tools directly.
"""
from __future__ import annotations

from typing import Sequence

from crewai import Agent

from claimguard.crew.tools import (
    ALL_CLAIM_TOOLS,
    AnomalyClaimTool,
    DocumentClaimTool,
    GraphClaimTool,
    IdentityClaimTool,
    PatternClaimTool,
    PolicyClaimTool,
)
from claimguard.llm_factory import get_llm


def build_fraud_agents(model_name: str) -> list[Agent]:
    """
    Six independent specialists; each agent exposes exactly one analysis tool.

    Every agent gets an explicit Ollama-backed LLM assignment.
    """
    specs: Sequence[tuple[str, str, str, type]] = (
        (
            "Fraud Risk & Anomaly Expert",
            "Surface abnormal amounts, history inconsistencies, and suspiciously stable profiles.",
            "You distrust surface patterns; tool output is raw signal only. Flag unusual jumps, "
            "historical mismatch, and 'too clean' stability even when scores look fine. "
            "Assume manipulated inputs may look normal. You never guess numbers — you call the tool.",
            AnomalyClaimTool,
        ),
        (
            "Fraud Pattern Analyst",
            "Detect repetition, timing regularity, and behavioral signatures.",
            "You are skeptical of repeated clean claims, evenly spaced timing (automation), and "
            "artificially consistent amounts. Attackers may mimic legitimacy — report irregularities "
            "and suspicious regularity via the tool only.",
            PatternClaimTool,
        ),
        (
            "Identity Verification Specialist",
            "Detect spoofed or reused identifiers and cross-record inconsistency.",
            "Identity fields may be forged or recycled. Look for history conflicts, ID collisions, "
            "and subtle drift — format match is not proof. Use the deterministic tool only.",
            IdentityClaimTool,
        ),
        (
            "Forensic Document Analyst",
            "Assess document completeness, consistency, and adversarial wording.",
            "Documents may be manipulated or misleading. Question missing types, extraction vs claim "
            "mismatches, and influence language. Even valid-looking packs may be too perfect — use the tool.",
            DocumentClaimTool,
        ),
        (
            "Compliance & Policy Risk Analyst",
            "Apply policy limits while flagging threshold gaming and borderline abuse.",
            "Do not rubber-stamp rules: probe borderline amounts, repeated near-limit claims, and "
            "limit optimization. Use the tool for ceilings and rejection history.",
            PolicyClaimTool,
        ),
        (
            "Network Fraud Analyst",
            "Interpret probabilistic graph risk across patients, providers, and claims.",
            "Graph risk is not binary truth. Weight indirect links, clusters, and weak signals; "
            "do not dismiss moderate fraud probability — interpret via the tool only.",
            GraphClaimTool,
        ),
    )

    agents: list[Agent] = []
    for role, goal, backstory, tool_cls in specs:
        tool_instance = next(t for t in ALL_CLAIM_TOOLS if isinstance(t, tool_cls))
        agents.append(
            Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=[tool_instance],
                llm=get_llm(model_name),
                verbose=False,
                allow_delegation=False,
            )
        )
    return agents
