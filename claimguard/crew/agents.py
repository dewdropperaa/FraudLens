"""
CrewAI Agent definitions for fraud analysis (paired 1:1 with deterministic tools).

LLM routing uses LangChain chat models from :mod:`claimguard.crew.llm`; CrewAI orchestrates
agents on top. When CREW_USE_LLM=0, orchestration bypasses LLM calls and invokes the same
tools directly.
"""
from __future__ import annotations

from typing import Any, Sequence

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


def build_fraud_agents(llm: Any) -> list[Agent]:
    """
    Six independent specialists; each agent exposes exactly one analysis tool.

    ``llm`` may be None only when agents are not used for kickoff (deterministic path).
    """
    specs: Sequence[tuple[str, str, str, type]] = (
        (
            "Anomaly Detection Specialist",
            "Flag statistical and behavioral anomalies in submitted claims.",
            "You specialize in spotting unusual amounts, velocity, and documentation gaps "
            "without blocking legitimate care. You never guess numbers — you call the tool.",
            AnomalyClaimTool,
        ),
        (
            "Pattern Mining Specialist",
            "Detect repeated or structured fraud patterns across claim history.",
            "You compare this claim to historical amounts and timing; you quantify spikes "
            "with z-scores and interval checks via the tool only.",
            PatternClaimTool,
        ),
        (
            "Identity Assurance Specialist",
            "Validate identifiers and cross-record consistency for the insured.",
            "You focus on ID format, collisions across history, and name drift — always "
            "through the deterministic tool.",
            IdentityClaimTool,
        ),
        (
            "Document Integrity Specialist",
            "Assess whether submitted document references meet minimum evidence rules.",
            "You evaluate counts, required doc types for the amount tier, and naming "
            "red flags using the tool.",
            DocumentClaimTool,
        ),
        (
            "Policy Compliance Specialist",
            "Ensure CNSS/CNOPS limits and annual utilization rules are respected.",
            "You apply insurer-specific ceilings and rejection history signals via the tool.",
            PolicyClaimTool,
        ),
        (
            "Graph Risk Specialist",
            "Score network risk across patients, providers, and claims.",
            "You interpret graph-derived fraud probability and patterns — only via the tool.",
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
                llm=llm,
                verbose=False,
                allow_delegation=False,
            )
        )
    return agents
