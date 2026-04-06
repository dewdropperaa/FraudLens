"""
Fraud detection crews: six independent single-task crews executed in parallel.

CrewAI 1.x removed ``Process.parallel``; independent mini-crews preserve parallel
execution without sequential blocking between specialists.
"""
from __future__ import annotations

from crewai import Crew, Process

from claimguard.crew.agents import build_fraud_agents
from claimguard.crew.llm import get_crew_llm
from claimguard.crew.tasks import build_tasks_for_agents


def build_mini_crews(claim_json: str) -> list[Crew]:
    """
    Build six Crew instances (one agent + one task each), ready for parallel kickoff.

    Uses the configured LangChain chat model (OpenAI or Ollama); CrewAI normalizes it internally.
    """
    llm = get_crew_llm()
    agents = build_fraud_agents(llm)
    tasks_per_agent = build_tasks_for_agents(claim_json, agents)
    crews: list[Crew] = []
    for agent, task in zip(agents, tasks_per_agent, strict=True):
        crews.append(
            Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
        )
    return crews
