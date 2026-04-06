"""One Task per fraud agent — each task is wired to a single deterministic tool."""
from __future__ import annotations

from crewai import Task

from claimguard.crew.models import AgentDecisionOutput


def build_tasks_for_agents(claim_json: str, agents: list) -> list[Task]:
    """
    Build six independent tasks. Descriptions embed the claim JSON so each mini-crew
    is self-contained (parallel kickoffs do not rely on shared Crew context).
    """
    instructions = (
        "You MUST call your tool exactly once with the full claim JSON string provided below "
        "as the sole argument. Do not fabricate scores. After the tool returns, reply with "
        "ONLY valid JSON matching the schema (no markdown fences): "
        '{"agent": "<exact agent name from tool output>", '
        '"decision": "APPROVED" or "REJECTED", '
        '"score": <float 0-100>, '
        '"reason": "<short string>", '
        '"explainability": "<audit notes: key signals and thresholds>", '
        '"details": { ... optional copy of tool details ... } }. '
        "Derive decision APPROVED iff the tool JSON has decision==true (boolean)."
        "\n\nCLAIM_JSON:\n"
        f"{claim_json}"
    )
    tasks: list[Task] = []
    for agent in agents:
        tasks.append(
            Task(
                description=instructions,
                expected_output=(
                    "A single JSON object matching AgentDecisionOutput: agent, decision, "
                    "score, reason, explainability, details."
                ),
                agent=agent,
                output_pydantic=AgentDecisionOutput,
            )
        )
    return tasks
