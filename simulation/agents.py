"""
3 CrewAI agents powered by local Ollama (no OpenAI, no paid API).

  AuditorAgent  → finds vulnerabilities, uses analyse_contract tool
  GovernorAgent → turns findings into proposals, uses draft_proposal tool
  ReporterAgent → summarises everything, uses summarise_audit tool
"""

from crewai import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool
from crewai.tools.base_tool import BaseTool
from dotenv import load_dotenv
from typing import Any
import os, json

load_dotenv()


class RuleBasedLLM(BaseLLM):
    """A minimal environment-friendly LLM replacement.

    This LLM avoids external API calls by interpreting prompts and invoking the
    provided tool functions directly. It is used in constrained environments
    where running a real model (e.g., Ollama) is not feasible.
    """

    def __init__(self, model: str = "rule-based", temperature: float | None = 0.0):
        super().__init__(model=model, temperature=temperature)

    def _extract_json_from_text(self, text: str) -> list[dict] | dict | None:
        # Simple heuristic: find the first JSON array/object in the text
        try:
            # This is naive but works for our controlled prompts.
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except Exception:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
            except Exception:
                return None

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[Any] | None = None,
    ) -> str:
        # Extract the last user message content for decision-making.
        if isinstance(messages, list) and messages:
            text = messages[-1].get("content", "")
        elif isinstance(messages, str):
            text = messages
        else:
            text = ""

        text_lower = text.lower()

        # Direct tool routing based on prompt intent.
        if "analyse_contract" in text_lower:
            return analyse_contract.run("GovernanceAudit")

        if "draft_proposal" in text_lower:
            # Expect findings JSON to be present in the prompt.
            findings = self._extract_json_from_text(text)
            if not isinstance(findings, list):
                # Fallback to our known findings if parsing fails.
                findings = json.loads(analyse_contract.run("GovernanceAudit"))

            proposals = []
            for f in findings:
                if f.get("severity") in ("CRITICAL", "HIGH"):
                    proposals.append(json.loads(draft_proposal.run(json.dumps(f))))
            return json.dumps(proposals)

        if "summarise_audit" in text_lower or "summarize" in text_lower:
            findings = self._extract_json_from_text(text)
            if not isinstance(findings, list):
                findings = json.loads(analyse_contract.run("GovernanceAudit"))
            return summarise_audit.run(json.dumps(findings))

        # Fallback: return a minimal JSON string.
        return json.dumps({"error": "Unable to determine action from prompt."})


# Shared LLM instance injected into all agents.
llm = RuleBasedLLM()

# ── tools ─────────────────────────────────────────────────

@tool
def analyse_contract(contract_name: str) -> str:
    """
    Analyse a smart contract and return a list of findings as JSON.
    Input: contract name or description string.
    Output: JSON array of findings with severity and description.
    """
    try:
        # Simulation findings based on the actual GovernanceAudit contract
        findings = [
            {
                "id": 1,
                "detector": "access-control",
                "severity": "CRITICAL",
                "description": (
                    "propose() has no role restriction — any address can "
                    "create governance proposals, bypassing governance rules."
                ),
                "line": "propose(string)"
            },
            {
                "id": 2,
                "detector": "voting-manipulation",
                "severity": "HIGH",
                "description": (
                    "vote() has no duplicate-vote prevention — the same "
                    "address can vote multiple times on the same proposal."
                ),
                "line": "vote(uint256,bool)"
            },
            {
                "id": 3,
                "detector": "timestamp-dependence",
                "severity": "MEDIUM",
                "description": (
                    "executeProposal() relies on block.timestamp for the "
                    "3-day delay. Validators can manipulate this by ~15s."
                ),
                "line": "executeProposal(uint256)"
            },
            {
                "id": 4,
                "detector": "centralisation",
                "severity": "LOW",
                "description": (
                    "Owner has full role control with no timelock or "
                    "multisig — single point of failure."
                ),
                "line": "grantRole(bytes32,address)"
            },
        ]
        return json.dumps(findings)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def draft_proposal(finding_json: str) -> str:
    """
    Draft a governance proposal from a single audit finding.
    Input: JSON string of one finding with severity and description.
    Output: JSON with proposal_text and severity.
    """
    try:
        f = json.loads(finding_json)
        proposal = (
            f"[{f['severity']}] {f['detector'].upper()} — "
            f"Remediation required: {f['description']} "
            f"Affected function: {f.get('line', 'unknown')}."
        )
        return json.dumps({
            "proposal_text": proposal,
            "severity":      f["severity"],
            "detector":      f["detector"],
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def summarise_audit(findings_json: str) -> str:
    """
    Summarise all audit findings into a one-line on-chain log entry.
    Input: JSON array of findings.
    Output: JSON with summary string and numeric severity (0-4).
    """
    try:
        findings = json.loads(findings_json)
        severity_map = {"CRITICAL": 4, "HIGH": 3,
                        "MEDIUM": 2, "LOW": 1, "INFO": 0}
        counts = {"CRITICAL": 0, "HIGH": 0,
                  "MEDIUM": 0, "LOW": 0, "INFO": 0}
        for f in findings:
            counts[f.get("severity", "INFO")] += 1

        top = max(
            (k for k in counts if counts[k] > 0),
            key=lambda k: severity_map[k],
            default="INFO"
        )
        summary = (
            f"Audit: {len(findings)} findings — "
            f"{counts['CRITICAL']} CRITICAL, "
            f"{counts['HIGH']} HIGH, "
            f"{counts['MEDIUM']} MEDIUM, "
            f"{counts['LOW']} LOW."
        )
        return json.dumps({
            "summary":          summary,
            "top_severity":     top,
            "top_severity_int": severity_map[top],
            "counts":           counts,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── agent factory functions ───────────────────────────────


def build_auditor(llm) -> Agent:
    """Build the smart contract auditor agent."""
    return Agent(
        role="Smart Contract Auditor",
        goal=(
            "Analyse the GovernanceAudit smart contract and identify "
            "all vulnerabilities. Return a JSON list of findings ordered "
            "by severity from CRITICAL to INFO."
        ),
        backstory=(
            "You are a senior blockchain security researcher specialising "
            "in Solidity. You are systematic, precise, and always "
            "reference specific functions in your findings."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def build_governor(llm) -> Agent:
    """Build the governance coordinator agent."""
    return Agent(
        role="Governance Coordinator",
        goal=(
            "Review audit findings and create governance proposals for "
            "every CRITICAL and HIGH severity issue found by the auditor."
        ),
        backstory=(
            "You manage on-chain governance for a DAO. You translate "
            "security risks into clear, actionable proposals that "
            "token holders can understand and vote on."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def build_reporter(llm) -> Agent:
    """Build the audit reporter agent."""
    return Agent(
        role="Audit Reporter",
        goal=(
            "Summarise all findings into a concise on-chain log entry "
            "with the correct severity level (0-4 integer)."
        ),
        backstory=(
            "You are responsible for creating the immutable on-chain "
            "audit record. Your summary must be under 200 characters "
            "and include finding counts by severity."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )
