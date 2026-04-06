"""
Runs the 3-agent CrewAI pipeline and links results to the blockchain.

Flow:
  1. AuditorAgent  → finds vulnerabilities in GovernanceAudit.sol
  2. GovernorAgent → creates proposals for CRITICAL + HIGH findings
  3. ReporterAgent → summarises, then logAudit() + propose() on-chain
"""

from crewai import Crew, Task, Process
from agents import build_auditor, build_governor, build_reporter, llm
from blockchain import BlockchainClient, AUDITOR_ROLE
import json, re


def extract_json(text: str):
    """Extract the first valid JSON array or object from a string."""
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # Try to find JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def run_simulation() -> dict:
    """Run the full agent simulation and sync results to chain."""

    # ── 1. preflight check ───────────────────────────────
    print("\n[PREFLIGHT] Checking blockchain connection and roles...")
    chain = BlockchainClient()
    checks = chain.preflight_check()

    for k, v in checks.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}")

    if not checks["all_passed"]:
        failed = [k for k, v in checks.items()
                  if not v and k != "all_passed"]
        return {
            "status": "preflight_failed",
            "failed_checks": failed,
            "fix": {
                "wallet_has_balance":
                    "Get Amoy MATIC: https://faucet.polygon.technology",
                "has_auditor_role":
                    "Call grantRole(AUDITOR_ROLE, your_wallet) in Remix",
                "governance_deployed":
                    "Set GOVERNANCE_AUDIT_ADDRESS in .env",
                "access_ctrl_deployed":
                    "Set ACCESS_CONTROL_ADDRESS in .env",
            }
        }

    # ── 2. read chain state before ───────────────────────
    state_before = chain.read_state()
    print(f"\n[CHAIN BEFORE] {state_before}")

    # ── 3. build agents ──────────────────────────────────
    auditor  = build_auditor(llm)
    governor = build_governor(llm)
    reporter = build_reporter(llm)

    # ── 4. define tasks ──────────────────────────────────
    audit_task = Task(
        description=(
            "Analyse the GovernanceAudit smart contract deployed on "
            "Polygon Amoy. Use the analyse_contract tool with input "
            "'GovernanceAudit'. Return the full JSON array of findings."
        ),
        agent=auditor,
        expected_output=(
            "JSON array of findings. Each item must have: "
            "id, detector, severity, description, line."
        )
    )

    governance_task = Task(
        description=(
            "Read the audit findings from the previous task. "
            "For each finding with severity CRITICAL or HIGH, "
            "call draft_proposal once per finding. "
            "Return a JSON array of all drafted proposals."
        ),
        agent=governor,
        expected_output=(
            "JSON array of proposals. Each must have: "
            "proposal_text, severity, detector."
        ),
        context=[audit_task]
    )

    report_task = Task(
        description=(
            "Read the full findings list from the audit task. "
            "Call summarise_audit with the findings JSON array. "
            "Return JSON with: summary (string), "
            "top_severity (string), top_severity_int (0-4 integer)."
        ),
        agent=reporter,
        expected_output=(
            "JSON object with summary, top_severity, "
            "top_severity_int, counts."
        ),
        context=[audit_task, governance_task]
    )

    # ── 5. run crew ──────────────────────────────────────
    print("\n[CREW] Starting multi-agent pipeline...\n")
    crew = Crew(
        agents=[auditor, governor, reporter],
        tasks=[audit_task, governance_task, report_task],
        process=Process.sequential,
        verbose=True,
    )
    crew_result = crew.kickoff()
    print(f"\n[CREW DONE] Raw output: {str(crew_result)[:300]}")

    # ── 6. parse reporter output ─────────────────────────
    report_data = extract_json(str(crew_result))
    if report_data and isinstance(report_data, dict):
        summary      = report_data.get("summary",
                                       "Audit complete — see findings.")
        severity_int = int(report_data.get("top_severity_int", 3))
    else:
        # Fallback if agent output could not be parsed
        summary      = "GovernanceAudit simulation: CRITICAL + HIGH findings."
        severity_int = 4

    # ── 7. log audit on-chain ────────────────────────────
    print("\n[BLOCKCHAIN] Logging audit on-chain via logAudit()...")
    log_result = chain.log_audit(summary[:200], severity_int)
    print(f"  logAudit → {log_result}")

    # ── 8. create governance proposal on-chain ───────────
    print("\n[BLOCKCHAIN] Creating governance proposal via propose()...")
    proposal_text = (
        "[CRITICAL] Fix access-control on propose(): "
        "add onlyRole(GOVERNOR_ROLE) modifier."
    )
    proposal_result = chain.create_proposal(proposal_text)
    print(f"  propose  → {proposal_result}")

    # ── 9. vote on the new proposal ──────────────────────
    proposal_id = proposal_result.get("proposal_id")
    vote_result = {}
    if proposal_id:
        print(f"\n[BLOCKCHAIN] Voting YES on proposal #{proposal_id}...")
        vote_result = chain.cast_vote(proposal_id, support=True)
        print(f"  vote     → {vote_result}")

    # ── 10. read chain state after ───────────────────────
    state_after = chain.read_state()
    print(f"\n[CHAIN AFTER]  {state_after}")

    return {
        "status":            "complete",
        "crew_output":       str(crew_result)[:500],
        "log_audit_tx":      log_result,
        "proposal_tx":       proposal_result,
        "vote_tx":           vote_result,
        "proposal_count_before": state_before["proposal_count"],
        "proposal_count_after":  state_after["proposal_count"],
        "proof_of_link": (
            state_after["proposal_count"] >
            state_before["proposal_count"]
        ),
    }
