"""
Entry point. Run: python simulate.py

What this proves:
  1. Ollama runs 3 CrewAI agents locally (no OpenAI)
  2. AuditorAgent finds vulnerabilities in GovernanceAudit.sol
  3. GovernorAgent drafts proposals for CRITICAL/HIGH findings
  4. ReporterAgent summarises the audit
  5. logAudit() is called on-chain (GovernanceAudit contract)
  6. propose() is called on-chain (GovernanceAudit contract)
  7. vote() is called on-chain
  8. proposal_count increases — proof agents changed blockchain state
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from crew import run_simulation

console = Console()


def print_preflight_help(failed: list):
    """Print guidance for fixing preflight check failures."""
    console.print("\n[yellow]How to fix:[/yellow]")
    fixes = {
        "rpc_connected":
            "Check ALCHEMY_URL in .env",
        "wallet_has_balance":
            "Get Sepolia ETH: https://sepoliafaucet.com",
        "has_auditor_role":
            "In Remix → AccessControl → grantRole("
            "AUDITOR_ROLE_bytes32, your_wallet_address)",
        "has_governor_role":
            "In Remix → AccessControl → grantRole("
            "GOVERNOR_ROLE_bytes32, your_wallet_address)",
        "governance_deployed":
            "Add GOVERNANCE_AUDIT_ADDRESS to .env",
        "access_ctrl_deployed":
            "Add ACCESS_CONTROL_ADDRESS to .env",
    }
    for f in failed:
        if f in fixes:
            console.print(f"  [red]•[/red] {f}: {fixes[f]}")


def main():
    console.print(Panel(
        "[bold]Agentic AI — Governance & Self-Audit Simulation[/bold]\n"
        "[dim]Ollama (local LLM) + CrewAI (3 agents) + "
        "Polygon Amoy (2 contracts)[/dim]",
        style="blue", expand=False
    ))

    try:
        result = run_simulation()
    except AssertionError as e:
        console.print(f"\n[red]CONFIG ERROR: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]UNEXPECTED ERROR: {e}[/red]")
        console.print("[dim]Run with: python simulate.py 2>&1 | tee sim.log[/dim]")
        sys.exit(1)

    # ── preflight failed ─────────────────────────────────
    if result["status"] == "preflight_failed":
        console.print("\n[red]PREFLIGHT FAILED[/red]")
        print_preflight_help(result["failed_checks"])
        sys.exit(1)

    # ── results table ─────────────────────────────────────
    t = Table(title="Simulation Results", show_lines=True)
    t.add_column("Step",   style="cyan",  min_width=22)
    t.add_column("Result", style="white", min_width=40)

    def tx_cell(tx: dict) -> str:
        if "error" in tx:
            return f"[red]ERROR: {tx['error'][:60]}[/red]"
        status = "[green]SUCCESS[/green]" if tx.get("status") == 1 \
                 else "[red]FAILED[/red]"
        return f"{status} | tx: {tx.get('tx_hash','?')[:20]}..."

    t.add_row("Agents ran",        "[green]YES[/green]")
    t.add_row("logAudit() on-chain", tx_cell(result["log_audit_tx"]))
    t.add_row("propose() on-chain",  tx_cell(result["proposal_tx"]))
    t.add_row("vote() on-chain",
              tx_cell(result["vote_tx"]) if result["vote_tx"]
              else "[yellow]skipped[/yellow]")
    t.add_row("Proposals before",
              str(result["proposal_count_before"]))
    t.add_row("Proposals after",
              str(result["proposal_count_after"]))
    t.add_row("Blockchain linked",
              "[green]YES[/green]" if result["proof_of_link"]
              else "[red]NO — tx may have failed[/red]")

    console.print(t)

    # ── final verdict ─────────────────────────────────────
    log_ok  = result["log_audit_tx"].get("status") == 1
    prop_ok = result["proposal_tx"].get("status") == 1
    linked  = result["proof_of_link"]

    if log_ok and prop_ok and linked:
        console.print(Panel(
            "[bold green]SUCCESS[/bold green]\n"
            "Multi-agent system is linked to the blockchain.\n"
            "Agents audited the contract, logged findings on-chain,\n"
            "created a governance proposal, and cast a vote.",
            style="green", expand=False
        ))
        console.print(
            f"\n[dim]Verify on-chain: "
            f"https://sepolia.etherscan.io/tx/"
            f"{result['log_audit_tx'].get('tx_hash','')}"
            "[/dim]"
        )
    else:
        console.print(Panel(
            "[bold yellow]PARTIAL[/bold yellow]\n"
            "Agents ran but some blockchain transactions failed.\n"
            "Check the table above and fix the failing steps.",
            style="yellow", expand=False
        ))


if __name__ == "__main__":
    main()
