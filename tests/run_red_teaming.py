from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claimguard.v2.redteam import StrictModeConfig, run_red_teaming, write_redteam_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ClaimGuard full red teaming engine.")
    parser.add_argument("--claims", type=int, default=100, help="Number of synthetic claims (50-100+ recommended).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--artifact-dir", type=str, default="tests/artifacts", help="Artifact output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Disable optional debug visualizations.")
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Backward-compatible no-op: real LLM mode is now the default (Ollama).",
    )
    parser.add_argument(
        "--simulated-agents",
        action="store_true",
        help="Explicitly enable deterministic simulation mode (disables Ollama calls).",
    )
    parser.add_argument("--critical-threshold", type=int, default=5, help="Strict mode critical failure threshold.")
    parser.add_argument("--hallucination-max", type=float, default=0.2, help="Max hallucination rate in strict mode.")
    parser.add_argument("--fraud-target", type=float, default=0.7, help="Min fraud detection rate in strict mode.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    strict = StrictModeConfig(
        critical_failures_threshold=args.critical_threshold,
        hallucination_rate_max=args.hallucination_max,
        fraud_detection_target=args.fraud_target,
    )
    report = run_red_teaming(
        claim_count=args.claims,
        random_seed=args.seed,
        artifact_dir=args.artifact_dir,
        generate_visualizations=not args.no_plots,
        use_simulated_agents=bool(args.simulated_agents),
        strict_mode=strict,
    )

    output_path = write_redteam_report(report, artifact_dir=args.artifact_dir)
    print("\n=== ClaimGuard Red Teaming Summary ===")
    print(json.dumps(report["summary"], indent=2))
    print("Accuracy:", json.dumps(report["accuracy_metrics"], indent=2))
    print("Reasoning:", json.dumps(report["reasoning_metrics"], indent=2))
    print("Strict Mode:", json.dumps(report["strict_mode"], indent=2))
    print(f"Report Artifact: {output_path}")
    if report.get("visualizations"):
        print("Visualizations:")
        for name, path in report["visualizations"].items():
            print(f"  - {name}: {path}")
    print("======================================\n")

    if report["strict_mode"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
