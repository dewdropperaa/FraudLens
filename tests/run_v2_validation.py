from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "tests" / "artifacts" / "v2_validation_report.json"
DASHBOARD_PATH = ROOT / "tests" / "artifacts" / "v2_dashboard.json"
TEST_TARGET = "tests/test_v2_validation_suite.py"


def _run_validation_suite() -> int:
    cmd = [sys.executable, "-m", "pytest", TEST_TARGET, "-q"]
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    return completed.returncode


def _read_json(path: Path) -> Dict[str, Any] | List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_summary(report: Dict[str, Any], dashboard: List[Dict[str, Any]]) -> None:
    retry_distribution = report.get("retry_distribution", {})
    perf = report.get("agent_performance_metrics", {})
    latencies = perf.get("avg_latency_ms_per_agent", {})
    avg_latency = sum(latencies.values()) / len(latencies) if latencies else 0.0

    top_failures: Dict[str, int] = {}
    for row in dashboard:
        for failure in row.get("failure_points", []):
            top_failures[failure] = top_failures.get(failure, 0) + 1
    ranked_failures = sorted(top_failures.items(), key=lambda item: item[1], reverse=True)[:3]

    print("\n=== ClaimGuard v2 Validation Summary ===")
    print(f"Accuracy            : {_pct(float(report.get('accuracy', 0.0)))}")
    print(f"False Positive Rate : {_pct(float(report.get('false_positive_rate', 0.0)))}")
    print(f"False Negative Rate : {_pct(float(report.get('false_negative_rate', 0.0)))}")
    print(f"Average Ts          : {float(report.get('avg_Ts', 0.0)):.2f}")
    print(
        "Retry Distribution  : "
        f"0={retry_distribution.get('0', 0)}, "
        f"1={retry_distribution.get('1', 0)}, "
        f"2={retry_distribution.get('2', 0)}, "
        f"3={retry_distribution.get('3', 0)}"
    )
    print(f"Avg Agent Latency   : {avg_latency:.2f} ms")
    print(f"Avg Confidence Drift: {float(perf.get('avg_confidence_drift', 0.0)):.4f}")
    print(f"Avg Contradictions  : {float(perf.get('avg_contradiction_frequency', 0.0)):.4f}")

    if ranked_failures:
        print("Top Failure Points  :")
        for reason, count in ranked_failures:
            print(f"  - {reason} ({count})")
    else:
        print("Top Failure Points  : none")

    print(f"Claims Simulated    : {len(dashboard)}")
    print(f"Report Artifact     : {REPORT_PATH}")
    print(f"Dashboard Artifact  : {DASHBOARD_PATH}")
    print("========================================\n")


def main() -> None:
    code = _run_validation_suite()
    if code != 0:
        raise SystemExit(code)

    report_raw = _read_json(REPORT_PATH)
    dashboard_raw = _read_json(DASHBOARD_PATH)
    if not isinstance(report_raw, dict):
        raise TypeError("Validation report must be a JSON object.")
    if not isinstance(dashboard_raw, list):
        raise TypeError("Dashboard artifact must be a JSON array.")
    _print_summary(report_raw, dashboard_raw)


if __name__ == "__main__":
    main()
