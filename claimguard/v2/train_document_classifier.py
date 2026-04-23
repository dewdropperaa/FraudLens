from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from claimguard.v2.document_classifier import train_document_classifier


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ClaimGuard document classifier from JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL rows with {extracted_text,label}.")
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=("logistic_regression", "random_forest"),
        help="Classifier backend.",
    )
    args = parser.parse_args()
    dataset_path = Path(args.dataset).resolve()
    rows = _load_jsonl(dataset_path)
    report = train_document_classifier(rows, model_type=args.model_type)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
