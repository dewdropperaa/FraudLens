from __future__ import annotations

import re
import sys
import hashlib
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}
IMPORT_LINE = re.compile(r"^\s*(from|import)\s+.+$", re.MULTILINE)


def _iter_files() -> List[Path]:
    files: List[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _check_pycache_and_pyc(files: List[Path]) -> List[str]:
    failures: List[str] = []
    for path in files:
        if "__pycache__" in path.parts:
            failures.append(f"__pycache__ directory artifact found: {path}")
        if path.suffix == ".pyc":
            failures.append(f"Compiled .pyc artifact found: {path}")
    return failures


def _check_duplicate_filenames(files: List[Path]) -> List[str]:
    by_name_and_hash: Dict[str, List[Path]] = {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = f"{path.name.lower()}::{digest}"
        by_name_and_hash.setdefault(key, []).append(path)
    failures: List[str] = []
    for key, paths in sorted(by_name_and_hash.items()):
        if len(paths) <= 1:
            continue
        name = key.split("::", 1)[0]
        if name == "__init__.py":
            continue
        rendered = ", ".join(str(p.relative_to(ROOT)) for p in paths)
        failures.append(f"Duplicate file content detected ({name}): {rendered}")
    return failures


def _check_mixed_path_imports(files: List[Path]) -> List[str]:
    failures: List[str] = []
    for path in files:
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in IMPORT_LINE.findall(text):
            _ = line
        for lineno, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped.startswith(("from ", "import ")):
                continue
            if "\\" in stripped:
                failures.append(
                    f"Mixed path separator in import at {path.relative_to(ROOT)}:{lineno}: {stripped}"
                )
            if stripped.startswith("from .") or stripped.startswith("import ."):
                failures.append(
                    f"Relative import detected (canonical root required) at "
                    f"{path.relative_to(ROOT)}:{lineno}: {stripped}"
                )
    return failures


def main() -> int:
    files = _iter_files()
    failures: List[str] = []
    failures.extend(_check_pycache_and_pyc(files))
    failures.extend(_check_duplicate_filenames(files))
    failures.extend(_check_mixed_path_imports(files))

    if failures:
        print("Repository integrity guard failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Repository integrity guard passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
