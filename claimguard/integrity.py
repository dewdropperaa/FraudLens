from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _canonical_source_for_module(repo_root: Path, module_name: str) -> Tuple[Path, Path]:
    parts = module_name.split(".")
    base = repo_root.joinpath(*parts)
    py_file = base.with_suffix(".py")
    pkg_init = base / "__init__.py"
    return py_file.resolve(), pkg_init.resolve()


def _iter_loaded_package_modules(package_prefix: str) -> Iterable[Tuple[str, str]]:
    for name, module in sorted(sys.modules.items()):
        if not (name == package_prefix or name.startswith(f"{package_prefix}.")):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file:
            yield name, str(Path(module_file).resolve())


def verify_single_source_execution(package_prefix: str = "claimguard") -> None:
    """
    Fail fast when modules are loaded from non-canonical paths.

    This prevents shadowed/duplicate execution paths such as importing the same
    logical module from stale artifacts or alternate roots.
    """
    repo_root = Path(__file__).resolve().parent.parent
    importlib.import_module(package_prefix)

    loaded = list(_iter_loaded_package_modules(package_prefix))
    print("[Integrity] Loaded module paths:")
    for module_name, module_path in loaded:
        print(f"[Integrity] {module_name} -> {module_path}")

    errors: List[str] = []
    seen_paths_by_module: Dict[str, str] = {}
    for module_name, module_path in loaded:
        if module_name in seen_paths_by_module and seen_paths_by_module[module_name] != module_path:
            errors.append(
                f"Module '{module_name}' loaded from multiple paths: "
                f"{seen_paths_by_module[module_name]} | {module_path}"
            )
            continue
        seen_paths_by_module[module_name] = module_path

        actual = Path(module_path).resolve()
        canonical_py, canonical_pkg = _canonical_source_for_module(repo_root, module_name)
        if actual not in {canonical_py, canonical_pkg}:
            errors.append(
                f"Module '{module_name}' loaded from non-canonical source '{actual}'. "
                f"Expected '{canonical_py}' or '{canonical_pkg}'."
            )

    if errors:
        detail = "\n".join(f"- {msg}" for msg in errors)
        raise RuntimeError(f"Repository integrity violation detected:\n{detail}")
