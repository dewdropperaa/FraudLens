from __future__ import annotations

from typing import Any, Callable, Dict

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]

_TOOL_REGISTRY: Dict[str, ToolFn] = {}


def register_tool(name: str, function: ToolFn) -> None:
    key = str(name or "").strip()
    if not key:
        raise ValueError("Tool name must be non-empty")
    if not callable(function):
        raise TypeError("Tool function must be callable")
    _TOOL_REGISTRY[key] = function


def get_tool(name: str) -> ToolFn | None:
    key = str(name or "").strip()
    if not key:
        return None
    return _TOOL_REGISTRY.get(key)


def list_tools() -> list[str]:
    return sorted(_TOOL_REGISTRY.keys())
