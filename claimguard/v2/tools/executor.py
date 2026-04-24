from __future__ import annotations

from typing import Any, Dict

from claimguard.v2.tools.registry import get_tool


def execute_tool(tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    print("[TOOL START]")
    print(f"[TOOL NAME] {tool_name}")
    print(f"[TOOL INPUT] {input_data}")

    tool = get_tool(tool_name)
    if tool is None:
        return {
            "tool": tool_name,
            "status": "ERROR",
            "output": {},
            "confidence": 0.0,
            "reason": f"Tool '{tool_name}' not found",
        }

    try:
        payload = tool(input_data if isinstance(input_data, dict) else {})
        if not isinstance(payload, dict):
            raise RuntimeError("Tool returned non-dict output")
    except Exception as exc:
        error_payload = {
            "tool": tool_name,
            "status": "ERROR",
            "output": {},
            "confidence": 0.0,
            "reason": str(exc),
        }
        print(f"[TOOL OUTPUT] {error_payload}")
        return error_payload

    merged = {
        "tool": str(payload.get("tool") or tool_name),
        "status": str(payload.get("status") or "DONE").upper(),
        "output": payload.get("output") if isinstance(payload.get("output"), dict) else {},
        "confidence": float(payload.get("confidence") or 0.0),
    }
    if merged["status"] not in {"DONE", "ERROR"}:
        merged["status"] = "ERROR"
        merged["output"] = {}
        merged["confidence"] = 0.0
        merged["reason"] = "Tool returned invalid status"
    if payload.get("reason") is not None:
        merged["reason"] = str(payload.get("reason"))

    print(f"[TOOL OUTPUT] {merged}")
    return merged
