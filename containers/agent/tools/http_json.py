"""Small JSON-over-HTTP helper for the OptArena Claude Code tools."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def endpoint(tool: str) -> str:
    override = os.environ.get(f"OPTARENA_{tool.upper()}_ENDPOINT", "").strip()
    if override:
        return override
    base = os.environ.get("OPTARENA_AGENT_API_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("OPTARENA_AGENT_API_URL must be set, or set the per-tool endpoint override")
    return f"{base}/{tool}"


def post_json(url: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url,
                                 data=data,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"},
                                 method="POST")
    seconds = timeout if timeout is not None else float(os.environ.get("TOOL_TIMEOUT_SECONDS", "120"))
    try:
        with urllib.request.urlopen(req, timeout=seconds) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {"ok": True, "status": response.status}
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"ok": True, "status": response.status, "text": body}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            details: Any = json.loads(text)
        except json.JSONDecodeError:
            details = text
        return {"ok": False, "status": exc.code, "error": details}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc.reason)}
