"""
Jev / TypeSafe System One adapter for OPA.

This is intentionally provider-agnostic because the exact JEV inference endpoint
and authentication contract are environment-specific. Set:
    JEV_ENABLED=true
    JEV_API_URL=https://...
    JEV_API_KEY=...
    JEV_MODEL=...

The client is fail-soft: when disabled, unconfigured, or unavailable, callers
receive None and OPA keeps using its existing Groq/deterministic paths.

Expected JSON response shape (or nested under `decision` / `data` / `output`):
{
  "intent": "generate_exam_plan",
  "confidence": 0.98,
  "entities": {"duration_min": 180, "scope": "upcoming_exams"},
  "requires_confirmation": false,
  "reason_code": null
}
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from pydantic import BaseModel, Field, ValidationError


class JevDecision(BaseModel):
    intent: str = Field(min_length=1, max_length=80)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    reason_code: Optional[str] = None


def jev_enabled() -> bool:
    return os.getenv("JEV_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("decision", "data", "output", "result"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    return payload


def classify_message(message: str, context: Optional[Dict[str, Any]] = None) -> Optional[JevDecision]:
    """
    Call JEV for a typed routing decision.

    This function never raises on provider/network/schema failures; OPA's
    existing Groq + deterministic fallback remains authoritative.
    """
    if not jev_enabled():
        return None

    url = os.getenv("JEV_API_URL", "").strip()
    if not url:
        return None

    payload = {
        "model": os.getenv("JEV_MODEL", "").strip() or None,
        "message": message,
        "context": context or {},
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = os.getenv("JEV_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = float(os.getenv("JEV_TIMEOUT_SECONDS", "8"))
    req = urlrequest.Request(url, data=body, headers=headers, method="POST")

    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        decoded = json.loads(raw)
        decision = _unwrap(decoded)
        return JevDecision.model_validate(decision)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, ValidationError, json.JSONDecodeError):
        return None
