"""Signed, short-lived capabilities issued by the web confirmation endpoint."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time

_used: dict[str, float] = {}
_lock = threading.Lock()


def _payload(name: str, arguments: dict) -> str:
    return json.dumps([name, arguments], sort_keys=True, separators=(",", ":"))


def issue(secret: str, name: str, arguments: dict) -> str:
    if not secret:
        raise ValueError("Mock action approvals are not configured")
    body = f"{int(time.time()) + 60}.{secrets.token_hex(16)}"
    signature = hmac.new(
        secret.encode(), f"{body}.{_payload(name, arguments)}".encode(), hashlib.sha256
    ).hexdigest()
    return f"{body}.{signature}"


def consume(secret: str, name: str, arguments: dict, token: str) -> None:
    try:
        expiry, nonce, signature = token.split(".")
        expires = int(expiry)
    except (ValueError, AttributeError) as exc:
        raise ValueError("Use the web preview confirmation button to approve this action") from exc
    body = f"{expiry}.{nonce}"
    expected = hmac.new(
        secret.encode(), f"{body}.{_payload(name, arguments)}".encode(), hashlib.sha256
    ).hexdigest()
    if not secret or expires < time.time() or not hmac.compare_digest(signature, expected):
        raise ValueError("Approval is invalid, expired, or does not match this action")
    with _lock:
        for key in list(_used):
            if _used[key] < time.time():
                del _used[key]
        if token in _used:
            raise ValueError("Approval has already been consumed")
        _used[token] = expires
