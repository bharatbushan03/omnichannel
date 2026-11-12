"""Webhook signature checks for Slack-style HMAC and Recall.ai (Standard Webhooks / Svix).

Recall.ai signs the raw body as ``{id}.{timestamp}.{payload}`` with HMAC-SHA256.
The key is the base64 portion of a ``whsec_`` secret. Current workspaces send
``webhook-id``, ``webhook-timestamp``, and ``webhook-signature``. Accounts that
still use Svix send the same values under ``svix-id``, ``svix-timestamp``, and
``svix-signature``. Both header sets are accepted.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
from collections.abc import Mapping


def _headers_lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _header(headers: Mapping[str, str], name: str) -> str:
    return _headers_lower(headers).get(name.lower(), "")


def verify_slack_style_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    secret: str,
    *,
    now: float | None = None,
    tolerance_s: int = 300,
) -> bool:
    """Slack (and the Teams gateway) sign ``v0:{timestamp}:{raw body}`` with HMAC-SHA256.

    The signature header value is ``v0=`` plus the hex digest. Requests older than
    ``tolerance_s`` are rejected so a captured body cannot be replayed.
    """
    if not secret or not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else now
    if abs(current - ts) > tolerance_s:
        return False
    try:
        payload = body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    signed = f"v0:{timestamp}:{payload}".encode("utf-8")
    computed = "v0=" + hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)


def verify_recall_signature(
    body: bytes,
    headers: Mapping[str, str],
    secret: str,
    *,
    now: float | None = None,
    tolerance_s: int = 300,
) -> bool:
    """Return True only when the Recall.ai / Standard Webhooks signature matches the raw body."""
    if not secret or not secret.startswith("whsec_"):
        return False

    msg_id = _header(headers, "webhook-id") or _header(headers, "svix-id")
    msg_timestamp = _header(headers, "webhook-timestamp") or _header(headers, "svix-timestamp")
    msg_signature = _header(headers, "webhook-signature") or _header(headers, "svix-signature")
    if not msg_id or not msg_timestamp or not msg_signature:
        return False

    try:
        ts = int(msg_timestamp)
    except ValueError:
        return False
    current = time.time() if now is None else now
    if abs(current - ts) > tolerance_s:
        return False

    try:
        key = base64.b64decode(secret[len("whsec_") :], validate=True)
        payload = body.decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return False

    signed = f"{msg_id}.{msg_timestamp}.{payload}".encode("utf-8")
    expected = hmac.new(key, signed, hashlib.sha256).digest()

    for part in msg_signature.split(" "):
        version, separator, signature = part.partition(",")
        if separator != "," or version != "v1" or not signature:
            continue
        try:
            provided = base64.b64decode(signature, validate=True)
        except ValueError:
            continue
        if len(provided) == len(expected) and hmac.compare_digest(provided, expected):
            return True
    return False
