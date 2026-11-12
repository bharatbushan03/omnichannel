"""Discord (ed25519) and Meta/Jira (sha256 HMAC) webhook checks."""
from __future__ import annotations

import hashlib
import hmac


def verify_sha256_signature(body: bytes, header: str, secret: str) -> bool:
    """Header value is ``sha256=<hex>``. Used by WhatsApp and Jira Cloud webhooks."""
    if not secret or not header or not header.startswith("sha256="):
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(header[len("sha256=") :], digest)


def verify_discord_signature(body: bytes, signature: str, timestamp: str, public_key_hex: str) -> bool:
    if not public_key_hex or not signature or not timestamp:
        return False
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError:
        return False
    try:
        key = VerifyKey(bytes.fromhex(public_key_hex))
        key.verify(timestamp.encode("utf-8") + body, bytes.fromhex(signature))
    except (BadSignatureError, ValueError):
        return False
    return True
