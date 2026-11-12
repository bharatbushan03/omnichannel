import base64
import hashlib
import hmac
import time

from security.webhook_signature import verify_recall_signature, verify_slack_style_signature

# Published Recall.ai / Standard Webhooks vector (Hookdeck guide).
_KNOWN_SECRET = "whsec_aG9va2RlY2stZ3VpZGUtdGVzdC1rZXktMzJieXRlcyE="
_KNOWN_BODY = (
    b'{"event":"bot.done","data":{"bot":{"id":"6c8b2f7a-1d2e-4f3a-9b8c-7d6e5f4a3b2c","metadata":{}}}}'
)
_KNOWN_HEADERS = {
    "webhook-id": "msg_2kRecallTest0001",
    "webhook-timestamp": "1761040800",
    "webhook-signature": "v1,2iHMjJ7i8bqfujiWPY8x+VV2JMmCoxLDU9JDZqj+dUw=",
}


def _sign(body: bytes, secret: str, msg_id: str, timestamp: str) -> dict[str, str]:
    key = base64.b64decode(secret[len("whsec_") :], validate=True)
    signed = f"{msg_id}.{timestamp}.{body.decode('utf-8')}".encode("utf-8")
    digest = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode("ascii")
    return {
        "webhook-id": msg_id,
        "webhook-timestamp": timestamp,
        "webhook-signature": f"v1,{digest}",
    }


def test_known_recall_signature_vector():
    assert verify_recall_signature(_KNOWN_BODY, _KNOWN_HEADERS, _KNOWN_SECRET, now=1761040800)


def test_recall_rejects_tampered_body():
    headers = _sign(b'{"event":"transcript.data"}', _KNOWN_SECRET, "msg_1", "1761040800")
    assert verify_recall_signature(b'{"event":"transcript.data"}', headers, _KNOWN_SECRET, now=1761040800)
    assert not verify_recall_signature(b'{"event":"other"}', headers, _KNOWN_SECRET, now=1761040800)


def test_recall_accepts_svix_header_names():
    headers = _sign(b"{}", _KNOWN_SECRET, "msg_svix", "1761040800")
    renamed = {
        "svix-id": headers["webhook-id"],
        "svix-timestamp": headers["webhook-timestamp"],
        "svix-signature": headers["webhook-signature"],
    }
    assert verify_recall_signature(b"{}", renamed, _KNOWN_SECRET, now=1761040800)


def test_recall_rejects_stale_timestamp_and_bad_secret():
    headers = _sign(b"{}", _KNOWN_SECRET, "msg_old", "1761040800")
    assert not verify_recall_signature(b"{}", headers, _KNOWN_SECRET, now=1761040800 + 301)
    assert not verify_recall_signature(b"{}", headers, "not-a-whsec-secret", now=1761040800)
    assert not verify_recall_signature(b"{}", {}, _KNOWN_SECRET, now=1761040800)


def test_slack_style_signature_round_trip():
    body = b'{"type":"event_callback"}'
    secret = "slack-secret"
    timestamp = "1700000000"
    signed = f"v0:{timestamp}:{body.decode()}".encode()
    signature = "v0=" + hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    assert verify_slack_style_signature(body, timestamp, signature, secret, now=1700000000)
    assert not verify_slack_style_signature(body, timestamp, signature, "other", now=1700000000)
    assert not verify_slack_style_signature(body, timestamp, signature, secret, now=1700000000 + 600)
    assert not verify_slack_style_signature(body, "nope", signature, secret, now=time.time())
