import hashlib
import hmac
import json
import time

from fastapi.testclient import TestClient

import ingestion.meeting_listener as meeting_listener
import ingestion.slack_listener as slack_listener
import ingestion.teams_listener as teams_listener
from ingestion.rate_limit import SlidingWindowRateLimiter
from security.consent import DISCLOSURE_TEXT, Jurisdiction, MeetingConsentTracker


class _Pipeline:
    def __init__(self):
        self.calls = []

    async def process_meeting_segment(self, **kwargs):
        self.calls.append(kwargs)

    async def process_message(self, **kwargs):
        self.calls.append(kwargs)


def _recall_headers(body: bytes, secret: str) -> dict[str, str]:
    import base64

    timestamp = str(int(time.time()))
    key = base64.b64decode(secret[len("whsec_") :], validate=True)
    signed = f"msg_test.{timestamp}.{body.decode('utf-8')}".encode("utf-8")
    digest = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode("ascii")
    return {
        "webhook-id": "msg_test",
        "webhook-timestamp": timestamp,
        "webhook-signature": f"v1,{digest}",
    }


def _slack_headers(body: bytes, secret: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    signed = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    signature = "v0=" + hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return {"x-slack-request-timestamp": timestamp, "x-slack-signature": signature}


def test_teams_meeting_url_is_not_labeled_zoom():
    assert meeting_listener.source_for_meeting_url("https://teams.microsoft.com/l/meetup-join/x").value == "teams_meeting"
    assert meeting_listener.source_for_meeting_url("https://zoom.us/j/1").value == "zoom"


def test_create_bot_payload_discloses_and_uses_deepgram():
    payload = meeting_listener.build_create_bot_payload(
        "https://meet.google.com/abc-defg-hij",
        "ws",
        "Standup",
        jurisdiction="two_party",
        join_at="2026-11-11T12:00:00+00:00",
    )
    assert payload["chat"]["on_bot_join"]["message"] == DISCLOSURE_TEXT
    assert payload["chat"]["on_participant_join"]["message"] == DISCLOSURE_TEXT
    provider = payload["recording_config"]["transcript"]["provider"]["deepgram_streaming"]
    assert provider["model"] == "nova-2"
    assert provider["diarize"] is True
    assert payload["recording_config"]["video_mixed_mp4"] is None
    assert "transcription_options" not in payload
    assert payload["metadata"]["platform"] == "google_meet"


def test_transcript_words_are_joined():
    payload = {
        "event": "transcript.data",
        "data": {
            "bot": {"id": "bot-1", "metadata": {"workspace_id": "ws", "meeting_title": "Standup", "platform": "zoom"}},
            "data": {
                "words": [
                    {"text": "Ship", "start_timestamp": {"relative": 12.5}},
                    {"text": "the", "start_timestamp": {"relative": 12.8}},
                    {"text": "fix", "start_timestamp": {"relative": 13.0}},
                ],
                "participant": {"name": "Ada"},
            },
        },
    }
    segments = meeting_listener.parse_transcript_segments(payload)
    assert segments == [
        {
            "text": "Ship the fix",
            "speaker": "Ada",
            "offset_s": 12.5,
            "meeting_id": "bot-1",
            "workspace_id": "ws",
            "meeting_title": "Standup",
            "platform": "zoom",
        }
    ]


def test_recall_webhook_rejects_a_bad_signature(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("RECALL_WEBHOOK_SECRET", "whsec_" + "aGVsbG8=")
    meeting_listener._limiter = SlidingWindowRateLimiter(max_requests=20)
    client = TestClient(meeting_listener.app)
    response = client.post("/meetings/webhook", json={"event": "transcript.data"})
    assert response.status_code == 401


def test_recall_webhook_ingests_after_disclosure(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    secret = "whsec_aG9va2RlY2stZ3VpZGUtdGVzdC1rZXktMzJieXRlcyE="
    monkeypatch.setenv("RECALL_WEBHOOK_SECRET", secret)
    pipeline = _Pipeline()
    meeting_listener._pipeline = pipeline
    meeting_listener._consent = MeetingConsentTracker(jurisdiction=Jurisdiction.TWO_PARTY)
    meeting_listener._consent.mark_announced("bot-9", "ws")
    meeting_listener._limiter = SlidingWindowRateLimiter(max_requests=20)

    body = {
        "event": "transcript.data",
        "data": {
            "bot": {
                "id": "bot-9",
                "metadata": {"workspace_id": "ws", "meeting_title": "Standup", "platform": "zoom"},
            },
            "transcript": {"text": "We decided to migrate the deploy pipeline.", "speaker": "Ada", "start_timestamp": {"relative": 3}},
        },
    }
    raw = json.dumps(body).encode("utf-8")
    client = TestClient(meeting_listener.app)
    response = client.post("/meetings/webhook", content=raw, headers=_recall_headers(raw, secret))
    assert response.status_code == 200
    assert pipeline.calls[0]["text"].startswith("We decided")
    assert pipeline.calls[0]["speaker_label"] == "Ada"


def test_recall_webhook_stops_on_objection(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("RECALL_API_KEY", raising=False)
    secret = "whsec_aG9va2RlY2stZ3VpZGUtdGVzdC1rZXktMzJieXRlcyE="
    monkeypatch.setenv("RECALL_WEBHOOK_SECRET", secret)
    pipeline = _Pipeline()
    left = []

    async def _leave(bot_id, client=None):
        left.append(bot_id)

    meeting_listener._pipeline = pipeline
    meeting_listener.leave_meeting_bot = _leave
    meeting_listener._consent = MeetingConsentTracker(jurisdiction=Jurisdiction.TWO_PARTY)
    meeting_listener._consent.mark_announced("bot-9", "ws")
    meeting_listener._limiter = SlidingWindowRateLimiter(max_requests=20)

    body = {
        "event": "transcript.data",
        "data": {
            "bot": {"id": "bot-9", "metadata": {"workspace_id": "ws", "meeting_title": "Standup"}},
            "transcript": {"text": "Please stop recording", "speaker": "Ada"},
        },
    }
    raw = json.dumps(body).encode("utf-8")
    client = TestClient(meeting_listener.app)
    response = client.post("/meetings/webhook", content=raw, headers=_recall_headers(raw, secret))
    assert response.status_code == 200
    assert pipeline.calls == []
    assert left == ["bot-9"]


def test_meeting_start_requires_admin_token(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("INGESTION_ADMIN_TOKEN", raising=False)
    meeting_listener._limiter = SlidingWindowRateLimiter(max_requests=20)
    client = TestClient(meeting_listener.app)
    response = client.post("/meetings/start", json={"meeting_url": "https://zoom.us/j/1", "workspace_id": "ws", "meeting_title": "Standup"})
    assert response.status_code == 503


def test_teams_webhook_verifies_signature_and_parses_activity(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("TEAMS_WEBHOOK_SECRET", "teams-secret")
    pipeline = _Pipeline()
    teams_listener._pipeline = pipeline
    teams_listener._limiter = SlidingWindowRateLimiter(max_requests=20)
    body = {
        "type": "message",
        "text": "We decided to freeze the API.",
        "from": {"id": "u1", "name": "Ada"},
        "recipient": {"id": "bot"},
        "channelData": {"team": {"id": "team-1"}, "channel": {"id": "ch-1", "name": "Engineering"}},
        "conversation": {"id": "thread-1"},
    }
    raw = json.dumps(body).encode("utf-8")
    headers = _slack_headers(raw, "teams-secret")
    client = TestClient(teams_listener.app)
    rejected = client.post("/teams/messages", content=raw, headers={"x-teams-timestamp": "1", "x-teams-signature": "v0=nope"})
    assert rejected.status_code == 401
    accepted = client.post(
        "/teams/messages",
        content=raw,
        headers={
            "x-teams-timestamp": headers["x-slack-request-timestamp"],
            "x-teams-signature": headers["x-slack-signature"],
        },
    )
    assert accepted.status_code == 200
    assert pipeline.calls[0]["channel_name"] == "Engineering"
    assert pipeline.calls[0]["workspace_id"] == "team-1"


def test_slack_names_resolve_and_bad_signature_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "slack-secret")

    class _Web:
        def conversations_info(self, channel):
            return {"channel": {"name": "eng-payments"}}

        def users_info(self, user):
            return {"user": {"profile": {"display_name": "Priya"}}}

    assert slack_listener.resolve_channel_name("C-test-1", client=_Web()) == "#eng-payments"
    assert slack_listener.resolve_user_name("U-test-1", client=_Web()) == "Priya"

    pipeline = _Pipeline()
    slack_listener._pipeline = pipeline
    slack_listener._limiter = SlidingWindowRateLimiter(max_requests=20)
    body = {"type": "url_verification", "challenge": "abc"}
    raw = json.dumps(body).encode("utf-8")
    client = TestClient(slack_listener.app)
    bad = client.post("/slack/events", content=raw, headers={"x-slack-request-timestamp": "1", "x-slack-signature": "v0=nope"})
    assert bad.status_code == 401
    good = client.post("/slack/events", content=raw, headers=_slack_headers(raw, "slack-secret"))
    assert good.status_code == 200
    assert good.json()["challenge"] == "abc"
