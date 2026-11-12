import pytest

from security.encryption import require_encryption_at_rest
from security.pii_redaction import PIIRedactor, RedactionError
from voice_agent.scope import parse_room_metadata


def test_production_refuses_unconfigured_encryption(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ENCRYPTION_AT_REST", "unconfigured")
    with pytest.raises(RuntimeError):
        require_encryption_at_rest()


def test_disk_attestation_allows_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ENCRYPTION_AT_REST", "disk")
    require_encryption_at_rest()


def test_room_metadata_json_and_plain_workspace():
    scope = parse_room_metadata('{"workspace_id":"ws1","user_id":"u1","participant_user_ids":["u1","u2"],"allowed_channel_ids":["C9"]}')
    assert scope.workspace_id == "ws1"
    assert scope.user_id == "u1"
    assert scope.participant_user_ids == ["u1", "u2"]
    assert scope.allowed_channel_ids == ["C9"]
    assert parse_room_metadata("ws_plain").workspace_id == "ws_plain"
    assert parse_room_metadata("").workspace_id == "ws_demo"
    assert parse_room_metadata({"workspace_id": "ws-dict", "user_id": "u9"}).workspace_id == "ws-dict"


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _Response("Ada [REDACTED_NAME] owns the service.")


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _Client:
    def __init__(self):
        self.chat = _Chat()


def test_llm_pass_replaces_names_and_sets_max_tokens():
    client = _Client()
    result = PIIRedactor(use_llm_pass=True, llm_client=client, model="test-model").redact("Ada owns the service.")
    assert "[REDACTED_NAME]" in result.redacted_text
    assert result.redaction_count >= 1
    assert client.chat.completions.kwargs["max_tokens"] == 2048
    assert client.chat.completions.kwargs["model"] == "test-model"


def test_llm_pass_without_a_client_fails_closed():
    with pytest.raises(RedactionError):
        PIIRedactor(use_llm_pass=True, llm_client=None).redact("Ada owns the service.")
