import asyncio
import hashlib
import hmac
import json
from pathlib import Path

from fastapi.testclient import TestClient

import channels.gateway as gateway
import ingestion.meeting_listener as meeting_listener
import ingestion.slack_listener as slack_listener
import ingestion.teams_listener as teams_listener
from agent_runtime.addressing import (
    adf_to_text,
    discord_question,
    jira_event_text,
    jira_should_answer,
    meeting_should_answer,
    slack_should_answer,
    teams_should_answer,
)
from agent_runtime.dev_tools import (
    git_diff,
    git_log,
    git_show,
    git_status,
    list_repository,
    read_repository_file,
    run_tests,
    search_repository,
)
from agent_runtime.runner import TOOL_SCHEMAS, TurnContext, execute_tool, run_agent_turn
from ingestion.rate_limit import SlidingWindowRateLimiter
from security.consent import DISCLOSURE_TEXT, Jurisdiction, MeetingConsentTracker
from tests.test_webhooks import _recall_headers, _slack_headers


class _Pipeline:
    def __init__(self):
        self.calls = []

    async def process_message(self, **kwargs):
        self.calls.append(kwargs)

    async def process_meeting_segment(self, **kwargs):
        self.calls.append(kwargs)


def test_repository_tools_stay_inside_the_workspace(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello from the repo\n", encoding="utf-8")
    secret = tmp_path / ".env"
    secret.write_text("SUPERSECRETVALUE\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-agent-test.txt"
    outside.write_text("do not read\n", encoding="utf-8")
    try:
        assert "hello from the repo" in read_repository_file("notes.txt", tmp_path)
        assert read_repository_file("../outside-agent-test.txt", tmp_path) == "Path is outside the workspace."
        assert read_repository_file(".env", tmp_path) == "That file is not available to the agent."
        assert "notes.txt" in search_repository("hello from the repo", tmp_path)
        assert search_repository("SUPERSECRETVALUE", tmp_path) == "No matches."
        assert git_log(tmp_path, path="../outside-agent-test.txt") == "Path is outside the workspace."
    finally:
        outside.unlink(missing_ok=True)


def test_every_development_tool_is_offered_to_the_model():
    names = {item["function"]["name"] for item in TOOL_SCHEMAS}
    assert names == {
        "lookup_knowledge_base",
        "search_repository",
        "read_repository_file",
        "list_repository",
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "run_tests",
        "jira_get_issue",
        "jira_search",
    }


def test_development_tools_list_diff_show_and_run(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "notes.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET\n", encoding="utf-8")
    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    listing = list_repository(tmp_path)
    assert "pkg/" in listing
    assert "test_sample.py" in listing
    assert ".env" not in listing
    assert git_show(tmp_path, "--all") == "Revision is not valid."
    assert git_show(tmp_path, "HEAD..main") == "Revision is not valid."
    assert git_diff(tmp_path, "../outside") == "Path is outside the workspace."
    assert run_tests(tmp_path, "../test_sample.py") == "Path is outside the workspace."
    assert run_tests(tmp_path, "pkg/notes.txt") == "Tests must be a directory or a Python file."
    assert "1 passed" in run_tests(tmp_path, "test_sample.py")
    assert "not a git repository" in git_status(tmp_path).lower()

    shown = git_show(Path.cwd(), "HEAD")
    assert shown != "Revision is not valid."
    assert shown.split()[0]


def test_voice_agent_exposes_the_same_development_tools():
    from livekit.agents.llm.tool_context import find_function_tools

    from voice_agent.agent import KnowledgeAgent

    voice_names = {tool.info.name for tool in find_function_tools(KnowledgeAgent)}
    text_names = {item["function"]["name"] for item in TOOL_SCHEMAS}
    assert voice_names == text_names


def test_voice_agent_lists_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    from voice_agent.agent import KnowledgeAgent
    from voice_agent.scope import RoomScope

    agent = KnowledgeAgent(RoomScope(workspace_id="ws"))
    listing = asyncio.run(agent._use("list_repository", {"path": "."}))
    assert "notes.txt" in listing
    refused = asyncio.run(agent._use("read_repository_file", {"path": ".env"}))
    assert refused == "That file is not available to the agent."


def test_execute_tool_runs_list_and_refuses_a_bad_revision(tmp_path):
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    turn = TurnContext(channel="slack", workspace_id="ws", workspace_root=tmp_path)
    listing = asyncio.run(execute_tool("list_repository", {"path": "."}, turn))
    assert "notes.txt" in listing
    refused = asyncio.run(execute_tool("git_show", {"revision": "HEAD; rm -rf"}, turn))
    assert refused == "Revision is not valid."


def test_addressing_rules():
    assert slack_should_answer({"type": "app_mention", "text": "<@U1> status"})
    assert slack_should_answer({"type": "message", "channel_type": "im", "text": "status"})
    assert not slack_should_answer({"type": "message", "channel_type": "channel", "text": "status"})
    assert not slack_should_answer({"type": "app_mention", "bot_id": "B1", "text": "status"})
    assert teams_should_answer({"text": "hi", "conversation": {"conversationType": "personal"}, "from": {}})
    assert teams_should_answer({"text": "hi", "entities": [{"type": "mention"}], "from": {}})
    assert teams_should_answer({"text": "Knowledge Agent, what shipped?", "from": {}})
    assert not teams_should_answer({"text": "unrelated", "from": {"role": "bot"}})
    assert meeting_should_answer("knowledge agent what did we decide")
    assert not meeting_should_answer(DISCLOSURE_TEXT)
    assert not meeting_should_answer("we decided to ship")
    body = {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "knowledge agent why"}]}]}
    assert adf_to_text(body) == "knowledge agent why"
    key, text, author = jira_event_text({"issue": {"key": "PAY-4"}, "comment": {"author": {"accountId": "acct"}, "body": body}})
    assert (key, text, author) == ("PAY-4", "knowledge agent why", "acct")
    assert jira_should_answer(text, "acct", "bot")
    assert not jira_should_answer(text, "bot", "bot")
    assert discord_question({"type": 2, "data": {"options": [{"name": "question", "value": "where?"}]}}) == "where?"
    assert discord_question({"type": 1}) is None


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Response:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class _Completions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return _Response(self._responses.pop(0))


class _Client:
    def __init__(self, responses):
        self.chat = type("Chat", (), {"completions": _Completions(responses)})()


def test_agent_turn_reads_a_file_then_answers(tmp_path):
    (tmp_path / "notes.txt").write_text("hello from the repo\n", encoding="utf-8")
    client = _Client(
        [
            _Message(tool_calls=[_ToolCall("c1", "read_repository_file", json.dumps({"path": "notes.txt"}))]),
            _Message(content="notes.txt says hello from the repo"),
        ]
    )
    reply = asyncio.run(
        run_agent_turn(
            "what is in notes.txt?",
            TurnContext(channel="discord", workspace_id="ws", workspace_root=tmp_path),
            client=client,
        )
    )
    assert reply == "notes.txt says hello from the repo"
    assert client.chat.completions.kwargs[0]["max_tokens"] == 800
    tool_messages = [item for item in client.chat.completions.kwargs[1]["messages"] if item["role"] == "tool"]
    assert "hello from the repo" in tool_messages[0]["content"]


def _recorded(monkeypatch, module):
    seen = []

    async def _send(text, turn, sender):
        seen.append((text, turn.channel, turn.workspace_id))

    monkeypatch.setattr(module, "answer_and_send", _send)
    return seen


def test_slack_mention_replies_and_a_channel_message_does_not(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "slack-secret")
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    pipeline = _Pipeline()
    slack_listener._pipeline = pipeline
    slack_listener._limiter = SlidingWindowRateLimiter(max_requests=20)
    seen = _recorded(monkeypatch, slack_listener)
    client = TestClient(slack_listener.app)

    mention = {
        "team_id": "T1",
        "event": {"type": "app_mention", "text": "<@U9> knowledge agent status", "channel": "C1", "user": "U1", "ts": "1.2"},
    }
    raw = json.dumps(mention).encode("utf-8")
    response = client.post("/slack/events", content=raw, headers=_slack_headers(raw, "slack-secret"))
    assert response.status_code == 200
    assert seen == [("knowledge agent status", "slack", "T1")]
    assert pipeline.calls[0]["text"].startswith("<@U9>")

    chatter = {"team_id": "T1", "event": {"type": "message", "text": "shipping today", "channel": "C1", "user": "U1", "channel_type": "channel"}}
    raw = json.dumps(chatter).encode("utf-8")
    response = client.post("/slack/events", content=raw, headers=_slack_headers(raw, "slack-secret"))
    assert response.status_code == 200
    assert len(seen) == 1
    assert len(pipeline.calls) == 2


def test_teams_personal_chat_replies(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("TEAMS_WEBHOOK_SECRET", "teams-secret")
    pipeline = _Pipeline()
    teams_listener._pipeline = pipeline
    teams_listener._limiter = SlidingWindowRateLimiter(max_requests=20)
    seen = _recorded(monkeypatch, teams_listener)
    body = {
        "type": "message",
        "text": "what shipped?",
        "serviceUrl": "https://smba.example/amer",
        "from": {"id": "u1", "name": "Ada"},
        "recipient": {"id": "bot"},
        "conversation": {"id": "conv-1", "conversationType": "personal"},
        "channelData": {"tenant": {"id": "tenant-1"}},
    }
    raw = json.dumps(body).encode("utf-8")
    headers = _slack_headers(raw, "teams-secret")
    client = TestClient(teams_listener.app)
    response = client.post(
        "/teams/messages",
        content=raw,
        headers={"x-teams-timestamp": headers["x-slack-request-timestamp"], "x-teams-signature": headers["x-slack-signature"]},
    )
    assert response.status_code == 200
    assert seen == [("what shipped?", "teams", "tenant-1")]
    assert pipeline.calls[0]["text"] == "what shipped?"


def test_meeting_wake_phrase_replies_in_chat(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    secret = "whsec_aG9va2RlY2stZ3VpZGUtdGVzdC1rZXktMzJieXRlcyE="
    monkeypatch.setenv("RECALL_WEBHOOK_SECRET", secret)
    pipeline = _Pipeline()
    meeting_listener._pipeline = pipeline
    meeting_listener._consent = MeetingConsentTracker(jurisdiction=Jurisdiction.TWO_PARTY)
    meeting_listener._consent.mark_announced("bot-9", "ws")
    meeting_listener._limiter = SlidingWindowRateLimiter(max_requests=20)
    seen = _recorded(monkeypatch, meeting_listener)
    body = {
        "event": "transcript.data",
        "data": {
            "bot": {"id": "bot-9", "metadata": {"workspace_id": "ws", "meeting_title": "Standup", "platform": "google_meet"}},
            "transcript": {"text": "knowledge agent what did we decide", "speaker": "Ada"},
        },
    }
    raw = json.dumps(body).encode("utf-8")
    client = TestClient(meeting_listener.app)
    response = client.post("/meetings/webhook", content=raw, headers=_recall_headers(raw, secret))
    assert response.status_code == 200
    assert seen == [("knowledge agent what did we decide", "google_meet", "ws")]
    assert pipeline.calls[0]["text"] == "knowledge agent what did we decide"


def _discord_headers(body: bytes, signing_key) -> dict[str, str]:
    timestamp = "1761040800"
    signature = signing_key.sign(timestamp.encode("utf-8") + body).signature.hex()
    return {"x-signature-ed25519": signature, "x-signature-timestamp": timestamp}


def test_discord_ping_and_ask(monkeypatch):
    from nacl.signing import SigningKey

    monkeypatch.setenv("APP_ENV", "development")
    signing_key = SigningKey.generate()
    monkeypatch.setenv("DISCORD_PUBLIC_KEY", signing_key.verify_key.encode().hex())
    monkeypatch.setenv("AGENT_WORKSPACE_ID", "ws_discord")
    gateway._limiter = SlidingWindowRateLimiter(max_requests=20)
    seen = _recorded(monkeypatch, gateway)
    client = TestClient(gateway.app)

    ping = b'{"type":1}'
    bad = client.post("/discord/interactions", content=ping, headers={"x-signature-ed25519": "00", "x-signature-timestamp": "1"})
    assert bad.status_code == 401
    ok = client.post("/discord/interactions", content=ping, headers=_discord_headers(ping, signing_key))
    assert ok.status_code == 200
    assert ok.json() == {"type": 1}

    ask = json.dumps(
        {
            "type": 2,
            "application_id": "app",
            "token": "interaction-token",
            "member": {"user": {"id": "user-1"}},
            "data": {"options": [{"name": "question", "value": "where is the retry?"}]},
        }
    ).encode("utf-8")
    response = client.post("/discord/interactions", content=ask, headers=_discord_headers(ask, signing_key))
    assert response.status_code == 200
    assert response.json()["type"] == 5
    assert seen == [("where is the retry?", "discord", "ws_discord")]


def test_whatsapp_verification_and_inbound_text(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-me")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "wa-secret")
    monkeypatch.setenv("AGENT_WORKSPACE_ID", "ws_wa")
    gateway._limiter = SlidingWindowRateLimiter(max_requests=20)
    seen = _recorded(monkeypatch, gateway)
    client = TestClient(gateway.app)

    challenge = client.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "abc"})
    assert challenge.status_code == 200
    assert challenge.text == "abc"
    mismatch = client.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "abc"})
    assert mismatch.status_code == 403

    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "metadata": {"phone_number_id": "999"},
                            "messages": [{"from": "15551212", "type": "text", "text": {"body": "status of the deploy?"}}],
                        }
                    }
                ]
            }
        ]
    }
    raw = json.dumps(payload).encode("utf-8")
    digest = hmac.new(b"wa-secret", raw, hashlib.sha256).hexdigest()
    rejected = client.post("/whatsapp/webhook", content=raw, headers={"x-hub-signature-256": "sha256=00"})
    assert rejected.status_code == 401
    accepted = client.post("/whatsapp/webhook", content=raw, headers={"x-hub-signature-256": f"sha256={digest}"})
    assert accepted.status_code == 200
    assert seen == [("status of the deploy?", "whatsapp", "ws_wa")]


def test_jira_comment_answers_unless_it_is_the_bot(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("JIRA_WEBHOOK_SECRET", "jira-secret")
    monkeypatch.setenv("JIRA_BOT_ACCOUNT_ID", "bot-account")
    monkeypatch.setenv("AGENT_WORKSPACE_ID", "ws_jira")
    gateway._limiter = SlidingWindowRateLimiter(max_requests=20)
    seen = _recorded(monkeypatch, gateway)
    client = TestClient(gateway.app)

    def _post(author):
        body = {
            "issue": {"key": "PAY-12"},
            "comment": {
                "author": {"accountId": author},
                "body": {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "knowledge agent why did this fail?"}]}]},
            },
        }
        raw = json.dumps(body).encode("utf-8")
        digest = hmac.new(b"jira-secret", raw, hashlib.sha256).hexdigest()
        return client.post("/jira/webhook", content=raw, headers={"x-hub-signature": f"sha256={digest}"})

    human = _post("human")
    assert human.status_code == 200
    assert seen == [("knowledge agent why did this fail?", "jira", "ws_jira")]
    bot = _post("bot-account")
    assert bot.status_code == 200
    assert len(seen) == 1
