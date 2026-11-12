"""When a channel message is a question for the agent, versus passive capture.

Passive ingestion still stores channel chatter. A reply is sent only when the
message is addressed to the agent: a Slack app mention or DM, a Teams personal
chat or @mention, a meeting utterance that names the agent, a Jira comment that
names the agent, a Discord /ask command, or any inbound WhatsApp text.
"""
from __future__ import annotations

import re

from security.consent import DISCLOSURE_TEXT

_WAKE = re.compile(r"\bknowledge agent\b", re.IGNORECASE)
_SLACK_MENTION = re.compile(r"<@[A-Z0-9]+>")


def slack_should_answer(event: dict) -> bool:
    if event.get("bot_id") or event.get("subtype") in ("bot_message", "message_changed", "channel_join"):
        return False
    if event.get("type") == "app_mention":
        return True
    return event.get("channel_type") == "im"


def clean_slack_text(text: str) -> str:
    return _SLACK_MENTION.sub("", text or "").strip()


def teams_should_answer(payload: dict) -> bool:
    sender = payload.get("from") or {}
    if sender.get("role") == "bot":
        return False
    conversation = payload.get("conversation") or {}
    if conversation.get("conversationType") == "personal":
        return True
    entities = payload.get("entities") or []
    if any(isinstance(item, dict) and item.get("type") == "mention" for item in entities):
        return True
    return bool(_WAKE.search(str(payload.get("text") or "")))


def meeting_should_answer(text: str) -> bool:
    body = text or ""
    if DISCLOSURE_TEXT.lower() in body.lower():
        return False
    return bool(_WAKE.search(body))


def jira_should_answer(text: str, author_account_id: str = "", bot_account_id: str = "") -> bool:
    if bot_account_id and author_account_id == bot_account_id:
        return False
    return bool(_WAKE.search(text or ""))


def adf_to_text(node) -> str:
    """Flatten a Jira Atlassian Document Format body into plain text."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(part for part in (adf_to_text(item) for item in node) if part)
    if isinstance(node, dict):
        if node.get("type") == "text":
            return str(node.get("text") or "")
        return adf_to_text(node.get("content"))
    return ""


def jira_event_text(payload: dict) -> tuple[str, str, str]:
    """Return issue key, plain comment text, and the comment author's account id."""
    issue = payload.get("issue") or {}
    key = str(issue.get("key") or "")
    comment = payload.get("comment") or {}
    author = str((comment.get("author") or {}).get("accountId") or "")
    body = comment.get("body")
    if body is None:
        fields = issue.get("fields") or {}
        body = fields.get("description") or fields.get("summary") or ""
    return key, adf_to_text(body).strip(), author


def discord_question(payload: dict) -> str | None:
    """Slash command `/ask` carries the question in an option named question, query, or text."""
    if payload.get("type") != 2:
        return None
    data = payload.get("data") or {}
    for option in data.get("options") or []:
        if isinstance(option, dict) and option.get("name") in ("question", "query", "text"):
            value = str(option.get("value") or "").strip()
            return value or None
    return None


def whatsapp_messages(payload: dict) -> list[dict]:
    """User text messages. Status callbacks have no `messages` array and produce nothing."""
    found: list[dict] = []
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            if not isinstance(change, dict):
                continue
            value = change.get("value") or {}
            phone_id = str((value.get("metadata") or {}).get("phone_number_id") or "")
            for message in value.get("messages") or []:
                if not isinstance(message, dict) or message.get("type") != "text":
                    continue
                text = str((message.get("text") or {}).get("body") or "").strip()
                if not text:
                    continue
                found.append(
                    {
                        "from": str(message.get("from") or ""),
                        "text": text,
                        "phone_number_id": phone_id,
                        "id": str(message.get("id") or ""),
                    }
                )
    return found
