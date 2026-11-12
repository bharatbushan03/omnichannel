"""Post an agent reply back to the channel that asked. Each sender no-ops when its secret is missing."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MAX = 3500


def clip_reply(text: str) -> str:
    body = (text or "").strip()
    if len(body) <= _MAX:
        return body
    return body[: _MAX - 1] + "…"


async def post_slack(channel: str, text: str, thread_ts: str | None = None, *, client=None) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if client is None and not token:
        logger.error("SLACK_BOT_TOKEN is not set; Slack reply was not sent")
        return
    web = client
    if web is None:
        from slack_sdk import WebClient

        web = WebClient(token=token)
    web.chat_postMessage(channel=channel, text=clip_reply(text), thread_ts=thread_ts)


async def post_teams(service_url: str, conversation_id: str, text: str, *, client=None, token: str | None = None) -> None:
    if not service_url or not conversation_id:
        logger.error("Teams reply is missing serviceUrl or conversation id")
        return
    import httpx

    bearer = token if token is not None else os.environ.get("TEAMS_BOT_TOKEN", "")
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    url = service_url.rstrip("/") + f"/v3/conversations/{conversation_id}/activities"
    http = client or httpx.AsyncClient(timeout=20.0)
    owns = client is None
    try:
        response = await http.post(url, json={"type": "message", "text": clip_reply(text)}, headers=headers)
        response.raise_for_status()
    finally:
        if owns:
            await http.aclose()


async def post_discord_followup(application_id: str, interaction_token: str, text: str, *, client=None) -> None:
    if not application_id or not interaction_token:
        logger.error("Discord follow-up is missing the application id or interaction token")
        return
    import httpx

    url = f"https://discord.com/api/v10/webhooks/{application_id}/{interaction_token}"
    http = client or httpx.AsyncClient(timeout=20.0)
    owns = client is None
    try:
        response = await http.post(url, json={"content": clip_reply(text)})
        response.raise_for_status()
    finally:
        if owns:
            await http.aclose()


async def post_whatsapp(phone_number_id: str, to: str, text: str, *, client=None) -> None:
    token = os.environ.get("WHATSAPP_TOKEN", "")
    if not phone_number_id or not to or not token:
        logger.error("WhatsApp reply is missing the phone number id, recipient, or token")
        return
    import httpx

    url = f"https://graph.facebook.com/v20.0/{phone_number_id}/messages"
    http = client or httpx.AsyncClient(timeout=20.0)
    owns = client is None
    try:
        response = await http.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": clip_reply(text)}},
        )
        response.raise_for_status()
    finally:
        if owns:
            await http.aclose()


async def post_jira_comment(issue_key: str, text: str, *, client=None) -> None:
    base = os.environ.get("JIRA_BASE_URL", "").strip().rstrip("/")
    email = os.environ.get("JIRA_EMAIL", "").strip()
    token = os.environ.get("JIRA_API_TOKEN", "").strip()
    if not (base and email and token and issue_key):
        logger.error("Jira comment was not sent; Jira credentials or issue key are missing")
        return
    import httpx

    http = client or httpx.AsyncClient(timeout=20.0)
    owns = client is None
    try:
        response = await http.post(
            f"{base}/rest/api/3/issue/{issue_key}/comment",
            auth=(email, token),
            json={
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": clip_reply(text)}]}
                    ],
                }
            },
        )
        response.raise_for_status()
    finally:
        if owns:
            await http.aclose()


async def post_recall_chat(bot_id: str, text: str, *, client=None) -> None:
    key = os.environ.get("RECALL_API_KEY", "")
    if not bot_id or not key:
        logger.error("Meeting chat reply was not sent; bot id or RECALL_API_KEY is missing")
        return
    import httpx

    base = os.environ.get("RECALL_API_BASE", "https://us-east-1.recall.ai/api/v1").rstrip("/")
    http = client or httpx.AsyncClient(timeout=20.0)
    owns = client is None
    try:
        response = await http.post(
            f"{base}/bot/{bot_id}/send_chat_message/",
            headers={"Authorization": f"Token {key}"},
            json={"message": clip_reply(text), "to": "everyone"},
        )
        response.raise_for_status()
    finally:
        if owns:
            await http.aclose()
