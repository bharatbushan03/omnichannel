"""
Slack Events API webhook listener (FastAPI). Handles `message.channels` and
`app_mention` events, verifies Slack's request signature, rate-limits callers,
and hands text to IngestionPipeline.process_message via a background task so
the webhook responds within Slack's 3-second timeout.

Run: uvicorn ingestion.slack_listener:app --port 8001
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from agent_runtime.addressing import clean_slack_text, slack_should_answer
from agent_runtime.dispatch import answer_and_send
from agent_runtime.outbound import post_slack
from agent_runtime.runner import TurnContext
from ingestion.pipeline import IngestionPipeline
from ingestion.rate_limit import enforce_rate_limit, limiter_from_env
from schemas.vector_payload_schema import SourceType
from security.encryption import require_encryption_at_rest
from security.webhook_signature import verify_slack_style_signature

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_limiter = limiter_from_env()
_pipeline: IngestionPipeline | None = None
_channel_names: dict[str, str] = {}
_user_names: dict[str, str] = {}


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def _signing_secret() -> str:
    return os.environ.get("SLACK_SIGNING_SECRET", "")


def verify_slack_signature(body: bytes, timestamp: str, signature: str, *, now: float | None = None) -> bool:
    if not _signing_secret():
        logger.warning("SLACK_SIGNING_SECRET not set — refusing the request")
        return False
    return verify_slack_style_signature(body, timestamp, signature, _signing_secret(), now=now)


def resolve_channel_name(channel_id: str, client=None) -> str:
    """Best-effort ``#name`` lookup. Falls back to the channel id when the
    Slack Web API is unconfigured or the call fails, so ingestion still proceeds."""
    if not channel_id:
        return channel_id
    if channel_id in _channel_names:
        return _channel_names[channel_id]
    web = client if client is not None else _slack_web_client()
    if web is None:
        return channel_id
    try:
        info = web.conversations_info(channel=channel_id)
        name = (info.get("channel") or {}).get("name") or channel_id
        resolved = name if str(name).startswith("#") else f"#{name}"
    except Exception:
        logger.warning("Slack conversations.info failed for %s", channel_id, exc_info=True)
        return channel_id
    _channel_names[channel_id] = resolved
    return resolved


def resolve_user_name(user_id: str, client=None) -> str:
    if not user_id:
        return user_id
    if user_id in _user_names:
        return _user_names[user_id]
    web = client if client is not None else _slack_web_client()
    if web is None:
        return user_id
    try:
        info = web.users_info(user=user_id)
        profile = (info.get("user") or {}).get("profile") or {}
        name = profile.get("display_name") or profile.get("real_name") or user_id
    except Exception:
        logger.warning("Slack users.info failed for %s", user_id, exc_info=True)
        return user_id
    _user_names[user_id] = name or user_id
    return _user_names[user_id]


def _slack_web_client():
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        return None
    from slack_sdk import WebClient

    return WebClient(token=token)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_encryption_at_rest()
    yield


app = FastAPI(title="Slack Ingestion Listener", lifespan=_lifespan)


@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    enforce_rate_limit(request, _limiter)
    body = await request.body()
    if not verify_slack_signature(
        body,
        request.headers.get("x-slack-request-timestamp", ""),
        request.headers.get("x-slack-signature", ""),
    ):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})
    event_type = event.get("type")

    if event_type in ("message", "app_mention"):
        # bot_id covers app posts that omit subtype and would otherwise be re-ingested.
        if event.get("bot_id") or event.get("subtype") in ("bot_message", "message_changed", "channel_join"):
            return {"ok": True}

        background_tasks.add_task(
            _ingest_slack_message,
            text=event.get("text", ""),
            workspace_id=payload.get("team_id", ""),
            channel_id=event.get("channel", ""),
            author_id=event.get("user", ""),
            thread_id=event.get("thread_ts"),
        )
        if slack_should_answer(event):
            background_tasks.add_task(
                _answer_slack,
                text=event.get("text", ""),
                workspace_id=payload.get("team_id", ""),
                channel_id=event.get("channel", ""),
                author_id=event.get("user", ""),
                thread_ts=event.get("thread_ts") or event.get("ts"),
            )

    return {"ok": True}


async def _ingest_slack_message(
    text: str, workspace_id: str, channel_id: str, author_id: str, thread_id: str | None
) -> None:
    try:
        await get_pipeline().process_message(
            text=text,
            workspace_id=workspace_id,
            channel_id=channel_id,
            channel_name=resolve_channel_name(channel_id),
            author_id=author_id,
            author_display_name=resolve_user_name(author_id),
            source=SourceType.SLACK,
            thread_id=thread_id,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to ingest Slack message from channel %s", channel_id)


async def _answer_slack(
    text: str, workspace_id: str, channel_id: str, author_id: str, thread_ts: str | None
) -> None:
    await answer_and_send(
        clean_slack_text(text),
        TurnContext(channel="slack", workspace_id=workspace_id or "ws_demo", user_id=author_id or None),
        lambda reply: post_slack(channel_id, reply, thread_ts),
    )
