"""
Microsoft Teams ingestion webhook.

Passive capture for a Bot Framework ``message`` activity. This process sits
behind a gateway that signs the raw body the same way Slack does
(``X-Teams-Timestamp`` + ``X-Teams-Signature`` = ``v0=`` HMAC-SHA256 of
``v0:{timestamp}:{body}``, key ``TEAMS_WEBHOOK_SECRET``). Requests without a
valid signature are rejected. Direct Bot Connector JWT validation belongs at
the gateway that terminates the Azure Bot channel; this service only stores
text that has already been authenticated.

Run: uvicorn ingestion.teams_listener:app --port 8003
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from agent_runtime.addressing import teams_should_answer
from agent_runtime.dispatch import answer_and_send
from agent_runtime.outbound import post_teams
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


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def _secret() -> str:
    return os.environ.get("TEAMS_WEBHOOK_SECRET", "")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_encryption_at_rest()
    yield


app = FastAPI(title="Teams Ingestion Listener", lifespan=_lifespan)


def parse_teams_activity(payload: dict) -> dict | None:
    if payload.get("type") != "message":
        return None
    sender = payload.get("from") or {}
    recipient = payload.get("recipient") or {}
    if sender.get("role") == "bot" or (sender.get("id") and sender.get("id") == recipient.get("id")):
        return None
    text = str(payload.get("text") or "").strip()
    if not text:
        return None
    channel_data = payload.get("channelData") or {}
    if not isinstance(channel_data, dict):
        channel_data = {}
    team = channel_data.get("team") or {}
    channel = channel_data.get("channel") or {}
    tenant = channel_data.get("tenant") or {}
    conversation = payload.get("conversation") or {}
    workspace_id = str(team.get("id") or tenant.get("id") or "")
    channel_id = str(channel.get("id") or conversation.get("id") or "")
    author = sender or {}
    return {
        "text": text,
        "workspace_id": workspace_id,
        "channel_id": channel_id,
        "channel_name": str(channel.get("name") or channel_id),
        "author_id": str(author.get("id") or ""),
        "author_display_name": str(author.get("name") or author.get("id") or ""),
        "thread_id": str(conversation.get("id") or "") or None,
        "service_url": str(payload.get("serviceUrl") or ""),
        "conversation_id": str(conversation.get("id") or ""),
        "answer": teams_should_answer(payload),
    }


@app.post("/teams/messages")
async def teams_messages(request: Request, background_tasks: BackgroundTasks):
    enforce_rate_limit(request, _limiter)
    body = await request.body()
    if not verify_slack_style_signature(
        body,
        request.headers.get("x-teams-timestamp", ""),
        request.headers.get("x-teams-signature", ""),
        _secret(),
    ):
        raise HTTPException(status_code=401, detail="Invalid Teams signature")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    activity = parse_teams_activity(payload)
    if activity is None:
        return {"ok": True}

    background_tasks.add_task(_ingest_teams_message, activity)
    if activity.get("answer"):
        background_tasks.add_task(_answer_teams, activity)
    return {"ok": True}


async def _ingest_teams_message(activity: dict) -> None:
    try:
        await get_pipeline().process_message(
            text=activity["text"],
            workspace_id=activity["workspace_id"],
            channel_id=activity["channel_id"],
            channel_name=activity["channel_name"],
            author_id=activity["author_id"],
            author_display_name=activity["author_display_name"],
            source=SourceType.TEAMS,
            thread_id=activity.get("thread_id"),
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to ingest Teams message from channel %s", activity.get("channel_id"))


async def _answer_teams(activity: dict) -> None:
    await answer_and_send(
        activity["text"],
        TurnContext(
            channel="teams",
            workspace_id=activity.get("workspace_id") or "ws_demo",
            user_id=activity.get("author_id") or None,
        ),
        lambda reply: post_teams(activity.get("service_url", ""), activity.get("conversation_id", ""), reply),
    )
