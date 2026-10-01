"""
Slack Events API webhook listener (FastAPI). Handles `message.channels` and
`app_mention` events, verifies Slack's request signature, and hands off
text to IngestionPipeline.process_message via a background task so the
webhook responds within Slack's 3-second timeout.

Run: uvicorn ingestion.slack_listener:app --port 8001
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from ingestion.pipeline import IngestionPipeline
from schemas.vector_payload_schema import SourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Slack Ingestion Listener")
pipeline = IngestionPipeline()

SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")


def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    if not SLACK_SIGNING_SECRET:
        logger.warning("SLACK_SIGNING_SECRET not set — refusing to skip verification in any non-dev environment")
        return False
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False  # replay protection
    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    computed = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), basestring.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, signature)


@app.post("/slack/events")
async def slack_events(
    request: Request,
    background_tasks: BackgroundTasks,
    x_slack_signature: str = Header(default=""),
    x_slack_request_timestamp: str = Header(default=""),
):
    body = await request.body()

    if not verify_slack_signature(body, x_slack_request_timestamp, x_slack_signature):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    payload = await request.json()

    # Slack URL verification handshake
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})
    event_type = event.get("type")

    if event_type in ("message", "app_mention"):
        # Ignore bot messages / message edits / channel-join notices to avoid feedback loops
        if event.get("subtype") in ("bot_message", "message_changed", "channel_join"):
            return {"ok": True}

        text = event.get("text", "")
        workspace_id = payload.get("team_id", "")
        channel_id = event.get("channel", "")
        author_id = event.get("user", "")
        thread_id = event.get("thread_ts")

        background_tasks.add_task(
            _ingest_slack_message,
            text=text,
            workspace_id=workspace_id,
            channel_id=channel_id,
            author_id=author_id,
            thread_id=thread_id,
        )

    return {"ok": True}


async def _ingest_slack_message(
    text: str, workspace_id: str, channel_id: str, author_id: str, thread_id: str | None
) -> None:
    try:
        # In production: resolve channel_name and author_display_name via Slack Web API
        # (conversations.info / users.info), cached to avoid rate-limiting on every message.
        await pipeline.process_message(
            text=text,
            workspace_id=workspace_id,
            channel_id=channel_id,
            channel_name=channel_id,  # placeholder — resolve via Slack API in production
            author_id=author_id,
            author_display_name=author_id,  # placeholder — resolve via Slack API in production
            source=SourceType.SLACK,
            thread_id=thread_id,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to ingest Slack message from channel %s", channel_id)
