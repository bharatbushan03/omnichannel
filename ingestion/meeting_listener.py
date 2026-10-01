"""
Passive meeting ingestion via Recall.ai bots (Zoom/Google Meet). Recall.ai
joins the call as a bot, records/transcribes, and posts webhook events for
transcript segments. This module: (1) starts a bot for a scheduled meeting,
(2) receives real-time transcript webhooks, (3) feeds segments into the
same IngestionPipeline used for Slack.

NOTE: Recall.ai's real-time transcript provider is configurable — this uses
Deepgram Nova-2 as specified, with speaker diarization enabled. Verify exact
webhook payload shape against current Recall.ai API docs before going live;
field names shown here follow their documented transcript.data event shape
as of the pipeline's design but Recall.ai's API does evolve.

Run: uvicorn ingestion.meeting_listener:app --port 8002
"""
from __future__ import annotations

import logging
import os

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from ingestion.pipeline import IngestionPipeline
from schemas.vector_payload_schema import SourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meeting Ingestion Listener")
pipeline = IngestionPipeline()

RECALL_API_KEY = os.environ.get("RECALL_API_KEY", "")
RECALL_API_BASE = "https://us-east-1.recall.ai/api/v1"
RECALL_WEBHOOK_SECRET = os.environ.get("RECALL_WEBHOOK_SECRET", "")


async def start_meeting_bot(meeting_url: str, workspace_id: str, meeting_title: str) -> dict:
    """Dispatch a Recall.ai bot to join a Zoom/Meet call and stream transcription
    through Deepgram Nova-2 with diarization enabled."""
    if not RECALL_API_KEY:
        raise RuntimeError("RECALL_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RECALL_API_BASE}/bot",
            headers={"Authorization": f"Token {RECALL_API_KEY}"},
            json={
                "meeting_url": meeting_url,
                "bot_name": "Knowledge Agent",
                "transcription_options": {
                    "provider": "deepgram",
                    "deepgram": {"model": "nova-2", "diarize": True},
                },
                "metadata": {"workspace_id": workspace_id, "meeting_title": meeting_title},
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


def _verify_recall_webhook(request_headers: dict) -> bool:
    """Recall.ai signs webhooks — verify per their current signing scheme
    (svix-style headers) before trusting payload contents. Placeholder check;
    wire to the actual verification library/method Recall.ai documents."""
    if not RECALL_WEBHOOK_SECRET:
        logger.warning("RECALL_WEBHOOK_SECRET not set — rejecting webhook in any non-dev environment")
        return False
    # TODO: implement real signature verification against Recall.ai's current scheme.
    return True


@app.post("/meetings/webhook")
async def recall_webhook(request: Request, background_tasks: BackgroundTasks):
    if not _verify_recall_webhook(dict(request.headers)):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    event_type = payload.get("event")

    if event_type == "transcript.data":
        data = payload.get("data", {})
        metadata = data.get("bot", {}).get("metadata", {})
        workspace_id = metadata.get("workspace_id", "")
        meeting_title = metadata.get("meeting_title", "")
        meeting_id = data.get("bot", {}).get("id", "")

        for word_group in data.get("transcript", {}).get("words", []):
            pass  # word-level payload varies — see segment handling below

        segment_text = data.get("transcript", {}).get("text", "")
        speaker = data.get("transcript", {}).get("speaker", "Unknown Speaker")
        offset_s = data.get("transcript", {}).get("start_timestamp", {}).get("relative", 0.0)

        if segment_text.strip():
            background_tasks.add_task(
                _ingest_meeting_segment,
                text=segment_text,
                workspace_id=workspace_id,
                meeting_id=meeting_id,
                meeting_title=meeting_title,
                speaker_label=speaker,
                offset_s=offset_s,
            )

    return {"ok": True}


async def _ingest_meeting_segment(
    text: str, workspace_id: str, meeting_id: str, meeting_title: str, speaker_label: str, offset_s: float
) -> None:
    try:
        await pipeline.process_meeting_segment(
            text=text,
            workspace_id=workspace_id,
            meeting_id=meeting_id,
            meeting_title=meeting_title,
            speaker_label=speaker_label,
            timestamp_offset_s=offset_s,
            source=SourceType.ZOOM,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to ingest meeting segment from meeting %s", meeting_id)
