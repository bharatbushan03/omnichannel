"""
Passive meeting ingestion via Recall.ai bots (Zoom/Google Meet).

The bot joins, posts the disclosure in chat, and streams Deepgram Nova-2
transcripts to ``/meetings/webhook``. Segments go through the same
IngestionPipeline as Slack, after signature verification, rate limiting, and
the jurisdiction consent check.

Run: uvicorn ingestion.meeting_listener:app --port 8002
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from agent_runtime.addressing import meeting_should_answer
from agent_runtime.dispatch import answer_and_send
from agent_runtime.outbound import post_recall_chat
from agent_runtime.runner import TurnContext
from ingestion.pipeline import IngestionPipeline
from ingestion.rate_limit import enforce_rate_limit, limiter_from_env
from schemas.vector_payload_schema import SourceType
from security.consent import DISCLOSURE_TEXT, MeetingConsentTracker, parse_jurisdiction
from security.encryption import require_encryption_at_rest
from security.webhook_signature import verify_recall_signature

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECALL_API_BASE = os.environ.get("RECALL_API_BASE", "https://us-east-1.recall.ai/api/v1")
_limiter = limiter_from_env()
_pipeline: IngestionPipeline | None = None
_consent = MeetingConsentTracker()


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def _secret() -> str:
    return os.environ.get("RECALL_WEBHOOK_SECRET", "")


def _api_key() -> str:
    return os.environ.get("RECALL_API_KEY", "")


def _admin_token() -> str:
    return os.environ.get("INGESTION_ADMIN_TOKEN", "")


def source_for_meeting_url(meeting_url: str) -> SourceType:
    host = urlparse(meeting_url).netloc.lower()
    if "meet.google" in host or host.endswith("meet.google.com"):
        return SourceType.GOOGLE_MEET
    if "teams.microsoft" in host or "teams.live" in host:
        return SourceType.TEAMS_MEETING
    return SourceType.ZOOM


def build_create_bot_payload(
    meeting_url: str,
    workspace_id: str,
    meeting_title: str,
    *,
    jurisdiction: str,
    webhook_url: str | None = None,
    join_at: str | None = None,
) -> dict:
    """Current Create Bot shape: recording_config + chat disclosure.

    ``transcription_options`` was removed in Recall API 1.11. Deepgram Nova-2
    with machine diarization is set under ``deepgram_streaming``. Video is
    disabled; this pipeline stores transcripts, not recordings.
    """
    payload: dict = {
        "meeting_url": meeting_url,
        "join_at": join_at or datetime.now(timezone.utc).isoformat(),
        "bot_name": "Knowledge Agent",
        "chat": {
            "on_bot_join": {
                "send_to": "everyone",
                "message": DISCLOSURE_TEXT,
                "pin": True,
            },
            "on_participant_join": {
                "send_to": "everyone",
                "message": DISCLOSURE_TEXT,
            },
        },
        "recording_config": {
            "video_mixed_mp4": None,
            "transcript": {
                "diarization": {"use_separate_streams_when_available": False},
                "provider": {
                    "deepgram_streaming": {"model": "nova-2", "diarize": True},
                },
            },
        },
        "metadata": {
            "workspace_id": workspace_id,
            "meeting_title": meeting_title,
            "disclosure": DISCLOSURE_TEXT,
            "jurisdiction": jurisdiction,
            "platform": source_for_meeting_url(meeting_url).value,
        },
    }
    if webhook_url:
        payload["recording_config"]["realtime_endpoints"] = [
            {"type": "webhook", "url": webhook_url, "events": ["transcript.data"]}
        ]
    return payload


async def start_meeting_bot(
    meeting_url: str,
    workspace_id: str,
    meeting_title: str,
    *,
    jurisdiction: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Dispatch a Recall.ai bot and record that the disclosure was configured."""
    if not _api_key():
        raise RuntimeError("RECALL_API_KEY not configured")

    chosen = parse_jurisdiction(jurisdiction)
    body = build_create_bot_payload(
        meeting_url,
        workspace_id,
        meeting_title,
        jurisdiction=chosen.value,
        webhook_url=os.environ.get("RECALL_REALTIME_WEBHOOK_URL") or None,
    )
    owns_client = client is None
    http = client or httpx.AsyncClient()
    try:
        response = await http.post(
            f"{RECALL_API_BASE}/bot/",
            headers={"Authorization": f"Token {_api_key()}", "Content-Type": "application/json"},
            json=body,
            timeout=30.0,
        )
        response.raise_for_status()
        created = response.json()
    finally:
        if owns_client:
            await http.aclose()

    bot_id = str(created.get("id") or "")
    _consent.mark_announced(bot_id, workspace_id, jurisdiction=chosen.value)
    return created


async def leave_meeting_bot(bot_id: str, client: httpx.AsyncClient | None = None) -> None:
    """Remove the bot from the call. Recall treats this as irreversible."""
    if not bot_id or not _api_key():
        logger.error("Cannot leave meeting %s: bot id or RECALL_API_KEY missing", bot_id)
        return
    owns_client = client is None
    http = client or httpx.AsyncClient()
    try:
        response = await http.post(
            f"{RECALL_API_BASE}/bot/{bot_id}/leave_call/",
            headers={"Authorization": f"Token {_api_key()}"},
            timeout=30.0,
        )
        response.raise_for_status()
    finally:
        if owns_client:
            await http.aclose()


def _words_to_text(words: list) -> str:
    parts: list[str] = []
    for word in words:
        if isinstance(word, str):
            parts.append(word)
        elif isinstance(word, dict):
            parts.append(str(word.get("text") or ""))
    return " ".join(part for part in parts if part).strip()


def _relative_start(value) -> float:
    if isinstance(value, dict):
        try:
            return float(value.get("relative") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def parse_transcript_segments(payload: dict) -> list[dict]:
    """Normalize current ``transcript.data`` events and the older flat shape.

    Current Recall realtime events nest words and the participant under
    ``data.data``. Older payloads put ``text`` / ``speaker`` on ``data.transcript``.
    """
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        return []
    bot = data.get("bot") or {}
    metadata = bot.get("metadata") or {}
    meeting_id = str(bot.get("id") or data.get("bot_id") or "")
    workspace_id = str(metadata.get("workspace_id") or "")
    meeting_title = str(metadata.get("meeting_title") or "")
    platform = str(metadata.get("platform") or SourceType.ZOOM.value)

    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    words = inner.get("words") if isinstance(inner.get("words"), list) else []
    participant = inner.get("participant") if isinstance(inner.get("participant"), dict) else {}

    if words:
        speaker = str(participant.get("name") or participant.get("id") or "Unknown Speaker")
        offset = _relative_start(words[0].get("start_timestamp") if isinstance(words[0], dict) else 0)
        text = _words_to_text(words)
        if not text:
            return []
        return [
            {
                "text": text,
                "speaker": speaker,
                "offset_s": offset,
                "meeting_id": meeting_id,
                "workspace_id": workspace_id,
                "meeting_title": meeting_title,
                "platform": platform,
            }
        ]

    transcript = data.get("transcript") if isinstance(data.get("transcript"), dict) else {}
    text = str(transcript.get("text") or "")
    if not text.strip():
        return []
    return [
        {
            "text": text.strip(),
            "speaker": str(transcript.get("speaker") or "Unknown Speaker"),
            "offset_s": _relative_start(transcript.get("start_timestamp")),
            "meeting_id": meeting_id,
            "workspace_id": workspace_id,
            "meeting_title": meeting_title,
            "platform": platform,
        }
    ]


def parse_bot_status(payload: dict) -> dict:
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    bot = data.get("bot") if isinstance(data.get("bot"), dict) else {}
    metadata = bot.get("metadata") or data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    event = str(payload.get("event") or "")
    if event == "bot.status_change":
        status = str((data.get("status") or {}).get("code") or "")
    elif event.startswith("bot."):
        status = event.split(".", 1)[1]
    else:
        status = ""
    return {
        "meeting_id": str(bot.get("id") or data.get("bot_id") or ""),
        "workspace_id": str(metadata.get("workspace_id") or ""),
        "status": status,
        "disclosure": str(metadata.get("disclosure") or ""),
        "jurisdiction": str(metadata.get("jurisdiction") or ""),
    }


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_encryption_at_rest()
    yield


app = FastAPI(title="Meeting Ingestion Listener", lifespan=_lifespan)


@app.post("/meetings/start")
async def start_meeting(request: Request):
    """Dispatch a bot. Requires ``INGESTION_ADMIN_TOKEN`` so this route cannot
    spend Recall credits anonymously."""
    enforce_rate_limit(request, _limiter)
    expected = _admin_token()
    provided = request.headers.get("x-admin-token", "")
    if not expected:
        raise HTTPException(status_code=503, detail="INGESTION_ADMIN_TOKEN is not configured")
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token")

    body = await request.json()
    meeting_url = str(body.get("meeting_url") or "")
    workspace_id = str(body.get("workspace_id") or "")
    meeting_title = str(body.get("meeting_title") or "")
    if not meeting_url or not workspace_id or not meeting_title:
        raise HTTPException(status_code=400, detail="meeting_url, workspace_id, and meeting_title are required")
    try:
        created = await start_meeting_bot(
            meeting_url,
            workspace_id,
            meeting_title,
            jurisdiction=body.get("jurisdiction"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to start meeting bot")
        raise HTTPException(status_code=502, detail="Failed to dispatch meeting bot") from exc
    return {"ok": True, "bot_id": created.get("id")}


@app.post("/meetings/webhook")
async def recall_webhook(request: Request, background_tasks: BackgroundTasks):
    enforce_rate_limit(request, _limiter)
    body = await request.body()
    if not verify_recall_signature(body, request.headers, _secret()):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    event_type = payload.get("event")
    if event_type in ("bot.status_change", "bot.in_call_recording", "bot.joining_call"):
        status = parse_bot_status(payload)
        if status["disclosure"] == DISCLOSURE_TEXT and status["meeting_id"]:
            _consent.mark_announced(
                status["meeting_id"],
                status["workspace_id"],
                jurisdiction=status["jurisdiction"] or None,
            )
        return {"ok": True}

    if event_type == "transcript.data":
        for segment in parse_transcript_segments(payload):
            decision = _consent.evaluate(
                segment["meeting_id"],
                segment["workspace_id"],
                segment["speaker"],
                segment["text"],
            )
            if decision.stop_recording:
                logger.info("Objection in meeting %s; removing the bot", segment["meeting_id"])
                background_tasks.add_task(leave_meeting_bot, segment["meeting_id"])
                continue
            if not decision.allow_ingest:
                logger.info(
                    "Skipping meeting segment (%s) in %s",
                    decision.reason,
                    segment["meeting_id"],
                )
                continue
            background_tasks.add_task(_ingest_meeting_segment, segment)
            if meeting_should_answer(segment["text"]):
                background_tasks.add_task(_answer_in_meeting, segment)

    return {"ok": True}


async def _ingest_meeting_segment(segment: dict) -> None:
    platform = segment.get("platform")
    try:
        source = SourceType(platform)
    except ValueError:
        source = SourceType.ZOOM
    try:
        await get_pipeline().process_meeting_segment(
            text=segment["text"],
            workspace_id=segment["workspace_id"],
            meeting_id=segment["meeting_id"],
            meeting_title=segment["meeting_title"],
            speaker_label=segment["speaker"],
            timestamp_offset_s=float(segment["offset_s"]),
            source=source,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to ingest meeting segment from meeting %s", segment.get("meeting_id"))


async def _answer_in_meeting(segment: dict) -> None:
    platform = str(segment.get("platform") or "zoom")
    channel = platform if platform in ("zoom", "google_meet", "teams_meeting") else "zoom"
    await answer_and_send(
        segment["text"],
        TurnContext(
            channel=channel,
            workspace_id=segment.get("workspace_id") or "ws_demo",
            user_id=segment.get("speaker") or None,
        ),
        lambda reply: post_recall_chat(segment.get("meeting_id", ""), reply),
    )
