"""Text-agent webhooks for Discord, WhatsApp, and Jira.

Slack, Teams, and Zoom/Meet replies are attached to their existing listeners.
Run: uvicorn channels.gateway:app --port 8010
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from agent_runtime.addressing import discord_question, jira_event_text, jira_should_answer, whatsapp_messages
from agent_runtime.dispatch import answer_and_send
from agent_runtime.outbound import post_discord_followup, post_jira_comment, post_whatsapp
from agent_runtime.runner import TurnContext
from channels.signatures import verify_discord_signature, verify_sha256_signature
from ingestion.rate_limit import enforce_rate_limit, limiter_from_env
from security.encryption import require_encryption_at_rest

load_dotenv()
logger = logging.getLogger(__name__)
_limiter = limiter_from_env()


def _workspace() -> str:
    return os.environ.get("AGENT_WORKSPACE_ID", "ws_demo")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_encryption_at_rest()
    yield


app = FastAPI(title="Channel Agent Gateway", lifespan=_lifespan)


@app.post("/discord/interactions")
async def discord_interactions(request: Request, background_tasks: BackgroundTasks):
    enforce_rate_limit(request, _limiter)
    body = await request.body()
    public_key = os.environ.get("DISCORD_PUBLIC_KEY", "")
    if not verify_discord_signature(
        body,
        request.headers.get("x-signature-ed25519", ""),
        request.headers.get("x-signature-timestamp", ""),
        public_key,
    ):
        raise HTTPException(status_code=401, detail="Invalid Discord signature")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    if payload.get("type") == 1:
        return {"type": 1}
    question = discord_question(payload)
    if not question:
        return {"type": 4, "data": {"content": "Ask with /ask and a question."}}
    application_id = str(payload.get("application_id") or "")
    token = str(payload.get("token") or "")
    user = (payload.get("member") or {}).get("user") or payload.get("user") or {}
    background_tasks.add_task(
        _answer,
        question,
        TurnContext(channel="discord", workspace_id=_workspace(), user_id=str(user.get("id") or "") or None),
        lambda reply: post_discord_followup(application_id, token, reply),
    )
    return {"type": 5}


@app.get("/whatsapp/webhook")
async def whatsapp_verify(request: Request):
    expected = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="WHATSAPP_VERIFY_TOKEN is not configured")
    mode = request.query_params.get("hub.mode", "")
    token = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")
    if mode == "subscribe" and hmac_compare(token, expected):
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    enforce_rate_limit(request, _limiter)
    body = await request.body()
    secret = os.environ.get("WHATSAPP_APP_SECRET", "")
    if not verify_sha256_signature(body, request.headers.get("x-hub-signature-256", ""), secret):
        raise HTTPException(status_code=401, detail="Invalid WhatsApp signature")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    for message in whatsapp_messages(payload):
        background_tasks.add_task(
            _answer,
            message["text"],
            TurnContext(channel="whatsapp", workspace_id=_workspace(), user_id=message["from"] or None),
            lambda reply, msg=message: post_whatsapp(msg["phone_number_id"], msg["from"], reply),
        )
    return {"ok": True}


@app.post("/jira/webhook")
async def jira_webhook(request: Request, background_tasks: BackgroundTasks):
    enforce_rate_limit(request, _limiter)
    body = await request.body()
    secret = os.environ.get("JIRA_WEBHOOK_SECRET", "")
    if not verify_sha256_signature(body, request.headers.get("x-hub-signature", ""), secret):
        raise HTTPException(status_code=401, detail="Invalid Jira signature")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    issue_key, text, author = jira_event_text(payload)
    bot_account = os.environ.get("JIRA_BOT_ACCOUNT_ID", "")
    if not jira_should_answer(text, author, bot_account) or not issue_key:
        return {"ok": True}
    background_tasks.add_task(
        _answer,
        text,
        TurnContext(channel="jira", workspace_id=_workspace(), user_id=author or None),
        lambda reply, key=issue_key: post_jira_comment(key, reply),
    )
    return {"ok": True}


def hmac_compare(provided: str, expected: str) -> bool:
    import hmac

    if len(provided) != len(expected):
        return False
    return hmac.compare_digest(provided, expected)


async def _answer(text: str, turn: TurnContext, sender) -> None:
    await answer_and_send(text, turn, sender)
