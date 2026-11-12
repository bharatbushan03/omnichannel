"""
LiveKit Agents entrypoint for the real-time voice Knowledge Agent.

Pipeline: Deepgram Nova-2 (STT) -> NVIDIA Nemotron via OpenAI-compatible
endpoint (LLM, tool-calling) -> Cartesia (TTS)
Transport: LiveKit WebRTC, with VAD-based turn detection and automatic
barge-in/interruption handling provided by the AgentSession runtime.

NOTE ON LATENCY: Nemotron via build.nvidia.com's free tier is a shared,
rate-limited endpoint. First-token latency will be noticeably higher and
less consistent than a dedicated low-latency provider (Groq, Cerebras, or
paid Anthropic/OpenAI). Expect this to be the dominant latency cost in the
voice pipeline, likely exceeding the 800ms budget discussed in
docs/ARCHITECTURE.md even on the no-retrieval fast path. Acceptable for a
free-tier demo; revisit before any real usage.

Run:
    export LIVEKIT_URL=wss://your-project.livekit.cloud
    export LIVEKIT_API_KEY=...
    export LIVEKIT_API_SECRET=...
    export DEEPGRAM_API_KEY=...
    export CARTESIA_API_KEY=...
    export NVIDIA_API_KEY=...      # free tier: https://build.nvidia.com
    python -m voice_agent.agent dev        # local dev mode, connects to a test room
    python -m voice_agent.agent start      # production worker mode

This targets livekit-agents >= 0.9 API shape (AgentSession / function_tool).
Verify against the installed package version — this is a fast-moving SDK and
method names have changed across releases; pin your version in requirements.txt.
"""
from __future__ import annotations

import logging
import os

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import cartesia, deepgram, openai as lk_openai, silero

from dotenv import load_dotenv

from ingestion.nvidia_client import DEFAULT_CHAT_MODEL, NVIDIA_BASE_URL
from security.consent import DISCLOSURE_TEXT
from security.encryption import require_encryption_at_rest
from voice_agent.kb_tool import init_kb_clients, query_knowledge_base
from voice_agent.scope import RoomScope, parse_room_metadata

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-knowledge-agent")

AGENT_INSTRUCTIONS = """You are an enterprise Knowledge Agent joining a live \
voice call to help the team with technical/operational questions.

Rules:
- Call `lookup_knowledge_base` ONLY when the user asks something that plausibly
  has an answer in past team decisions, debug logs, schedules, or docs.
  Do not call it for greetings, small talk, or questions you can answer directly.
- When you do use retrieved context, cite the source channel/meeting and
  timestamp briefly (e.g. "per #eng-payments last Tuesday...").
- If the knowledge base has no relevant results, say so plainly — do not
  fabricate an answer.
- Keep responses concise. This is a live voice conversation, not a document.
"""


def prewarm(proc: JobProcess) -> None:
    """Runs once per worker process before handling jobs — load heavy resources
    here so they're not paid for on every call. VAD model load + KB client init."""
    proc.userdata["vad"] = silero.VAD.load()
    init_kb_clients()


class KnowledgeAgent(Agent):
    def __init__(self, scope: RoomScope) -> None:
        super().__init__(instructions=AGENT_INSTRUCTIONS)
        self.scope = scope

    @function_tool()
    async def lookup_knowledge_base(self, context: RunContext, query: str) -> str:
        """Search the company's internal knowledge base (Slack decisions, debug
        logs, meeting transcripts, docs) for information relevant to `query`.

        Args:
            query: A focused natural-language search query, e.g. "how did we
                fix the payments-service 500 errors last week".
        """
        logger.info("KB tool called: %s", query)
        return await query_knowledge_base(
            query=query,
            workspace_id=self.scope.workspace_id,
            allowed_channel_ids=self.scope.allowed_channel_ids,
            user_id=self.scope.user_id,
            participant_user_ids=self.scope.participant_user_ids,
            source_interface="voice",
        )


async def entrypoint(ctx: JobContext) -> None:
    require_encryption_at_rest()
    await ctx.connect()

    # workspace_id, user ids, and channel scope come from room metadata set when
    # the backend dispatched this agent. A plain string is still a workspace id.
    scope = parse_room_metadata(ctx.room.metadata)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2", language="en"),
        llm=lk_openai.LLM(
            model=DEFAULT_CHAT_MODEL,
            base_url=NVIDIA_BASE_URL,
            # api_key is read from NVIDIA_API_KEY automatically only if you pass
            # it explicitly — the OpenAI plugin defaults to OPENAI_API_KEY, so
            # NVIDIA's key must be forwarded here explicitly.
            api_key=os.environ.get("NVIDIA_API_KEY"),
        ),
        tts=cartesia.TTS(),
        # Turn detection: VAD-based endpointing with a short silence threshold.
        # livekit-agents also supports a semantic turn-detector plugin for
        # higher accuracy on natural pauses — see their docs for
        # `turn_detector.EOUModel` if VAD-only endpointing clips users too eagerly.
    )

    agent = KnowledgeAgent(scope=scope)

    await session.start(agent=agent, room=ctx.room)

    # Spoken disclosure on join. The sentence is fixed so the consent log and
    # the voice greeting match. See security/governance.md.
    await session.generate_reply(
        instructions=(
            "Say this disclosure verbatim, then one short sentence that you can "
            "answer questions about past decisions, incidents, and docs during "
            f"this call: {DISCLOSURE_TEXT}"
        )
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
