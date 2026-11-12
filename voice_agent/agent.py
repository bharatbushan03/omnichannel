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

from dotenv import load_dotenv

from agent_runtime.runner import TurnContext, execute_tool
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
- Call lookup_knowledge_base when the question is about past team decisions, \
incidents, meetings, or docs. Skip it for greetings and small talk.
- For code questions, call the development tools: list_repository, \
search_repository, read_repository_file, git_status, git_diff, git_log, \
git_show, and run_tests. For tracked work, call jira_get_issue or jira_search. \
These tools are read-only.
- Cite the source channel, file, or commit when you use a tool result.
- If a tool returns nothing useful, say so. Do not invent an answer.
- Keep responses concise. This is a live voice conversation, not a document.
"""


def prewarm(proc: JobProcess) -> None:
    """Runs once per worker process before handling jobs — load heavy resources
    here so they're not paid for on every call. VAD model load + KB client init."""
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()
    init_kb_clients()


class KnowledgeAgent(Agent):
    def __init__(self, scope: RoomScope) -> None:
        super().__init__(instructions=AGENT_INSTRUCTIONS)
        self.scope = scope

    def _turn(self) -> TurnContext:
        return TurnContext(
            channel="voice",
            workspace_id=self.scope.workspace_id,
            user_id=self.scope.user_id,
            allowed_channel_ids=self.scope.allowed_channel_ids,
        )

    async def _use(self, name: str, arguments: dict) -> str:
        async def _lookup(query: str, _ctx: TurnContext) -> str:
            return await query_knowledge_base(
                query=query,
                workspace_id=self.scope.workspace_id,
                allowed_channel_ids=self.scope.allowed_channel_ids,
                user_id=self.scope.user_id,
                participant_user_ids=self.scope.participant_user_ids,
                source_interface="voice",
            )

        logger.info("Voice tool called: %s", name)
        return await execute_tool(name, arguments, self._turn(), knowledge_lookup=_lookup)

    @function_tool()
    async def lookup_knowledge_base(self, context: RunContext, query: str) -> str:
        """Search ingested Slack, Teams, and meeting knowledge.

        Args:
            query: A focused natural-language search query.
        """
        return await self._use("lookup_knowledge_base", {"query": query})

    @function_tool()
    async def search_repository(self, context: RunContext, query: str) -> str:
        """Search the workspace source tree for a text snippet.

        Args:
            query: The text to find.
        """
        return await self._use("search_repository", {"query": query})

    @function_tool()
    async def read_repository_file(self, context: RunContext, path: str, max_lines: int = 200) -> str:
        """Read a file inside the workspace. Refuses secrets and paths outside the repo.

        Args:
            path: Workspace-relative file path.
            max_lines: Maximum lines to return.
        """
        return await self._use("read_repository_file", {"path": path, "max_lines": max_lines})

    @function_tool()
    async def list_repository(self, context: RunContext, path: str = ".", limit: int = 80) -> str:
        """List files and directories inside the workspace. Skips secrets and hidden folders.

        Args:
            path: Directory relative to the workspace. Use "." for the root.
            limit: Maximum entries to return.
        """
        return await self._use("list_repository", {"path": path, "limit": limit})

    @function_tool()
    async def git_status(self, context: RunContext) -> str:
        """Show the branch and uncommitted files in the workspace."""
        return await self._use("git_status", {})

    @function_tool()
    async def git_diff(self, context: RunContext, path: str = "", staged: bool = False) -> str:
        """Show the uncommitted diff.

        Args:
            path: Optional file to limit the diff to.
            staged: When true, show the index instead of the working tree.
        """
        return await self._use("git_diff", {"path": path, "staged": staged})

    @function_tool()
    async def git_log(self, context: RunContext, limit: int = 10, path: str = "") -> str:
        """Show recent commit subjects.

        Args:
            limit: How many commits to show.
            path: Optional file to limit the history to.
        """
        return await self._use("git_log", {"limit": limit, "path": path})

    @function_tool()
    async def git_show(self, context: RunContext, revision: str) -> str:
        """Show one commit's subject and diffstat.

        Args:
            revision: A commit hash or branch name.
        """
        return await self._use("git_show", {"revision": revision})

    @function_tool()
    async def run_tests(self, context: RunContext, path: str = "") -> str:
        """Run pytest for the workspace, or for one Python file or directory inside it.

        Args:
            path: Optional test file or directory. Empty runs the default suite.
        """
        return await self._use("run_tests", {"path": path})

    @function_tool()
    async def jira_get_issue(self, context: RunContext, issue_key: str) -> str:
        """Read one Jira issue by key.

        Args:
            issue_key: An issue key such as PAY-12.
        """
        return await self._use("jira_get_issue", {"issue_key": issue_key})

    @function_tool()
    async def jira_search(self, context: RunContext, jql: str) -> str:
        """Run a read-only JQL search.

        Args:
            jql: The JQL query.
        """
        return await self._use("jira_search", {"jql": jql})


async def entrypoint(ctx: JobContext) -> None:
    from livekit.plugins import cartesia, deepgram, openai as lk_openai

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
