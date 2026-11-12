"""Tool-calling turn for every text channel, meeting chat, and the voice agent.

The model may call the knowledge base and the read-only development tools.
Workspace id is taken from the channel event, not from the model, so a prompt
cannot point retrieval at another tenant.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from agent_runtime.dev_tools import (
    git_diff,
    git_log,
    git_show,
    git_status,
    jira_get_issue,
    jira_search,
    list_repository,
    read_repository_file,
    run_tests,
    search_repository,
    workspace_root,
)
from security.pii_redaction import PIIRedactor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the enterprise Knowledge Agent. People reach you from \
Slack, Microsoft Teams, Zoom, Google Meet, Discord, Jira, and WhatsApp.

Rules:
- Call lookup_knowledge_base when the question is about past team decisions, \
incidents, meetings, or docs. The workspace is already scoped for you.
- For code questions, use the development tools: list_repository, \
search_repository, read_repository_file, git_status, git_diff, git_log, \
git_show, and run_tests. For tracked work, use jira_get_issue or jira_search. \
These tools are read-only.
- Cite the source channel or file when you use a tool result.
- If a tool returns nothing useful, say so. Do not invent an answer.
- Keep the reply short enough to post in chat.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_knowledge_base",
            "description": "Search ingested Slack, Teams, and meeting knowledge.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repository",
            "description": "Search the workspace source tree for a text snippet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_repository_file",
            "description": "Read a file inside the workspace. Refuses secrets and paths outside the repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_lines": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_repository",
            "description": "List files and directories inside the workspace. Skips secrets and hidden folders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the branch and uncommitted files in the workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show the uncommitted diff. Set staged true for the index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "staged": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent commit subjects. Optionally limit to one workspace path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "path": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_show",
            "description": "Show one commit's subject and diffstat. Pass a hash or branch name.",
            "parameters": {
                "type": "object",
                "properties": {"revision": {"type": "string"}},
                "required": ["revision"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest for the workspace, or for one Python file or directory inside it.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jira_get_issue",
            "description": "Read one Jira issue by key, such as PAY-12.",
            "parameters": {
                "type": "object",
                "properties": {"issue_key": {"type": "string"}},
                "required": ["issue_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jira_search",
            "description": "Run a read-only JQL search.",
            "parameters": {
                "type": "object",
                "properties": {"jql": {"type": "string"}},
                "required": ["jql"],
            },
        },
    },
]


@dataclass
class TurnContext:
    channel: str
    workspace_id: str
    user_id: str | None = None
    allowed_channel_ids: list[str] | None = None
    workspace_root: Path | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


_redactor = PIIRedactor()


def _clip(text: str, limit: int = 4000) -> str:
    body = text or ""
    if len(body) <= limit:
        return body
    return body[:limit] + "\n... truncated"


def _redact(text: str) -> str:
    try:
        return _redactor.redact(text or "").redacted_text
    except Exception:  # noqa: BLE001
        return "[tool output withheld]"


async def _default_knowledge_lookup(query: str, ctx: TurnContext) -> str:
    from voice_agent.kb_tool import query_knowledge_base

    return await query_knowledge_base(
        query=query,
        workspace_id=ctx.workspace_id,
        allowed_channel_ids=ctx.allowed_channel_ids,
        user_id=ctx.user_id,
        source_interface=(
            ctx.channel
            if ctx.channel
            in {
                "slack",
                "teams",
                "voice",
                "discord",
                "whatsapp",
                "jira",
                "zoom",
                "google_meet",
                "teams_meeting",
            }
            else "voice"
        ),
    )


async def execute_tool(name: str, arguments: dict, ctx: TurnContext, *, knowledge_lookup=None) -> str:
    root = workspace_root(ctx.workspace_root)
    try:
        if name == "lookup_knowledge_base":
            lookup = knowledge_lookup or _default_knowledge_lookup
            result = lookup(str(arguments.get("query") or ""), ctx)
            if asyncio.iscoroutine(result):
                result = await result
            return _clip(_redact(str(result)))
        if name == "search_repository":
            return _clip(_redact(search_repository(str(arguments.get("query") or ""), root)))
        if name == "read_repository_file":
            return _clip(
                _redact(
                    read_repository_file(
                        str(arguments.get("path") or ""),
                        root,
                        max_lines=int(arguments.get("max_lines") or 200),
                    )
                )
            )
        if name == "list_repository":
            return _clip(
                _redact(
                    list_repository(
                        root,
                        path=str(arguments.get("path") or "."),
                        limit=int(arguments.get("limit") or 80),
                    )
                )
            )
        if name == "git_status":
            return _clip(_redact(git_status(root)))
        if name == "git_diff":
            return _clip(
                _redact(
                    git_diff(
                        root,
                        path=arguments.get("path"),
                        staged=bool(arguments.get("staged")),
                    )
                )
            )
        if name == "git_log":
            return _clip(
                _redact(
                    git_log(
                        root,
                        limit=int(arguments.get("limit") or 10),
                        path=arguments.get("path"),
                    )
                )
            )
        if name == "git_show":
            return _clip(_redact(git_show(root, str(arguments.get("revision") or ""))))
        if name == "run_tests":
            return _clip(_redact(run_tests(root, arguments.get("path"))))
        if name == "jira_get_issue":
            return _clip(_redact(jira_get_issue(str(arguments.get("issue_key") or ""))))
        if name == "jira_search":
            return _clip(_redact(jira_search(str(arguments.get("jql") or ""))))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tool %s failed", name)
        return f"Tool {name} failed: {exc}"
    return f"Unknown tool: {name}"


def _as_calls(message) -> list[ToolCall]:
    raw = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    calls: list[ToolCall] = []
    for item in raw or []:
        if isinstance(item, dict):
            function = item.get("function") or {}
            calls.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or "{}"),
                )
            )
            continue
        function = getattr(item, "function", None)
        calls.append(
            ToolCall(
                id=str(getattr(item, "id", "") or ""),
                name=str(getattr(function, "name", "") or ""),
                arguments=str(getattr(function, "arguments", "") or "{}"),
            )
        )
    return [call for call in calls if call.name]


def _content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _client():
    from ingestion.nvidia_client import get_nvidia_client

    return get_nvidia_client()


async def run_agent_turn(
    user_text: str,
    ctx: TurnContext,
    *,
    client=None,
    knowledge_lookup=None,
    max_rounds: int = 3,
) -> str:
    text = (user_text or "").strip()
    if not text:
        return "I didn't catch a question."
    try:
        chat = client or _client()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent model client is unavailable")
        return f"I can't answer right now: {exc}"

    from ingestion.nvidia_client import DEFAULT_CHAT_MODEL

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text[:8000]},
    ]
    for _ in range(max_rounds):
        try:
            response = await asyncio.to_thread(
                chat.chat.completions.create,
                model=DEFAULT_CHAT_MODEL,
                max_tokens=800,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent completion failed")
            return f"I couldn't complete that: {exc}"
        message = response.choices[0].message
        calls = _as_calls(message)
        if not calls:
            answer = _redact(_content(message).strip())
            return answer or "I don't have an answer for that."
        messages.append(
            {
                "role": "assistant",
                "content": _content(message),
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ],
            }
        )
        for call in calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            result = await execute_tool(call.name, arguments, ctx, knowledge_lookup=knowledge_lookup)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "I couldn't finish that with the tools I have."
