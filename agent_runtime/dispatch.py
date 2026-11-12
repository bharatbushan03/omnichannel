"""Run one agent turn and hand the reply to a channel sender."""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from agent_runtime.runner import TurnContext, run_agent_turn

logger = logging.getLogger(__name__)


async def answer_and_send(
    text: str,
    turn: TurnContext,
    sender: Callable[[str], Awaitable[None]],
) -> None:
    try:
        reply = await run_agent_turn(text, turn)
        await sender(reply)
    except Exception:  # noqa: BLE001
        logger.exception("Agent reply failed on %s", turn.channel)
