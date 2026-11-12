"""Room metadata contract for the voice agent.

The dispatching backend sets LiveKit room metadata. A JSON object carries the
tenant and the people on the call. A plain string remains a workspace id so
older rooms keep working.

Most restrictive shared scope: when several participant ids are present and no
channel list was precomputed, retrieval intersects their channel memberships.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class RoomScope:
    workspace_id: str = "ws_demo"
    user_id: str | None = None
    participant_user_ids: list[str] = field(default_factory=list)
    allowed_channel_ids: list[str] | None = None


def parse_room_metadata(metadata: str | dict | None) -> RoomScope:
    if isinstance(metadata, dict):
        data = metadata
    else:
        if metadata is None or not str(metadata).strip():
            return RoomScope()
        text = str(metadata).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return RoomScope(workspace_id=text)
        if not isinstance(data, dict):
            return RoomScope(workspace_id=text)

    workspace_id = str(data.get("workspace_id") or "ws_demo")
    user_id = data.get("user_id")
    user_id = str(user_id) if user_id else None

    participants: list[str] = []
    raw_participants = data.get("participant_user_ids") or []
    if isinstance(raw_participants, list):
        participants = [str(item) for item in raw_participants if item]

    channels = data.get("allowed_channel_ids")
    allowed: list[str] | None
    if isinstance(channels, list):
        allowed = [str(item) for item in channels]
    else:
        allowed = None

    return RoomScope(
        workspace_id=workspace_id,
        user_id=user_id,
        participant_user_ids=participants,
        allowed_channel_ids=allowed,
    )
