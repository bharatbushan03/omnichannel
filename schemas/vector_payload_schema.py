"""
Qdrant collection schema + payload contract for the knowledge vector store.

Collection: `knowledge_chunks`
Vector: 1024-dim (NVIDIA nv-embedqa-e5-v5, free tier via build.nvidia.com).
If you swap embedding models, update `size` below to match — a mismatch
between this config and the model's actual output dimension causes Qdrant
to reject every upsert.
Distance: Cosine.

This file is the single source of truth for payload shape. Both the ingestion
pipeline and the retrieval layer import `ChunkPayload` from here — do not
duplicate the schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4


class SourceType(str, Enum):
    SLACK = "slack"
    TEAMS = "teams"
    ZOOM = "zoom"
    GOOGLE_MEET = "google_meet"
    TEAMS_MEETING = "teams_meeting"
    MANUAL = "manual"


class ContentClass(str, Enum):
    TECHNICAL_DECISION = "technical_decision"
    DEBUG_LOG = "debug_log"
    SCHEDULE = "schedule"
    DOC_REFERENCE = "doc_reference"
    MEETING_SEGMENT = "meeting_segment"


@dataclass
class ChunkPayload:
    """One row = one embedded chunk. Mirrors the Qdrant point payload exactly."""

    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""                    # REQUIRED — hard filter at query time
    channel_id: str = ""                      # Slack/Teams channel or meeting ID
    channel_name: str = ""
    source: SourceType = SourceType.SLACK
    content_class: ContentClass = ContentClass.TECHNICAL_DECISION

    text: str = ""                            # redacted, chunked text
    text_raw_hash: str = ""                   # sha256 of pre-redaction text, for audit trail only

    author_id: str = ""
    author_display_name: str = ""
    speaker_label: Optional[str] = None       # diarization label, meetings only ("Speaker 2")

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    thread_id: Optional[str] = None           # Slack thread_ts / Teams reply chain id
    meeting_id: Optional[str] = None
    meeting_timestamp_offset_s: Optional[float] = None  # seconds into the call, meetings only

    entities: list[str] = field(default_factory=list)   # entity IDs extracted from this chunk, FK -> Neo4j
    acl_scope: list[str] = field(default_factory=list)   # list of role/group IDs allowed to retrieve this

    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    chunk_index: int = 0                       # position within source document/message
    token_count: int = 0

    def to_qdrant_payload(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        d["content_class"] = self.content_class.value
        return d


QDRANT_COLLECTION_CONFIG = {
    "collection_name": "knowledge_chunks",
    "vectors_config": {
        "size": 1024,
        "distance": "Cosine",
    },
    # Payload indexes — required for fast filtering by workspace/channel/acl at query time.
    "payload_indexes": [
        {"field_name": "workspace_id", "field_schema": "keyword"},
        {"field_name": "channel_id", "field_schema": "keyword"},
        {"field_name": "source", "field_schema": "keyword"},
        {"field_name": "content_class", "field_schema": "keyword"},
        {"field_name": "acl_scope", "field_schema": "keyword"},
        {"field_name": "timestamp", "field_schema": "datetime"},
    ],
}
