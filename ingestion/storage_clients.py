"""
Thin client wrappers around Qdrant and Neo4j used by both the ingestion
pipeline and the voice agent's retrieval tool. Keeping these in one place
avoids the two runtimes drifting on connection/query logic.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingestion.entity_extractor import ALLOWED_ENTITY_TYPES, ALLOWED_RELATIONS
from schemas.vector_payload_schema import ChunkPayload, QDRANT_COLLECTION_CONFIG

logger = logging.getLogger(__name__)

_ENTITY_LABEL = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
_REL_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")

SCHEMA_PATCHES = (
    "ALTER TABLE meeting_consent_log ADD COLUMN IF NOT EXISTS jurisdiction TEXT",
    "ALTER TABLE meeting_consent_log ADD COLUMN IF NOT EXISTS disclosure_text TEXT",
    "ALTER TABLE meeting_consent_log ADD COLUMN IF NOT EXISTS objection_speaker TEXT",
    "ALTER TABLE meeting_consent_log ADD COLUMN IF NOT EXISTS policy_action TEXT",
    "ALTER TABLE meeting_consent_log ADD COLUMN IF NOT EXISTS objection_at TIMESTAMPTZ",
    "ALTER TABLE meeting_consent_log ADD COLUMN IF NOT EXISTS excluded_speakers TEXT[]",
    """CREATE INDEX IF NOT EXISTS idx_consent_meeting
       ON meeting_consent_log (workspace_id, meeting_id, announced_at DESC)""",
    """
    DO $$
    DECLARE r record;
    BEGIN
      FOR r IN
        SELECT con.conname
          FROM pg_constraint con
          JOIN pg_class rel ON rel.oid = con.conrelid
          JOIN pg_attribute att ON att.attrelid = rel.oid AND att.attnum = ANY (con.conkey)
         WHERE rel.relname = 'retrieval_audit_log'
           AND con.contype = 'c'
           AND att.attname = 'source_interface'
      LOOP
        EXECUTE format('ALTER TABLE retrieval_audit_log DROP CONSTRAINT %I', r.conname);
      END LOOP;
    END $$;
    """,
    """
    ALTER TABLE retrieval_audit_log
      ADD CONSTRAINT retrieval_audit_log_source_interface_check
      CHECK (source_interface IN (
        'slack', 'teams', 'voice', 'discord', 'whatsapp', 'jira', 'zoom', 'google_meet', 'teams_meeting'
      ))
    """,
)


def entity_merge_query(entity_type: str) -> str:
    """MERGE on id AND workspace_id. Id alone collides across tenants."""
    entity_type = validate_entity_type(entity_type)
    return f"""
    MERGE (n:{entity_type} {{id: $entity_id, workspace_id: $workspace_id}})
    ON CREATE SET n.name = $name, n.created_at = datetime(),
        n.channel_ids = CASE WHEN $channel_id IS NULL OR $channel_id = '' THEN [] ELSE [$channel_id] END
    ON MATCH SET n.name = $name,
        n.channel_ids = CASE
            WHEN $channel_id IS NULL OR $channel_id = '' OR $channel_id IN coalesce(n.channel_ids, [])
            THEN coalesce(n.channel_ids, [])
            ELSE coalesce(n.channel_ids, []) + $channel_id
        END
    """


def relation_merge_query(relation: str) -> str:
    relation = validate_relation(relation)
    return f"""
    MATCH (a {{id: $source_id, workspace_id: $workspace_id}}),
          (b {{id: $target_id, workspace_id: $workspace_id}})
    MERGE (a)-[r:{relation}]->(b)
    ON CREATE SET r.created_at = datetime()
    """


def validate_entity_type(entity_type: str) -> str:
    if entity_type not in ALLOWED_ENTITY_TYPES or not _ENTITY_LABEL.fullmatch(entity_type):
        raise ValueError(f"Unsupported entity type: {entity_type}")
    return entity_type


def validate_relation(relation: str) -> str:
    if relation not in ALLOWED_RELATIONS or not _REL_TYPE.fullmatch(relation):
        raise ValueError(f"Unsupported relation type: {relation}")
    return relation


AUDIT_INTERFACES = frozenset({
    "slack",
    "teams",
    "voice",
    "discord",
    "whatsapp",
    "jira",
    "zoom",
    "google_meet",
    "teams_meeting",
})


class VectorStore:
    def __init__(self, url: str | None = None, api_key: str | None = None):
        self.client = QdrantClient(
            url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.environ.get("QDRANT_API_KEY"),
        )
        self.collection_name = QDRANT_COLLECTION_CONFIG["collection_name"]

    def ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=QDRANT_COLLECTION_CONFIG["vectors_config"]["size"],
                distance=qmodels.Distance.COSINE,
            ),
        )
        for idx in QDRANT_COLLECTION_CONFIG["payload_indexes"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=idx["field_name"],
                field_schema=idx["field_schema"],
            )
        logger.info("Created Qdrant collection %s", self.collection_name)

    def upsert_chunk(self, embedding: list[float], payload: ChunkPayload) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(
                    id=payload.chunk_id,
                    vector=embedding,
                    payload=payload.to_qdrant_payload(),
                )
            ],
        )

    def search(
        self,
        query_vector: list[float],
        workspace_id: str,
        allowed_channel_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Workspace scoping is mandatory and applied as a hard filter, not a
        post-filter — unauthorized content is never returned, let alone ranked.

        An empty channel list means the caller has no channel access. It is not
        the same as ``None`` (workspace-wide)."""
        if allowed_channel_ids is not None and len(allowed_channel_ids) == 0:
            return []
        must_conditions = [
            qmodels.FieldCondition(key="workspace_id", match=qmodels.MatchValue(value=workspace_id))
        ]
        if allowed_channel_ids is not None:
            must_conditions.append(
                qmodels.FieldCondition(key="channel_id", match=qmodels.MatchAny(any=allowed_channel_ids))
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qmodels.Filter(must=must_conditions),
            limit=limit,
        )
        return [{"score": r.score, **r.payload} for r in results]

    def delete_chunks_before(self, cutoff: datetime, sources: list[str] | None = None) -> int:
        """Delete points with ``timestamp`` older than ``cutoff``.

        ``sources`` restricts the purge to those payload values (chat vs meeting).
        Qdrant's delete-by-filter does not return a count; this returns 0 after
        a successful delete so callers can still tell success from a raised error.
        """
        must = [
            qmodels.FieldCondition(
                key="timestamp",
                range=qmodels.DatetimeRange(lt=cutoff),
            )
        ]
        if sources:
            must.append(qmodels.FieldCondition(key="source", match=qmodels.MatchAny(any=sources)))
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=must)),
        )
        return 0


class GraphStore:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                user or os.environ.get("NEO4J_USER", "neo4j"),
                password or os.environ.get("NEO4J_PASSWORD", "password"),
            ),
        )

    def close(self) -> None:
        self.driver.close()

    def merge_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        workspace_id: str,
        channel_id: str | None = None,
    ) -> None:
        query = entity_merge_query(entity_type)
        with self.driver.session() as session:
            session.run(
                query,
                entity_id=entity_id,
                name=name,
                workspace_id=workspace_id,
                channel_id=channel_id or "",
            )

    def merge_relation(self, source_id: str, relation: str, target_id: str, workspace_id: str) -> None:
        query = relation_merge_query(relation)
        with self.driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id, workspace_id=workspace_id)

    def traverse_from_entity(self, entity_name_fragment: str, workspace_id: str, limit: int = 10) -> list[dict]:
        query = """
        MATCH (n {workspace_id: $workspace_id})
        WHERE toLower(n.name) CONTAINS toLower($fragment)
        OPTIONAL MATCH (n)-[r]->(m)
        WHERE m.workspace_id = $workspace_id
        RETURN labels(n) AS source_labels, n.name AS source_name,
               type(r) AS relation, labels(m) AS target_labels, m.name AS target_name
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, fragment=entity_name_fragment, workspace_id=workspace_id, limit=limit)
            return [dict(record) for record in result]

    def delete_meetings_before(self, cutoff_iso: str) -> int:
        query = """
        MATCH (m:Meeting)
        WHERE m.created_at < datetime($cutoff)
        WITH m LIMIT 10000
        WITH collect(m) AS meetings
        FOREACH (node IN meetings | DETACH DELETE node)
        RETURN size(meetings) AS deleted
        """
        with self.driver.session() as session:
            record = session.run(query, cutoff=cutoff_iso).single()
            if record is None:
                return 0
            return int(record["deleted"] or 0)


@dataclass
class ChannelACLEntry:
    workspace_id: str
    channel_id: str
    user_id: str


class PostgresStore:
    """Sync Postgres access for ACL lookups, retrieval audit, consent, and purge.

    Connections open per call so this object is safe to construct at import
    time. ``psycopg`` is imported on first use. Pass ``connect`` in tests to
    supply a stand-in with an ``execute`` / ``commit`` connection.
    """

    def __init__(self, dsn: str | None = None, connect=None):
        self.dsn = dsn if dsn is not None else os.environ.get(
            "POSTGRES_DSN", "postgresql://agent:localdevpassword@localhost:5432/knowledge_agent"
        )
        self._connect = connect
        self._schema_ready = False

    def _connection(self):
        if self._connect is not None:
            return self._connect()
        if not self.dsn:
            raise RuntimeError("POSTGRES_DSN is not set")
        import psycopg

        conn = psycopg.connect(self.dsn)
        if not self._schema_ready:
            for statement in SCHEMA_PATCHES:
                conn.execute(statement)
            conn.commit()
            self._schema_ready = True
        return conn

    def ensure_schema(self) -> None:
        with self._connection() as conn:
            for statement in SCHEMA_PATCHES:
                conn.execute(statement)
            conn.commit()

    def get_allowed_channels(self, workspace_id: str, user_id: str) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT channel_id FROM channel_acl WHERE workspace_id = %s AND user_id = %s",
                (workspace_id, user_id),
            ).fetchall()
        return [row[0] for row in rows]

    def write_retrieval_audit(
        self,
        *,
        workspace_id: str,
        user_id: str | None,
        query_text: str,
        channel_scope: list[str],
        result_chunk_ids: list[str],
        source_interface: str,
    ) -> None:
        if source_interface not in AUDIT_INTERFACES:
            raise ValueError(f"Unsupported audit source: {source_interface}")
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_audit_log
                    (workspace_id, user_id, query_text, channel_scope, result_chunk_ids, source_interface)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (workspace_id, user_id, query_text, channel_scope, result_chunk_ids, source_interface),
            )
            conn.commit()

    def record_announcement(
        self,
        *,
        meeting_id: str,
        workspace_id: str,
        jurisdiction: str,
        disclosure_text: str,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO meeting_consent_log
                    (workspace_id, meeting_id, jurisdiction, disclosure_text)
                VALUES (%s, %s, %s, %s)
                """,
                (workspace_id, meeting_id, jurisdiction, disclosure_text),
            )
            conn.commit()

    def record_objection(
        self,
        *,
        meeting_id: str,
        workspace_id: str,
        speaker: str,
        policy_action: str,
        excluded_speakers: list[str] | None = None,
        jurisdiction: str | None = None,
    ) -> None:
        speakers = list(excluded_speakers or [])
        with self._connection() as conn:
            updated = conn.execute(
                """
                UPDATE meeting_consent_log
                   SET objection_raised = true,
                       objection_speaker = %s,
                       policy_action = %s,
                       objection_at = now(),
                       jurisdiction = COALESCE(%s, jurisdiction),
                       excluded_speakers = %s,
                       recording_stopped_at = CASE WHEN %s = 'stop' THEN COALESCE(recording_stopped_at, now())
                                                  ELSE recording_stopped_at END
                 WHERE id = (
                    SELECT id FROM meeting_consent_log
                     WHERE workspace_id = %s AND meeting_id = %s
                     ORDER BY announced_at DESC
                     LIMIT 1
                 )
                """,
                (speaker, policy_action, jurisdiction, speakers, policy_action, workspace_id, meeting_id),
            )
            if getattr(updated, "rowcount", 0) == 0:
                conn.execute(
                    """
                    INSERT INTO meeting_consent_log
                        (workspace_id, meeting_id, objection_raised, objection_speaker,
                         policy_action, objection_at, recording_stopped_at, jurisdiction, excluded_speakers)
                    VALUES (
                        %s, %s, true, %s, %s, now(),
                        CASE WHEN %s = 'stop' THEN now() ELSE NULL END,
                        %s, %s
                    )
                    """,
                    (workspace_id, meeting_id, speaker, policy_action, policy_action, jurisdiction, speakers),
                )
            conn.commit()

    def get_consent(self, meeting_id: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT jurisdiction,
                       recording_stopped_at IS NOT NULL,
                       COALESCE(excluded_speakers, '{}')
                  FROM meeting_consent_log
                 WHERE meeting_id = %s
                 ORDER BY announced_at DESC
                 LIMIT 1
                """,
                (meeting_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "jurisdiction": row[0],
            "stopped": bool(row[1]),
            "excluded_speakers": list(row[2] or []),
        }

    def consent_announced(self, meeting_id: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM meeting_consent_log WHERE meeting_id = %s LIMIT 1",
                (meeting_id,),
            ).fetchone()
        return row is not None

    def recording_stopped(self, meeting_id: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM meeting_consent_log
                 WHERE meeting_id = %s AND recording_stopped_at IS NOT NULL
                 LIMIT 1
                """,
                (meeting_id,),
            ).fetchone()
        return row is not None

    def purge_audit_before(self, cutoff: datetime) -> int:
        with self._connection() as conn:
            audit = conn.execute(
                "DELETE FROM retrieval_audit_log WHERE created_at < %s",
                (cutoff,),
            )
            consent = conn.execute(
                "DELETE FROM meeting_consent_log WHERE announced_at < %s",
                (cutoff,),
            )
            conn.commit()
        return int(audit.rowcount or 0) + int(consent.rowcount or 0)


# Backward-compatible name used by earlier scaffold notes.
ACLStore = PostgresStore
