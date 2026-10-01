"""
Thin client wrappers around Qdrant and Neo4j used by both the ingestion
pipeline and the voice agent's retrieval tool. Keeping these in one place
avoids the two runtimes drifting on connection/query logic.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from schemas.vector_payload_schema import ChunkPayload, QDRANT_COLLECTION_CONFIG

logger = logging.getLogger(__name__)


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
        post-filter — unauthorized content is never returned, let alone ranked."""
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

    def merge_entity(self, entity_id: str, entity_type: str, name: str, workspace_id: str) -> None:
        # entity_type is constrained upstream to ALLOWED_ENTITY_TYPES — safe to interpolate as a label.
        query = f"""
        MERGE (n:{entity_type} {{id: $entity_id}})
        ON CREATE SET n.name = $name, n.workspace_id = $workspace_id, n.created_at = datetime()
        ON MATCH SET n.name = $name
        """
        with self.driver.session() as session:
            session.run(query, entity_id=entity_id, name=name, workspace_id=workspace_id)

    def merge_relation(self, source_id: str, relation: str, target_id: str) -> None:
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        MERGE (a)-[r:{relation}]->(b)
        ON CREATE SET r.created_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id)

    def traverse_from_entity(self, entity_name_fragment: str, workspace_id: str, limit: int = 10) -> list[dict]:
        query = """
        MATCH (n {workspace_id: $workspace_id})
        WHERE toLower(n.name) CONTAINS toLower($fragment)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN labels(n) AS source_labels, n.name AS source_name,
               type(r) AS relation, labels(m) AS target_labels, m.name AS target_name
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, fragment=entity_name_fragment, workspace_id=workspace_id, limit=limit)
            return [dict(record) for record in result]


@dataclass
class ChannelACLEntry:
    workspace_id: str
    channel_id: str
    user_id: str


class ACLStore:
    """Postgres-backed channel access control. Stub connection logic — wire to
    your actual Postgres pool (asyncpg/SQLAlchemy) in production. Interface
    shown so the retrieval layer has a stable contract to call against."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("POSTGRES_DSN", "postgresql://localhost/knowledge_agent")
        # In production: self.pool = await asyncpg.create_pool(self.dsn)

    def get_allowed_channels(self, workspace_id: str, user_id: str) -> list[str]:
        """Return channel_ids the user_id is a member of in workspace_id.
        Placeholder implementation — replace with real query:
            SELECT channel_id FROM channel_acl WHERE workspace_id = $1 AND user_id = $2
        """
        raise NotImplementedError("Wire this to your Postgres channel_acl table.")
