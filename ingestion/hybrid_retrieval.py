"""
Hybrid retrieval: RRF fusion of dense (Qdrant) + sparse (BM25) results,
with Neo4j graph traversal as a SEPARATE context-augmentation step.

This is the corrected design vs. the original spec's "RRF combining vector +
graph" — RRF only fuses ranked lists of the same comparable unit (text
chunks). Graph output is structured entity/relation data and is appended
to the assembled context after fusion, not blended into the ranking.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from ingestion.embedder import Embedder
from ingestion.storage_clients import VectorStore, GraphStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    channel_name: str
    timestamp: str
    rrf_score: float


@dataclass
class GraphContext:
    facts: list[str] = field(default_factory=list)


@dataclass
class AssembledContext:
    chunks: list[RetrievedChunk]
    graph: GraphContext

    def to_prompt_string(self, max_chunks: int = 6) -> str:
        parts = ["## Retrieved knowledge chunks"]
        for c in self.chunks[:max_chunks]:
            parts.append(f"- [{c.source}/{c.channel_name} @ {c.timestamp}] {c.text}")
        if self.graph.facts:
            parts.append("\n## Related entities and relationships")
            parts.extend(f"- {fact}" for fact in self.graph.facts)
        return "\n".join(parts)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],  # each inner list = chunk_ids in rank order, best first
    k: int = 60,
) -> list[str]:
    """Standard RRF: score(d) = sum over lists of 1 / (k + rank(d)). Returns
    chunk_ids ordered by fused score, descending."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True), scores


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedder: Embedder,
        bm25_corpus: dict[str, str] | None = None,  # chunk_id -> text, in-memory BM25 index
        audit_log=None,
        acl_store=None,
        audit_fail_closed: bool | None = None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.audit_log = audit_log
        self.acl_store = acl_store
        if audit_fail_closed is None:
            audit_fail_closed = os.environ.get("AUDIT_FAIL_CLOSED", "false").lower() in ("1", "true", "yes")
        self.audit_fail_closed = audit_fail_closed
        self._bm25_corpus = bm25_corpus or {}
        self._bm25_index: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        if self._bm25_corpus:
            self._build_bm25()

    def _shared_channels(self, workspace_id: str, user_ids: list[str]) -> list[str]:
        """Intersect channel membership. Any lookup failure returns no channels."""
        if self.acl_store is None:
            logger.error("ACL store is not configured; refusing channel-scoped retrieval")
            return []
        shared: set[str] | None = None
        for user_id in user_ids:
            try:
                allowed = set(self.acl_store.get_allowed_channels(workspace_id, user_id))
            except Exception:
                logger.exception("ACL lookup failed for %s; fail closed", user_id)
                return []
            shared = allowed if shared is None else shared & allowed
        return sorted(shared or [])

    def _build_bm25(self) -> None:
        self._bm25_ids = list(self._bm25_corpus.keys())
        tokenized = [self._bm25_corpus[cid].lower().split() for cid in self._bm25_ids]
        self._bm25_index = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, limit: int) -> list[str]:
        if self._bm25_index is None:
            return []
        scores = self._bm25_index.get_scores(query.lower().split())
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, score in ranked[:limit] if score > 0]

    def retrieve(
        self,
        query: str,
        workspace_id: str,
        allowed_channel_ids: list[str] | None = None,
        graph_entity_hint: str | None = None,
        limit: int = 10,
        user_id: str | None = None,
        participant_user_ids: list[str] | None = None,
        source_interface: str = "voice",
    ) -> AssembledContext:
        user_ids = list(participant_user_ids or [])
        if not user_ids and user_id:
            user_ids = [user_id]
        if allowed_channel_ids is None and user_ids:
            allowed_channel_ids = self._shared_channels(workspace_id, user_ids)

        # An empty channel list is "no access", not "search everything". Skip
        # both the vector store and the graph so a failed ACL lookup cannot
        # still return entity facts from the workspace.
        if allowed_channel_ids is not None and len(allowed_channel_ids) == 0:
            assembled = AssembledContext(chunks=[], graph=GraphContext())
            self._write_audit(
                workspace_id=workspace_id,
                user_id=user_id,
                query_text=query,
                channel_scope=[],
                result_chunk_ids=[],
                source_interface=source_interface,
            )
            return assembled

        # 1. Dense vector search — "query" mode, must match the "passage" mode
        # used when this content was embedded for storage, or retrieval accuracy drops.
        query_vector = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(
            query_vector=query_vector,
            workspace_id=workspace_id,
            allowed_channel_ids=allowed_channel_ids,
            limit=limit,
        )
        dense_ranked_ids = [h["chunk_id"] for h in dense_hits]
        dense_by_id = {h["chunk_id"]: h for h in dense_hits}

        # 2. Sparse BM25 search (in-memory index — swap for Elasticsearch/OpenSearch at scale)
        sparse_ranked_ids = self._bm25_search(query, limit=limit)

        # 3. RRF fusion over the two ranked lists
        fused_ids, _scores = reciprocal_rank_fusion([dense_ranked_ids, sparse_ranked_ids])

        chunks = []
        for cid in fused_ids[:limit]:
            payload = dense_by_id.get(cid)
            if payload is None:
                continue  # BM25-only hit without vector metadata — skip, or fetch payload separately
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    channel_name=payload.get("channel_name", ""),
                    timestamp=payload.get("timestamp", ""),
                    rrf_score=_scores.get(cid, 0.0),
                )
            )

        # 4. Graph traversal — separate augmentation, not fused into the ranking
        graph_facts = []
        entity_hint = graph_entity_hint or query
        try:
            graph_rows = self.graph_store.traverse_from_entity(entity_hint, workspace_id=workspace_id)
            for row in graph_rows:
                if row.get("relation"):
                    graph_facts.append(
                        f"{row['source_name']} -[{row['relation']}]-> {row.get('target_name', '?')}"
                    )
        except Exception as e:  # noqa: BLE001 — retrieval must degrade gracefully, never crash the voice turn
            logger.warning("Graph traversal failed, continuing with vector-only context: %s", e)

        assembled = AssembledContext(chunks=chunks, graph=GraphContext(facts=graph_facts))
        self._write_audit(
            workspace_id=workspace_id,
            user_id=user_id,
            query_text=query,
            channel_scope=list(allowed_channel_ids or []),
            result_chunk_ids=[chunk.chunk_id for chunk in assembled.chunks],
            source_interface=source_interface,
        )
        return assembled

    def _write_audit(
        self,
        *,
        workspace_id: str,
        user_id: str | None,
        query_text: str,
        channel_scope: list[str],
        result_chunk_ids: list[str],
        source_interface: str,
    ) -> None:
        if self.audit_log is None:
            if self.audit_fail_closed:
                raise RuntimeError("Retrieval audit log is required but not configured")
            return
        try:
            self.audit_log.write_retrieval_audit(
                workspace_id=workspace_id,
                user_id=user_id,
                query_text=query_text,
                channel_scope=channel_scope,
                result_chunk_ids=result_chunk_ids,
                source_interface=source_interface,
            )
        except Exception:
            if self.audit_fail_closed:
                logger.exception("Audit log write failed; withholding results")
                raise
            logger.exception("Audit log write failed")
