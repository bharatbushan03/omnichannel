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
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self._bm25_corpus = bm25_corpus or {}
        self._bm25_index: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        if self._bm25_corpus:
            self._build_bm25()

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
    ) -> AssembledContext:
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

        return AssembledContext(chunks=chunks, graph=GraphContext(facts=graph_facts))
