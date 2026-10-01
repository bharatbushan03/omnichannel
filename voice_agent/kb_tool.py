"""
The tool the voice agent's LLM calls to query the hybrid knowledge base
mid-conversation. Kept as a standalone function (not a method) so it maps
cleanly onto livekit-agents' function-tool decorator pattern.

This is the tool-gating point that keeps latency down: it is only invoked
when the LLM's own judgment says the turn needs a KB lookup, not on every
turn.
"""
from __future__ import annotations

import logging

from ingestion.embedder import Embedder
from ingestion.hybrid_retrieval import HybridRetriever
from ingestion.storage_clients import GraphStore, VectorStore

logger = logging.getLogger(__name__)

# Module-level singletons — initialized once per agent process, reused across
# calls/turns to avoid reconnecting to Qdrant/Neo4j on every tool invocation.
_vector_store: VectorStore | None = None
_graph_store: GraphStore | None = None
_embedder: Embedder | None = None
_retriever: HybridRetriever | None = None


def init_kb_clients() -> None:
    global _vector_store, _graph_store, _embedder, _retriever
    _vector_store = VectorStore()
    _graph_store = GraphStore()
    _embedder = Embedder()
    _retriever = HybridRetriever(
        vector_store=_vector_store,
        graph_store=_graph_store,
        embedder=_embedder,
    )


async def query_knowledge_base(
    query: str,
    workspace_id: str,
    allowed_channel_ids: list[str] | None = None,
) -> str:
    """Query the hybrid knowledge base (vector + graph) and return an
    assembled context string ready to inject into the LLM's next turn.

    Args:
        query: The natural-language question to search for.
        workspace_id: Tenant scope — never omit this, it is the access boundary.
        allowed_channel_ids: If the calling user's channel membership is known,
            pass it to further restrict retrieval. None means workspace-wide
            (only appropriate for workspace-level admin/broadcast contexts).

    Returns:
        A formatted context string (chunks + graph facts) or a
        "no results" message the LLM can relay to the user.
    """
    if _retriever is None:
        init_kb_clients()

    try:
        context = _retriever.retrieve(  # type: ignore[union-attr]
            query=query,
            workspace_id=workspace_id,
            allowed_channel_ids=allowed_channel_ids,
            limit=6,
        )
    except Exception:  # noqa: BLE001 — a failed lookup must not crash the live voice turn
        logger.exception("Knowledge base query failed for: %s", query)
        return "Knowledge base lookup failed. Let the user know you couldn't retrieve that information right now."

    if not context.chunks and not context.graph.facts:
        return "No relevant information found in the knowledge base for this query."

    return context.to_prompt_string(max_chunks=5)
