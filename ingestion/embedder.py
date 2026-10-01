"""
Chunking + embedding generation. Chunks are sized for retrieval quality
(not just token limits) — short, semantically coherent units beat huge
chunks for RAG precision.

Embeddings use NVIDIA's free-tier nv-embedqa-e5-v5 model (OpenAI-compatible
endpoint). This model is ASYMMETRIC: it requires an `input_type` of either
"passage" (when indexing/storing text) or "query" (when embedding a search
query). Using the wrong one silently degrades retrieval quality — it will
not error, it will just return worse search results. Always use
`embed_passages()` in the ingestion pipeline and `embed_query()` in the
retrieval layer; never call `embed_batch`/`embed_one` directly unless you
are deliberately choosing the mode yourself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import openai

from ingestion.nvidia_client import DEFAULT_EMBED_MODEL, get_nvidia_client

logger = logging.getLogger(__name__)

InputType = Literal["passage", "query"]


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    token_estimate: int


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 100) -> list[TextChunk]:
    """Simple sliding-window chunker. For production, prefer a sentence-boundary-aware
    splitter (e.g. langchain's RecursiveCharacterTextSplitter) — this keeps the
    dependency footprint minimal for the scaffold."""
    if len(text) <= max_chars:
        return [TextChunk(text=text, chunk_index=0, token_estimate=len(text) // 4)]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to break on a sentence/paragraph boundary near the window edge
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + max_chars // 2:
            end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(TextChunk(text=chunk, chunk_index=idx, token_estimate=len(chunk) // 4))
            idx += 1
        start = end - overlap_chars if end < len(text) else end
    return chunks


class Embedder:
    def __init__(self, client: openai.OpenAI | None = None, model: str = DEFAULT_EMBED_MODEL):
        self.client = client or get_nvidia_client()
        self.model = model

    def embed_batch(self, texts: list[str], input_type: InputType = "passage") -> list[list[float]]:
        """Low-level call. Prefer embed_passages()/embed_query() below unless
        you have a specific reason to pick the mode manually."""
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        return [item.embedding for item in response.data]

    def embed_one(self, text: str, input_type: InputType = "passage") -> list[float]:
        return self.embed_batch([text], input_type=input_type)[0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Use when embedding content to be STORED (ingestion pipeline)."""
        return self.embed_batch(texts, input_type="passage")

    def embed_query(self, text: str) -> list[float]:
        """Use when embedding a SEARCH QUERY (retrieval layer)."""
        return self.embed_one(text, input_type="query")


if __name__ == "__main__":
    sample = "This is a long technical decision log entry. " * 40
    for c in chunk_text(sample):
        print(c.chunk_index, len(c.text), c.token_estimate)
