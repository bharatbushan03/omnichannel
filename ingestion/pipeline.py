"""
Main ingestion orchestration. Wire this to your Slack/Teams webhook handler
or Recall.ai transcript webhook — each calls `IngestionPipeline.process_message`
or `.process_meeting_segment` per incoming unit of text.

Run standalone for a smoke test: `python -m ingestion.pipeline`
"""
from __future__ import annotations

import asyncio
import logging

from ingestion.classifier import ChatterClassifier
from ingestion.embedder import Embedder, chunk_text
from ingestion.entity_extractor import EntityExtractor
from ingestion.storage_clients import GraphStore, VectorStore
from schemas.vector_payload_schema import ChunkPayload, ContentClass, SourceType
from security.pii_redaction import PIIRedactor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        classifier: ChatterClassifier | None = None,
        redactor: PIIRedactor | None = None,
        extractor: EntityExtractor | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
    ):
        self.classifier = classifier or ChatterClassifier()
        self.redactor = redactor or PIIRedactor()
        self.extractor = extractor or EntityExtractor()
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.graph_store = graph_store or GraphStore()

    async def process_message(
        self,
        text: str,
        workspace_id: str,
        channel_id: str,
        channel_name: str,
        author_id: str,
        author_display_name: str,
        source: SourceType = SourceType.SLACK,
        thread_id: str | None = None,
    ) -> None:
        classification = await self.classifier.classify_async(text)
        if not classification.keep:
            logger.debug("Dropped casual message from %s", author_id)
            return

        redaction = self._redact_or_drop(text, author_id)
        if redaction is None:
            return
        extraction = self.extractor.extract(redaction.redacted_text)

        # Write graph entities/relations
        entity_ids = []
        for entity in extraction.entities:
            self.graph_store.merge_entity(
                entity.id, entity.type, entity.name, workspace_id, channel_id=channel_id
            )
            entity_ids.append(entity.id)
        for relation in extraction.relations:
            self.graph_store.merge_relation(
                relation.source_id, relation.relation, relation.target_id, workspace_id
            )

        # Chunk + embed + write to vector store
        chunks = chunk_text(redaction.redacted_text)
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_passages(texts)  # "passage" mode — this text is being stored

        for chunk, embedding in zip(chunks, embeddings):
            payload = ChunkPayload(
                workspace_id=workspace_id,
                channel_id=channel_id,
                channel_name=channel_name,
                source=source,
                content_class=ContentClass(classification.content_class.value),
                text=chunk.text,
                text_raw_hash=redaction.raw_text_hash,
                author_id=author_id,
                author_display_name=author_display_name,
                thread_id=thread_id,
                entities=entity_ids,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_estimate,
            )
            self.vector_store.upsert_chunk(embedding, payload)

        logger.info(
            "Ingested message from %s/%s: %d chunks, %d entities, %d relations",
            channel_name, author_display_name, len(chunks), len(extraction.entities), len(extraction.relations),
        )

    async def process_meeting_segment(
        self,
        text: str,
        workspace_id: str,
        meeting_id: str,
        meeting_title: str,
        speaker_label: str,
        timestamp_offset_s: float,
        source: SourceType = SourceType.ZOOM,
    ) -> None:
        """Same pipeline as process_message, tagged for meeting provenance
        (speaker diarization label + offset instead of author_id/thread_id)."""
        classification = await self.classifier.classify_async(text)
        if not classification.keep:
            return

        redaction = self._redact_or_drop(text, speaker_label)
        if redaction is None:
            return
        extraction = self.extractor.extract(redaction.redacted_text)

        entity_ids = []
        for entity in extraction.entities:
            self.graph_store.merge_entity(
                entity.id, entity.type, entity.name, workspace_id, channel_id=meeting_id
            )
            entity_ids.append(entity.id)
        for relation in extraction.relations:
            self.graph_store.merge_relation(
                relation.source_id, relation.relation, relation.target_id, workspace_id
            )

        chunks = chunk_text(redaction.redacted_text)
        embeddings = self.embedder.embed_passages([c.text for c in chunks])  # "passage" mode — stored text

        for chunk, embedding in zip(chunks, embeddings):
            payload = ChunkPayload(
                workspace_id=workspace_id,
                channel_id=meeting_id,
                channel_name=meeting_title,
                source=source,
                content_class=ContentClass.MEETING_SEGMENT,
                text=chunk.text,
                text_raw_hash=redaction.raw_text_hash,
                speaker_label=speaker_label,
                meeting_id=meeting_id,
                meeting_timestamp_offset_s=timestamp_offset_s,
                entities=entity_ids,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_estimate,
            )
            self.vector_store.upsert_chunk(embedding, payload)

        logger.info("Ingested meeting segment from %s in %s", speaker_label, meeting_title)

    def _redact_or_drop(self, text: str, subject: str):
        """Fail closed: a redaction error drops the message instead of storing raw text."""
        try:
            return self.redactor.redact(text)
        except Exception:
            logger.exception("PII redaction failed; dropping content from %s", subject)
            return None


async def _smoke_test() -> None:
    """Runs the pipeline against fake stores if real ones aren't reachable —
    demonstrates the call shape without requiring live infra."""
    pipeline = IngestionPipeline()
    try:
        pipeline.vector_store.ensure_collection()
    except Exception as e:  # noqa: BLE001
        logger.warning("Qdrant not reachable (expected if infra isn't running): %s", e)

    await pipeline.process_message(
        text="We decided to migrate the payments-service deploy pipeline to GitHub Actions after Jenkins kept timing out.",
        workspace_id="ws_demo",
        channel_id="C123",
        channel_name="#eng-payments",
        author_id="U1",
        author_display_name="Priya",
    )


if __name__ == "__main__":
    asyncio.run(_smoke_test())
