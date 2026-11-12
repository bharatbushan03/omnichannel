import asyncio

from ingestion.classifier import ClassificationResult, ContentClass
from ingestion.pipeline import IngestionPipeline


class _Keep:
    async def classify_async(self, text):
        return ClassificationResult(ContentClass.TECHNICAL_DECISION, 0.99, keep=True)


class _Boom:
    def redact(self, text):
        raise RuntimeError("redactor unavailable")


class _MustNotRun:
    def extract(self, text):
        raise AssertionError("entity extraction ran after a redaction failure")

    def embed_passages(self, texts):
        raise AssertionError("embedding ran after a redaction failure")

    def merge_entity(self, *args, **kwargs):
        raise AssertionError("graph write ran after a redaction failure")

    def merge_relation(self, *args, **kwargs):
        raise AssertionError("graph write ran after a redaction failure")

    def upsert_chunk(self, *args, **kwargs):
        raise AssertionError("vector write ran after a redaction failure")


def test_message_is_dropped_when_redaction_fails():
    pipeline = IngestionPipeline(
        classifier=_Keep(),
        redactor=_Boom(),
        extractor=_MustNotRun(),
        embedder=_MustNotRun(),
        vector_store=_MustNotRun(),
        graph_store=_MustNotRun(),
    )
    asyncio.run(
        pipeline.process_message(
            text="email me at ada@example.com",
            workspace_id="ws",
            channel_id="C1",
            channel_name="#eng",
            author_id="U1",
            author_display_name="Ada",
        )
    )


def test_meeting_segment_is_dropped_when_redaction_fails():
    pipeline = IngestionPipeline(
        classifier=_Keep(),
        redactor=_Boom(),
        extractor=_MustNotRun(),
        embedder=_MustNotRun(),
        vector_store=_MustNotRun(),
        graph_store=_MustNotRun(),
    )
    asyncio.run(
        pipeline.process_meeting_segment(
            text="my key is sk-abcdefghijklmnopqrst1234",
            workspace_id="ws",
            meeting_id="bot-1",
            meeting_title="Standup",
            speaker_label="Ada",
            timestamp_offset_s=1.5,
        )
    )
