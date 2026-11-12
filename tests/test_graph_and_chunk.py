from ingestion.embedder import chunk_text
from ingestion.storage_clients import entity_merge_query, relation_merge_query


def test_graph_writes_are_scoped_to_the_workspace():
    entity = entity_merge_query("Project")
    relation = relation_merge_query("HAS_ISSUE")
    assert "{id: $entity_id, workspace_id: $workspace_id}" in entity
    assert "workspace_id: $workspace_id" in relation
    assert "MATCH (a {id: $source_id})" not in relation


def test_chunk_window_still_finishes_when_overlap_is_too_large():
    text = "abcdefghij" * 30
    chunks = chunk_text(text, max_chars=20, overlap_chars=100)
    assert chunks
    assert chunks[-1].text.endswith(text[-1])
    assert len(chunks) < len(text)
