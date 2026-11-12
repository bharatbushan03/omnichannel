import pytest

from ingestion.hybrid_retrieval import HybridRetriever
from ingestion.storage_clients import validate_entity_type, validate_relation


class _Embedder:
    def embed_query(self, query):
        return [0.1, 0.2]


class _Vector:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("allowed_channel_ids") == []:
            return []
        return [
            {
                "chunk_id": "c1",
                "text": "retry the payments deploy",
                "source": "slack",
                "channel_name": "#eng",
                "timestamp": "2026-01-01T00:00:00",
            }
        ]


class _Graph:
    def traverse_from_entity(self, fragment, workspace_id, limit=10):
        return [{"source_name": "payments", "relation": "HAS_ISSUE", "target_name": "500s"}]


class _Audit:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail

    def write_retrieval_audit(self, **kwargs):
        if self.fail:
            raise RuntimeError("postgres down")
        self.rows.append(kwargs)


class _ACL:
    def __init__(self):
        self.membership = {
            ("ws", "u1"): ["C1"],
            ("ws", "u2"): ["C1", "C2"],
        }

    def get_allowed_channels(self, workspace_id, user_id):
        if user_id == "broken":
            raise RuntimeError("acl down")
        return list(self.membership.get((workspace_id, user_id), []))


def _retriever(**kwargs):
    vector = _Vector()
    audit = kwargs.pop("audit", _Audit())
    acl = kwargs.pop("acl", _ACL())
    retriever = HybridRetriever(
        vector_store=vector,
        graph_store=_Graph(),
        embedder=_Embedder(),
        audit_log=audit,
        acl_store=acl,
        audit_fail_closed=kwargs.pop("audit_fail_closed", False),
    )
    return retriever, vector, audit


def test_retrieve_writes_audit_row():
    retriever, _vector, audit = _retriever()
    context = retriever.retrieve("payments 500", "ws", allowed_channel_ids=["C1"], user_id="u1")
    assert context.chunks[0].chunk_id == "c1"
    assert audit.rows[0]["result_chunk_ids"] == ["c1"]
    assert audit.rows[0]["channel_scope"] == ["C1"]
    assert audit.rows[0]["source_interface"] == "voice"
    assert "HAS_ISSUE" in context.graph.facts[0]


def test_shared_channel_scope_is_the_intersection():
    retriever, vector, _audit = _retriever()
    retriever.retrieve(
        "payments",
        "ws",
        participant_user_ids=["u1", "u2"],
    )
    assert vector.calls[0]["allowed_channel_ids"] == ["C1"]


def test_acl_failure_returns_no_chunks():
    retriever, vector, _audit = _retriever()
    context = retriever.retrieve("payments", "ws", user_id="broken")
    assert context.chunks == []
    assert context.graph.facts == []
    assert vector.calls == []


def test_audit_failure_can_withhold_results():
    retriever, _vector, _audit = _retriever(audit=_Audit(fail=True), audit_fail_closed=True)
    with pytest.raises(RuntimeError):
        retriever.retrieve("payments", "ws", allowed_channel_ids=["C1"])


def test_missing_audit_log_is_fatal_when_required():
    retriever = HybridRetriever(
        vector_store=_Vector(),
        graph_store=_Graph(),
        embedder=_Embedder(),
        audit_log=None,
        audit_fail_closed=True,
    )
    with pytest.raises(RuntimeError):
        retriever.retrieve("payments", "ws", allowed_channel_ids=["C1"])


def test_graph_labels_reject_injection():
    with pytest.raises(ValueError):
        validate_entity_type("Project}) DETACH DELETE n")
    with pytest.raises(ValueError):
        validate_relation("HAS_ISSUE})-[:PWNED]->(")
    assert validate_entity_type("Project") == "Project"
    assert validate_relation("RESOLVED_BY") == "RESOLVED_BY"
