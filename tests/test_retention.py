from datetime import datetime, timedelta, timezone

from ingestion.retention import RetentionPolicy, purge


class _Vector:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def delete_chunks_before(self, cutoff, sources=None):
        if self.fail and sources and "slack" in sources:
            raise RuntimeError("qdrant down")
        self.calls.append((cutoff, tuple(sources or [])))
        return 4


class _Graph:
    def __init__(self):
        self.cutoffs = []

    def delete_meetings_before(self, cutoff_iso):
        self.cutoffs.append(cutoff_iso)
        return 2


class _Audit:
    def __init__(self):
        self.cutoffs = []

    def purge_audit_before(self, cutoff):
        self.cutoffs.append(cutoff)
        return 7


def test_purge_uses_separate_ttls():
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)
    policy = RetentionPolicy(chunk_ttl_days=10, meeting_ttl_days=5, audit_ttl_days=20)
    vector, graph, audit = _Vector(), _Graph(), _Audit()
    report = purge(vector_store=vector, graph_store=graph, audit_store=audit, policy=policy, now=now)

    chat_cutoff, chat_sources = vector.calls[0]
    meeting_cutoff, meeting_sources = vector.calls[1]
    assert chat_cutoff == now - timedelta(days=10)
    assert chat_sources == ("slack", "teams", "manual")
    assert meeting_cutoff == now - timedelta(days=5)
    assert meeting_sources == ("zoom", "google_meet", "teams_meeting")
    assert graph.cutoffs[0] == (now - timedelta(days=5)).isoformat()
    assert audit.cutoffs[0] == now - timedelta(days=20)
    assert report.chat_chunks_deleted == 4
    assert report.meeting_nodes_deleted == 2
    assert report.audit_rows_deleted == 7
    assert report.errors == []


def test_one_store_failing_does_not_skip_the_others():
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)
    policy = RetentionPolicy(chunk_ttl_days=10, meeting_ttl_days=5, audit_ttl_days=20)
    vector, graph, audit = _Vector(fail=True), _Graph(), _Audit()
    report = purge(vector_store=vector, graph_store=graph, audit_store=audit, policy=policy, now=now)
    assert report.chat_chunks_deleted is None
    assert report.meeting_chunks_deleted == 4
    assert report.audit_rows_deleted == 7
    assert report.errors and report.errors[0].startswith("chat_chunks:")
