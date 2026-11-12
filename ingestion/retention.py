"""Scheduled purge for vector chunks, meeting transcripts, and audit logs.

Defaults (override with env, values are days):

- ``CHUNK_TTL_DAYS`` (365) — Slack and Teams chunks
- ``MEETING_TTL_DAYS`` (90) — Zoom and Google Meet transcript chunks, and Meeting nodes
- ``AUDIT_TTL_DAYS`` (730) — retrieval audit rows and meeting consent rows

Meeting consent is kept on the audit schedule so a disclosure can still be
shown after the transcript itself has been deleted.

Run: ``python -m ingestion.retention``
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from security.encryption import require_encryption_at_rest

load_dotenv()

logger = logging.getLogger(__name__)

MEETING_SOURCES = ("zoom", "google_meet", "teams_meeting")
CHAT_SOURCES = ("slack", "teams", "manual")


@dataclass(frozen=True)
class RetentionPolicy:
    chunk_ttl_days: int = 365
    meeting_ttl_days: int = 90
    audit_ttl_days: int = 730

    def chunk_cutoff(self, now: datetime) -> datetime:
        return now - timedelta(days=self.chunk_ttl_days)

    def meeting_cutoff(self, now: datetime) -> datetime:
        return now - timedelta(days=self.meeting_ttl_days)

    def audit_cutoff(self, now: datetime) -> datetime:
        return now - timedelta(days=self.audit_ttl_days)


def policy_from_env() -> RetentionPolicy:
    def _days(name: str, default: int) -> int:
        raw = os.environ.get(name, str(default))
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %s", name, raw, default)
            return default
        if value < 1:
            logger.warning("%s must be >= 1; using %s", name, default)
            return default
        return value

    return RetentionPolicy(
        chunk_ttl_days=_days("CHUNK_TTL_DAYS", 365),
        meeting_ttl_days=_days("MEETING_TTL_DAYS", 90),
        audit_ttl_days=_days("AUDIT_TTL_DAYS", 730),
    )


@dataclass
class PurgeReport:
    chat_chunks_deleted: int | None = None
    meeting_chunks_deleted: int | None = None
    meeting_nodes_deleted: int | None = None
    audit_rows_deleted: int | None = None
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def purge(
    *,
    vector_store=None,
    graph_store=None,
    audit_store=None,
    policy: RetentionPolicy | None = None,
    now: datetime | None = None,
) -> PurgeReport:
    """Delete expired knowledge. Each store is optional so a partial outage
    still purges the stores that are reachable. Errors are collected, not raised,
    so one down database does not skip the others."""
    policy = policy or policy_from_env()
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    report = PurgeReport()

    if vector_store is not None:
        try:
            report.chat_chunks_deleted = int(
                vector_store.delete_chunks_before(policy.chunk_cutoff(current), sources=list(CHAT_SOURCES))
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chat chunk purge failed")
            report.errors.append(f"chat_chunks: {exc}")
        try:
            report.meeting_chunks_deleted = int(
                vector_store.delete_chunks_before(policy.meeting_cutoff(current), sources=list(MEETING_SOURCES))
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Meeting transcript purge failed")
            report.errors.append(f"meeting_chunks: {exc}")

    if graph_store is not None:
        try:
            report.meeting_nodes_deleted = int(
                graph_store.delete_meetings_before(policy.meeting_cutoff(current).isoformat())
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Meeting graph purge failed")
            report.errors.append(f"meeting_nodes: {exc}")

    if audit_store is not None:
        try:
            report.audit_rows_deleted = int(audit_store.purge_audit_before(policy.audit_cutoff(current)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Audit log purge failed")
            report.errors.append(f"audit_rows: {exc}")

    logger.info(
        "Retention purge chat_chunks=%s meeting_chunks=%s meeting_nodes=%s audit_rows=%s errors=%s",
        report.chat_chunks_deleted,
        report.meeting_chunks_deleted,
        report.meeting_nodes_deleted,
        report.audit_rows_deleted,
        len(report.errors or []),
    )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    require_encryption_at_rest()
    from ingestion.storage_clients import GraphStore, PostgresStore, VectorStore

    store = PostgresStore()
    try:
        store.ensure_schema()
    except Exception:  # noqa: BLE001
        logger.exception("Postgres schema check failed; audit purge may no-op")
    report = purge(vector_store=VectorStore(), graph_store=GraphStore(), audit_store=store)
    if report.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
