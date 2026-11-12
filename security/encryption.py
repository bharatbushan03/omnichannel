"""Encryption-at-rest gate.

Postgres (community) and Neo4j (community) do not encrypt data files themselves.
At rest, protection is the disk or the managed service: BitLocker or an encrypted
cloud volume locally, and transparent data encryption on a managed Postgres or
Neo4j deployment. This module does not pretend to encrypt those files. It
refuses to start production processes until an operator attests that the volume
or managed service is encrypted, via ``ENCRYPTION_AT_REST=disk`` or ``managed``.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ACCEPTED = frozenset({"disk", "managed", "enabled"})


def encryption_at_rest_mode() -> str:
    return os.environ.get("ENCRYPTION_AT_REST", "unconfigured").strip().lower()


def require_encryption_at_rest() -> None:
    mode = encryption_at_rest_mode()
    if mode in ACCEPTED:
        logger.info("Encryption at rest attested as %s", mode)
        return
    if os.environ.get("APP_ENV", "development").strip().lower() == "production":
        raise RuntimeError(
            "ENCRYPTION_AT_REST must be 'disk' or 'managed' when APP_ENV=production. "
            "Enable disk encryption or managed transparent data encryption for the "
            "Postgres, Neo4j, and Qdrant volumes before starting."
        )
    logger.warning(
        "ENCRYPTION_AT_REST is %s. Local development may continue; production startup will not.",
        mode or "unconfigured",
    )
