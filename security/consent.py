"""Jurisdiction-aware meeting consent.

Every meeting is disclosed with the same sentence. Whether a transcript segment
may be stored depends on the workspace jurisdiction and on what to do when
someone objects:

- ``one_party``: disclosure is sent on join; segments may be stored before the
  announcement is confirmed on this process. An objection still stops or
  excludes the speaker, according to policy.
- ``two_party``: nothing is stored until the disclosure is recorded. An
  objection follows ``on_objection`` (stop the meeting, or drop that speaker).
- ``all_party``: same as two-party, except any objection stops the whole meeting.
  Excluding a single speaker is not enough when every participant must agree.

Stopping the recording is the default. Excluding one speaker is only available
where the jurisdiction allows the rest of the meeting to continue.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

DISCLOSURE_TEXT = "AI Knowledge Agent joined and is recording for knowledge capture"

_OBJECTION = re.compile(
    r"\b("
    r"i (?:do not|don't|dont) consent"
    r"|stop recording"
    r"|(?:do not|don't|dont) record(?: this)?"
    r"|i object to (?:the |this )?recording"
    r"|revoke (?:my )?consent"
    r"|turn off the recording"
    r")\b",
    re.IGNORECASE,
)


class Jurisdiction(str, Enum):
    ONE_PARTY = "one_party"
    TWO_PARTY = "two_party"
    ALL_PARTY = "all_party"


class ObjectionAction(str, Enum):
    STOP = "stop"
    EXCLUDE_SPEAKER = "exclude_speaker"


@dataclass(frozen=True)
class SegmentDecision:
    allow_ingest: bool
    stop_recording: bool
    exclude_speaker: bool
    reason: str


def objection_in(text: str) -> bool:
    return bool(_OBJECTION.search(text or ""))


def parse_jurisdiction(value: str | None = None) -> Jurisdiction:
    raw = (value if value is not None else os.environ.get("MEETING_JURISDICTION", "two_party")).strip().lower()
    try:
        return Jurisdiction(raw)
    except ValueError:
        logger.warning("Unknown MEETING_JURISDICTION %r; using two_party", raw)
        return Jurisdiction.TWO_PARTY


def parse_objection_action(value: str | None = None) -> ObjectionAction:
    raw = (value if value is not None else os.environ.get("MEETING_ON_OBJECTION", "stop")).strip().lower()
    try:
        return ObjectionAction(raw)
    except ValueError:
        logger.warning("Unknown MEETING_ON_OBJECTION %r; using stop", raw)
        return ObjectionAction.STOP


def decide_segment(
    *,
    announced: bool,
    already_stopped: bool,
    speaker_excluded: bool,
    objection: bool,
    jurisdiction: Jurisdiction,
    on_objection: ObjectionAction,
) -> SegmentDecision:
    if already_stopped:
        return SegmentDecision(False, False, False, "recording_stopped")
    # An objection stops or excludes even when the disclosure has not been
    # recorded yet. Checking it later left the bot in the call.
    if objection:
        action = ObjectionAction.STOP if jurisdiction == Jurisdiction.ALL_PARTY else on_objection
        if action == ObjectionAction.STOP:
            return SegmentDecision(False, True, False, "objection_stop")
        return SegmentDecision(False, False, True, "objection_exclude_speaker")
    if jurisdiction != Jurisdiction.ONE_PARTY and not announced:
        return SegmentDecision(False, False, False, "not_announced")
    if speaker_excluded:
        return SegmentDecision(False, False, False, "speaker_excluded")
    return SegmentDecision(True, False, False, "ok")


def speaker_key(speaker: str) -> str:
    return (speaker or "").strip().casefold()


class MeetingConsentTracker:
    """In-process consent state, written through to Postgres when a store is attached."""

    def __init__(
        self,
        store=None,
        jurisdiction: Jurisdiction | None = None,
        on_objection: ObjectionAction | None = None,
    ):
        self.store = store
        self.jurisdiction = jurisdiction or parse_jurisdiction()
        self.on_objection = on_objection or parse_objection_action()
        self._announced: set[str] = set()
        self._stopped: set[str] = set()
        self._excluded: dict[str, set[str]] = {}
        self._jurisdictions: dict[str, Jurisdiction] = {}
        self._hydrated: set[str] = set()

    def mark_announced(self, meeting_id: str, workspace_id: str, jurisdiction: str | None = None) -> None:
        if not meeting_id:
            return
        chosen = parse_jurisdiction(jurisdiction) if jurisdiction else self.jurisdiction
        self._jurisdictions[meeting_id] = chosen
        self._announced.add(meeting_id)
        self._write(
            "record_announcement",
            meeting_id=meeting_id,
            workspace_id=workspace_id,
            jurisdiction=chosen.value,
            disclosure_text=DISCLOSURE_TEXT,
        )

    def is_announced(self, meeting_id: str) -> bool:
        if meeting_id in self._announced:
            return True
        if self.store is None or not hasattr(self.store, "consent_announced"):
            return False
        try:
            announced = bool(self.store.consent_announced(meeting_id))
        except Exception:
            logger.exception("Consent lookup failed for meeting %s", meeting_id)
            return False
        if announced:
            self._announced.add(meeting_id)
        return announced

    def is_stopped(self, meeting_id: str) -> bool:
        if meeting_id in self._stopped:
            return True
        if self.store is None or not hasattr(self.store, "recording_stopped"):
            return False
        try:
            stopped = bool(self.store.recording_stopped(meeting_id))
        except Exception:
            logger.exception("Consent stop-state lookup failed for meeting %s", meeting_id)
            return False
        if stopped:
            self._stopped.add(meeting_id)
        return stopped

    def _hydrate(self, meeting_id: str) -> None:
        """Load jurisdiction, stop, and excluded speakers recorded by another process."""
        if meeting_id in self._hydrated:
            return
        if self.store is None or not hasattr(self.store, "get_consent"):
            self._hydrated.add(meeting_id)
            return
        try:
            row = self.store.get_consent(meeting_id)
        except Exception:
            logger.exception("Consent hydrate failed for meeting %s", meeting_id)
            return
        self._hydrated.add(meeting_id)
        if not row:
            return
        self._announced.add(meeting_id)
        if row.get("jurisdiction") and meeting_id not in self._jurisdictions:
            self._jurisdictions[meeting_id] = parse_jurisdiction(str(row["jurisdiction"]))
        if row.get("stopped"):
            self._stopped.add(meeting_id)
        for name in row.get("excluded_speakers") or []:
            self._excluded.setdefault(meeting_id, set()).add(speaker_key(str(name)))

    def evaluate(self, meeting_id: str, workspace_id: str, speaker: str, text: str) -> SegmentDecision:
        self._hydrate(meeting_id)
        key = speaker_key(speaker)
        decision = decide_segment(
            announced=self.is_announced(meeting_id),
            already_stopped=self.is_stopped(meeting_id),
            speaker_excluded=key in self._excluded.get(meeting_id, set()),
            objection=objection_in(text),
            jurisdiction=self._jurisdictions.get(meeting_id, self.jurisdiction),
            on_objection=self.on_objection,
        )
        if decision.reason == "objection_stop":
            self._stopped.add(meeting_id)
            self._persist_objection(meeting_id, workspace_id, speaker, ObjectionAction.STOP.value)
        elif decision.reason == "objection_exclude_speaker":
            self._excluded.setdefault(meeting_id, set()).add(key)
            self._persist_objection(meeting_id, workspace_id, speaker, ObjectionAction.EXCLUDE_SPEAKER.value)
        return decision

    def _persist_objection(self, meeting_id: str, workspace_id: str, speaker: str, policy_action: str) -> None:
        self._write(
            "record_objection",
            meeting_id=meeting_id,
            workspace_id=workspace_id,
            speaker=speaker,
            policy_action=policy_action,
            excluded_speakers=sorted(self._excluded.get(meeting_id, set())),
            jurisdiction=self._jurisdictions.get(meeting_id, self.jurisdiction).value,
        )

    def _write(self, method: str, **kwargs) -> None:
        if self.store is None or not hasattr(self.store, method):
            return
        try:
            getattr(self.store, method)(**kwargs)
        except Exception:
            logger.exception("Failed to persist meeting consent via %s", method)
