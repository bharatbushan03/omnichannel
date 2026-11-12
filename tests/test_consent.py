from security.consent import (
    Jurisdiction,
    MeetingConsentTracker,
    ObjectionAction,
    decide_segment,
    objection_in,
)


def test_objection_phrases():
    assert objection_in("Please stop recording this call")
    assert objection_in("I don't consent to this")
    assert objection_in("I do not consent")
    assert not objection_in("We should record the decision in the runbook")


def test_two_party_blocks_until_announced():
    decision = decide_segment(
        announced=False,
        already_stopped=False,
        speaker_excluded=False,
        objection=False,
        jurisdiction=Jurisdiction.TWO_PARTY,
        on_objection=ObjectionAction.STOP,
    )
    assert decision.allow_ingest is False
    assert decision.reason == "not_announced"


def test_one_party_allows_before_announcement_is_confirmed():
    decision = decide_segment(
        announced=False,
        already_stopped=False,
        speaker_excluded=False,
        objection=False,
        jurisdiction=Jurisdiction.ONE_PARTY,
        on_objection=ObjectionAction.STOP,
    )
    assert decision.allow_ingest is True


def test_all_party_objection_stops_even_if_policy_is_exclude():
    decision = decide_segment(
        announced=True,
        already_stopped=False,
        speaker_excluded=False,
        objection=True,
        jurisdiction=Jurisdiction.ALL_PARTY,
        on_objection=ObjectionAction.EXCLUDE_SPEAKER,
    )
    assert decision.stop_recording is True
    assert decision.allow_ingest is False


def test_two_party_can_exclude_one_speaker():
    decision = decide_segment(
        announced=True,
        already_stopped=False,
        speaker_excluded=False,
        objection=True,
        jurisdiction=Jurisdiction.TWO_PARTY,
        on_objection=ObjectionAction.EXCLUDE_SPEAKER,
    )
    assert decision.exclude_speaker is True
    assert decision.stop_recording is False
    assert decision.allow_ingest is False


def test_jurisdiction_is_stored_per_meeting():
    tracker = MeetingConsentTracker(
        jurisdiction=Jurisdiction.TWO_PARTY,
        on_objection=ObjectionAction.EXCLUDE_SPEAKER,
    )
    tracker.mark_announced("all", "ws", jurisdiction="all_party")
    tracker.mark_announced("two", "ws", jurisdiction="two_party")
    assert tracker.evaluate("all", "ws", "Ada", "stop recording").stop_recording is True
    assert tracker.evaluate("two", "ws", "Ada", "stop recording").exclude_speaker is True


def test_objection_before_announcement_still_stops():
    decision = decide_segment(
        announced=False,
        already_stopped=False,
        speaker_excluded=False,
        objection=True,
        jurisdiction=Jurisdiction.TWO_PARTY,
        on_objection=ObjectionAction.STOP,
    )
    assert decision.stop_recording is True
    assert decision.allow_ingest is False


def test_excluded_speaker_matches_case_and_reloads_from_store():
    class _Store:
        def get_consent(self, meeting_id):
            return {
                "jurisdiction": "all_party",
                "stopped": False,
                "excluded_speakers": ["Ada"],
            }

    tracker = MeetingConsentTracker(
        store=_Store(),
        jurisdiction=Jurisdiction.TWO_PARTY,
        on_objection=ObjectionAction.STOP,
    )
    skipped = tracker.evaluate("bot-1", "ws", " ada ", "the deploy is fine")
    assert skipped.reason == "speaker_excluded"
    stopped = tracker.evaluate("bot-1", "ws", "Grace", "please stop recording")
    assert stopped.stop_recording is True


def test_tracker_remembers_exclusion():
    tracker = MeetingConsentTracker(
        jurisdiction=Jurisdiction.TWO_PARTY,
        on_objection=ObjectionAction.EXCLUDE_SPEAKER,
    )
    tracker.mark_announced("bot-1", "ws")
    first = tracker.evaluate("bot-1", "ws", "Ada", "stop recording")
    assert first.reason == "objection_exclude_speaker"
    second = tracker.evaluate("bot-1", "ws", "ada", "the deploy is fine")
    assert second.reason == "speaker_excluded"
    third = tracker.evaluate("bot-1", "ws", "Grace", "the deploy is fine")
    assert third.allow_ingest is True
