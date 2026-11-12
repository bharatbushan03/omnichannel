from security.pii_redaction import PIIRedactor


def test_redacts_email():
    r = PIIRedactor()
    result = r.redact("Reach me at test@example.com please")
    assert "test@example.com" not in result.redacted_text
    assert "[REDACTED_EMAIL]" in result.redacted_text


def test_redacts_api_key():
    r = PIIRedactor()
    result = r.redact("token is sk-abcdefghijklmnopqrst1234")
    assert "sk-abcdefghijklmnopqrst1234" not in result.redacted_text


def test_redacts_phone():
    r = PIIRedactor()
    result = r.redact("call me at 555-123-4567")
    assert "555-123-4567" not in result.redacted_text


def test_no_false_positive_on_plain_text():
    r = PIIRedactor()
    result = r.redact("We deployed the fix at 3pm and it worked fine.")
    assert result.redacted_text == "We deployed the fix at 3pm and it worked fine."
    assert result.redaction_count == 0


def test_version_string_is_not_an_ip_and_ticket_ids_are_not_cards():
    r = PIIRedactor()
    version = r.redact("version 1.2.3.4 of the agent")
    assert version.redacted_text == "version 1.2.3.4 of the agent"
    ticket = r.redact("ticket 1234567890123 opened")
    assert "1234567890123" in ticket.redacted_text
    card = r.redact("card 4111111111111111 on file")
    assert "4111111111111111" not in card.redacted_text
    assert "[REDACTED_CREDIT_CARD]" in card.redacted_text
    ip = r.redact("IP 10.1.2.3 in the vpc")
    assert "10.1.2.3" not in ip.redacted_text


def test_hash_is_deterministic():
    r = PIIRedactor()
    result1 = r.redact("same input")
    result2 = r.redact("same input")
    assert result1.raw_text_hash == result2.raw_text_hash
