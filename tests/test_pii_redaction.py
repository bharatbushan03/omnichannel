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


def test_hash_is_deterministic():
    r = PIIRedactor()
    result1 = r.redact("same input")
    result2 = r.redact("same input")
    assert result1.raw_text_hash == result2.raw_text_hash
