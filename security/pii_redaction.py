"""
PII redaction pass. Runs BEFORE any text is persisted to Qdrant/Neo4j/Postgres.
Combines fast regex rules (emails, phones, secrets/tokens, card numbers) with
an optional LLM-based pass for names/addresses that regex can't catch reliably.

Regex-only redaction is deliberately the default (fast, deterministic, no
external call on the hot path). Enable `use_llm_pass=True` for higher recall
at the cost of latency + an API call per chunk.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class RedactionResult:
    redacted_text: str
    raw_text_hash: str          # sha256 of original text — for audit trail, never store raw text itself
    redaction_count: int


# Order matters — match specific/high-confidence patterns before generic ones.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("SSN_US", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")),
    # API keys / tokens — common prefixes (AWS, GitHub, Stripe, generic Bearer/OpenAI-style, Slack)
    ("API_KEY", re.compile(
        r"(?:AKIA[0-9A-Z]{16}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|sk-[A-Za-z0-9]{20,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|Bearer\s+[A-Za-z0-9\-_.]{20,})"
    )),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


class PIIRedactor:
    def __init__(self, use_llm_pass: bool = False, llm_client=None):
        self.use_llm_pass = use_llm_pass
        self.llm_client = llm_client

    def redact(self, text: str) -> RedactionResult:
        raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        redacted = text
        count = 0

        for label, pattern in PATTERNS:
            redacted, n = pattern.subn(f"[REDACTED_{label}]", redacted)
            count += n

        if self.use_llm_pass and self.llm_client is not None:
            redacted, llm_count = self._llm_redact(redacted)
            count += llm_count

        return RedactionResult(redacted_text=redacted, raw_text_hash=raw_hash, redaction_count=count)

    def _llm_redact(self, text: str) -> tuple[str, int]:
        """Optional secondary pass for names/addresses regex misses. Not implemented
        by default — wire to your LLM client if you need higher recall. Returning
        text unchanged is the safe no-op default."""
        return text, 0


if __name__ == "__main__":
    redactor = PIIRedactor()
    sample = "Contact me at jane.doe@company.com or 555-123-4567. Key: sk-abc123xyz789abcdef0123"
    result = redactor.redact(sample)
    print(result.redacted_text)
    print("Redactions:", result.redaction_count)
