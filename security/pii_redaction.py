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
# Cards and IPs are applied with checkers (Luhn, octet range, version prefix)
# so a 13-digit ticket or "version 1.2.3.4" is not wiped out of a decision log.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("SSN_US", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # A separator or a leading + is required. A bare 10–13 digit id is not a phone.
    ("PHONE", re.compile(
        r"(?<!\d)(?:"
        r"\+\d{1,3}[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}"
        r"|\(?\d{3}\)?[ -]\d{3}[ -]\d{4}"
        r")(?!\d)"
    )),
    # API keys / tokens — common prefixes (AWS, GitHub, Stripe, generic Bearer/OpenAI-style, Slack)
    ("API_KEY", re.compile(
        r"(?:AKIA[0-9A-Z]{16}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|sk-[A-Za-z0-9]{20,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|Bearer\s+[A-Za-z0-9\-_.]{20,})"
    )),
]

_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_VERSION_PREFIXES = frozenset({"version", "ver", "release", "rev", "v"})


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _replace_cards(text: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            count += 1
            return "[REDACTED_CREDIT_CARD]"
        return match.group(0)

    return _CARD_PATTERN.sub(repl, text), count


def _replace_ips(text: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        parts = match.group(0).split(".")
        if any(int(part) > 255 for part in parts):
            return match.group(0)
        prefix = text[max(0, match.start() - 24) : match.start()].lower().rstrip()
        token = prefix.split()[-1] if prefix else ""
        if token in _VERSION_PREFIXES:
            return match.group(0)
        count += 1
        return "[REDACTED_IP_ADDRESS]"

    return _IP_PATTERN.sub(repl, text), count


class RedactionError(RuntimeError):
    """Raised when a requested redaction pass cannot prove the text is safe to store."""


LLM_REDACT_SYSTEM_PROMPT = """You redact residual personal data from workplace text \
that has already had emails, phone numbers, and secrets replaced with [REDACTED_*] tokens.

Replace:
- personal names (people, not product or company names) with [REDACTED_NAME]
- street addresses with [REDACTED_ADDRESS]

Leave every existing [REDACTED_*] token unchanged. Do not add commentary.
Return only the redacted text."""


class PIIRedactor:
    def __init__(self, use_llm_pass: bool = False, llm_client=None, model: str | None = None):
        self.use_llm_pass = use_llm_pass
        self.llm_client = llm_client
        self.model = model

    def redact(self, text: str) -> RedactionResult:
        raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        redacted = text
        count = 0

        redacted, card_count = _replace_cards(redacted)
        count += card_count
        for label, pattern in PATTERNS:
            redacted, n = pattern.subn(f"[REDACTED_{label}]", redacted)
            count += n
        redacted, ip_count = _replace_ips(redacted)
        count += ip_count

        if self.use_llm_pass:
            redacted, llm_count = self._llm_redact(redacted)
            count += llm_count

        return RedactionResult(redacted_text=redacted, raw_text_hash=raw_hash, redaction_count=count)

    def _llm_redact(self, text: str) -> tuple[str, int]:
        """Secondary pass for names and addresses. Failure raises so the pipeline
        can drop the message instead of storing text the model did not redact."""
        if self.llm_client is None:
            raise RedactionError("LLM redaction was requested but no client is configured")

        from ingestion.nvidia_client import DEFAULT_CHAT_MODEL

        before = text.count("[REDACTED_")
        response = self.llm_client.chat.completions.create(
            model=self.model or DEFAULT_CHAT_MODEL,
            max_tokens=2048,  # required by NVIDIA NIM
            messages=[
                {"role": "system", "content": LLM_REDACT_SYSTEM_PROMPT},
                {"role": "user", "content": text[:6000]},
            ],
        )
        redacted = (response.choices[0].message.content or "").strip()
        if redacted.startswith("```"):
            lines = redacted.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            redacted = "\n".join(lines).strip()
        if not redacted:
            raise RedactionError("LLM redaction returned empty text")
        added = max(redacted.count("[REDACTED_") - before, 0)
        return redacted, added


if __name__ == "__main__":
    redactor = PIIRedactor()
    sample = "Contact me at jane.doe@company.com or 555-123-4567. Key: sk-abc123xyz789abcdef0123"
    result = redactor.redact(sample)
    print(result.redacted_text)
    print("Redactions:", result.redaction_count)
