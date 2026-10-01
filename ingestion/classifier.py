"""
Lightweight classifier that filters raw channel/meeting text down to
knowledge-worthy content before it hits the (expensive) NER + embedding
stages of the pipeline.

Uses a free-tier NVIDIA NIM (Nemotron) call with a constrained JSON output.
Swap `model` for a distilled classifier (e.g. a fine-tuned DistilBERT) in
production — an LLM call per message is fine at hackathon/demo scale, not
at high-volume enterprise scale.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import openai

from ingestion.nvidia_client import DEFAULT_CHAT_MODEL, get_nvidia_client

logger = logging.getLogger(__name__)


class ContentClass(str, Enum):
    TECHNICAL_DECISION = "technical_decision"
    DEBUG_LOG = "debug_log"
    SCHEDULE = "schedule"
    DOC_REFERENCE = "doc_reference"
    CASUAL = "casual"


@dataclass
class ClassificationResult:
    content_class: ContentClass
    confidence: float
    keep: bool  # False for CASUAL — pipeline should drop the message


CLASSIFIER_SYSTEM_PROMPT = """You classify a single workplace chat message or \
meeting transcript segment into exactly one category. Respond with ONLY a JSON \
object, no other text.

Categories:
- technical_decision: a decision about architecture, tooling, approach, or process
- debug_log: an error, stack trace, incident report, or troubleshooting exchange
- schedule: a date, deadline, meeting time, or planning commitment
- doc_reference: a link to or summary of external documentation/specs
- casual: greetings, small talk, reactions, off-topic chat — anything with no
  durable operational value

Output schema:
{"content_class": "<one of the categories above>", "confidence": <0.0-1.0>}
"""


class ChatterClassifier:
    def __init__(self, client: Optional[openai.OpenAI] = None, model: str = DEFAULT_CHAT_MODEL):
        self.client = client or get_nvidia_client()
        self.model = model

    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(ContentClass.CASUAL, 1.0, keep=False)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=100,  # required by NVIDIA NIM — omitting this causes a server error
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": text[:4000]},
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if the model wraps its JSON
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            content_class = ContentClass(parsed["content_class"])
            confidence = float(parsed["confidence"])
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            logger.warning("Classifier parse failure, defaulting to casual/drop: %s", e)
            return ClassificationResult(ContentClass.CASUAL, 0.0, keep=False)

        keep = content_class != ContentClass.CASUAL and confidence >= 0.5
        return ClassificationResult(content_class, confidence, keep=keep)

    async def classify_async(self, text: str) -> ClassificationResult:
        """Async variant — use in async ingestion workers to avoid blocking the event loop."""
        import asyncio
        return await asyncio.to_thread(self.classify, text)


if __name__ == "__main__":
    clf = ChatterClassifier()
    samples = [
        "lol yeah that meeting ran way too long",
        "We decided to switch the deploy pipeline from Jenkins to GitHub Actions to cut build time.",
        "Traceback (most recent call last): ConnectionError at db_pool.py:44",
    ]
    for s in samples:
        print(s[:50], "->", clf.classify(s))
