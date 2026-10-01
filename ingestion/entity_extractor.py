"""
Extracts (entity, relation, entity) triples from redacted text for the
Neo4j knowledge graph, using structured LLM output.

Output triples map directly onto the relationship types defined in
schemas/neo4j_schema.cypher — this module does not invent new relation
types; it constrains the LLM to the fixed vocabulary below so the graph
stays queryable instead of accumulating one-off edge labels.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import openai

from ingestion.nvidia_client import DEFAULT_CHAT_MODEL, get_nvidia_client

logger = logging.getLogger(__name__)

ALLOWED_ENTITY_TYPES = ["Project", "Issue", "DeployScript", "Person", "Decision", "Document", "Meeting"]
ALLOWED_RELATIONS = [
    "HAS_ISSUE", "RESOLVED_BY", "REPORTED", "OWNS",
    "MADE_DECISION", "AFFECTS", "DISCUSSED", "REFERENCED_IN",
]

EXTRACTION_SYSTEM_PROMPT = f"""Extract structured entities and relationships from \
workplace technical text. Respond with ONLY a JSON object, no other text.

Allowed entity types: {ALLOWED_ENTITY_TYPES}
Allowed relation types: {ALLOWED_RELATIONS}

Rules:
- Only extract entities/relations that are explicitly stated or strongly implied.
- If nothing extractable is present, return {{"entities": [], "relations": []}}.
- Entity "id" should be a short stable slug derived from the name (lowercase, hyphenated).

Output schema:
{{
  "entities": [{{"id": "...", "type": "...", "name": "..."}}],
  "relations": [{{"source_id": "...", "relation": "...", "target_id": "..."}}]
}}
"""


@dataclass
class ExtractedEntity:
    id: str
    type: str
    name: str


@dataclass
class ExtractedRelation:
    source_id: str
    relation: str
    target_id: str


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


class EntityExtractor:
    def __init__(self, client: openai.OpenAI | None = None, model: str = DEFAULT_CHAT_MODEL):
        self.client = client or get_nvidia_client()
        self.model = model

    def extract(self, text: str) -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,  # required by NVIDIA NIM — omitting this causes a server error
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text[:6000]},
                ],
            )
            raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning("Entity extraction parse failure: %s", e)
            return ExtractionResult()

        entities = [
            ExtractedEntity(id=e["id"], type=e["type"], name=e["name"])
            for e in parsed.get("entities", [])
            if e.get("type") in ALLOWED_ENTITY_TYPES
        ]
        valid_ids = {e.id for e in entities}
        relations = [
            ExtractedRelation(source_id=r["source_id"], relation=r["relation"], target_id=r["target_id"])
            for r in parsed.get("relations", [])
            if r.get("relation") in ALLOWED_RELATIONS
            and r.get("source_id") in valid_ids
            and r.get("target_id") in valid_ids
        ]
        return ExtractionResult(entities=entities, relations=relations)


if __name__ == "__main__":
    extractor = EntityExtractor()
    sample = (
        "Priya reported that the payments-service was throwing 500s under load. "
        "We resolved it by rolling out the retry-backoff deploy script."
    )
    result = extractor.extract(sample)
    print(result)
