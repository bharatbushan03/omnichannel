# Enterprise Omnichannel AI Knowledge & Voice Agent

A scaffold for an AI assistant that passively ingests knowledge from Slack/Teams
chat and Zoom/Meet calls, stores it in a hybrid vector + graph knowledge base,
and answers questions via text and low-latency voice.

**LLM/embedding provider: NVIDIA NIM (free tier).** All classification, entity
extraction, embedding, and voice-agent reasoning run on NVIDIA's free
build.nvidia.com endpoint (Nemotron for chat, `nv-embedqa-e5-v5` for
embeddings) via the standard OpenAI-compatible API. Get a free key at
https://build.nvidia.com. STT (Deepgram) and TTS (Cartesia) have no NVIDIA
equivalent and still require their own paid keys — there is no free
substitute for either in this stack.

**Status: production-shaped scaffold, not a drop-in deploy.** Every module runs
and is unit-tested where logic is testable offline (RRF fusion, PII redaction).
Modules that require live infra (Qdrant/Neo4j/Postgres) or paid APIs
(Deepgram/Cartesia/Recall.ai/LiveKit Cloud) are structurally correct and follow
each SDK's documented shape, but were not run end-to-end against live services —
verify against current SDK versions before deploying, especially `livekit-agents`
and Recall.ai's webhook payload shape, both of which change frequently.

## What was corrected from the original spec

1. **RRF ≠ vector+graph fusion.** RRF fuses two ranked lists of the *same*
   result type (dense vector + BM25 sparse). Graph traversal output is
   structured entity data, appended to context separately. See
   `docs/ARCHITECTURE.md` §1 for the full reasoning and `ingestion/hybrid_retrieval.py`
   for the implementation.
2. **Sub-800ms latency is a fast-path target, not a guarantee.** Turns that
   trigger a knowledge base lookup will typically exceed 1s once retrieval
   round-trip is included, and NVIDIA's free-tier endpoint adds further,
   less predictable latency on top of that — see below.
3. **Ingestion and voice-agent are separate runtimes** sharing a storage layer,
   not one unified pipeline.

## About the NVIDIA NIM free tier

- **Embeddings are asymmetric.** `nv-embedqa-e5-v5` requires an `input_type`
  of `"passage"` when storing text and `"query"` when searching. Get this
  wrong and you won't get an error — you'll get quietly worse search
  results. `ingestion/embedder.py` exposes `embed_passages()` and
  `embed_query()` specifically so call sites can't mix this up; don't call
  `embed_batch()`/`embed_one()` directly elsewhere in the codebase.
- **Vector dimension is 1024**, not 1536 — `schemas/vector_payload_schema.py`
  and the Qdrant collection config are already set to match
  `nv-embedqa-e5-v5`'s output. If you swap embedding models later, update
  `size` in `QDRANT_COLLECTION_CONFIG` to match or every upsert will fail.
- **`max_tokens` is required on every chat completion call** — NVIDIA NIM
  rejects requests that omit it. Already set in `classifier.py` and
  `entity_extractor.py`; keep this in mind if you add new call sites.
- **Free-tier rate limits and latency are real constraints.** It's a shared
  endpoint, not a dedicated deployment. Expect throttling under load and
  inconsistent first-token latency in the voice agent — likely worse than
  the already-optimistic 800ms target in the original spec. Fine for a
  demo; revisit before anything real.
- **Deepgram and Cartesia still need their own keys.** Nemotron only
  replaces the LLM and embedding calls. There is no free STT/TTS swap
  available in this stack — the voice agent will not run without both.

## Project layout

```
docs/ARCHITECTURE.md          Full architecture, Mermaid diagram, data flow, latency budget
schemas/
  vector_payload_schema.py    Qdrant collection config + payload dataclass (shared by ingestion + retrieval)
  neo4j_schema.cypher         Constraints, indexes, node/relationship contracts, example queries
  postgres_acl_schema.sql     Workspace/channel ACL tables, audit log, meeting consent log
ingestion/
  nvidia_client.py             Shared NVIDIA NIM client config (base_url, models, API key handling)
  classifier.py                Chatter classifier (technical_decision/debug_log/schedule/doc_reference/casual) — Nemotron
  entity_extractor.py          LLM-based (entity, relation, entity) triple extraction — Nemotron
  embedder.py                  Chunking + embedding generation — NVIDIA nv-embedqa-e5-v5
  storage_clients.py           Qdrant / Neo4j / Postgres client wrappers
  hybrid_retrieval.py          RRF fusion + graph traversal + context assembly
  pipeline.py                  Orchestrates classify -> redact -> extract -> embed -> store
  slack_listener.py            FastAPI webhook for Slack Events API (message.channels, app_mention)
  meeting_listener.py          Recall.ai bot dispatch + transcript webhook receiver (Zoom/Meet)
voice_agent/
  agent.py                     LiveKit Agents entrypoint — Deepgram STT -> Nemotron (tool-calling) -> Cartesia TTS
  kb_tool.py                   The `lookup_knowledge_base` function tool called mid-conversation
security/
  pii_redaction.py             Regex-based PII redaction (+ optional LLM pass hook)
  governance.md                Full security/governance strategy, including known gaps
tests/                         Unit tests for RRF fusion and PII redaction (9/9 passing)
docker-compose.yml             Local Qdrant + Neo4j + Postgres for development
requirements.txt
.env.example
```

## Setup

```bash
# 1. Local infra
docker-compose up -d

# 2. Python deps
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
# minimum required: NVIDIA_API_KEY (free at https://build.nvidia.com)
# for the voice agent, also: DEEPGRAM_API_KEY, CARTESIA_API_KEY, LIVEKIT_*

# 4. Initialize Neo4j schema
cat schemas/neo4j_schema.cypher | cypher-shell -u neo4j -p localdevpassword

# 5. Run tests
pytest tests/ -v
```

## Running the components

```bash
# Slack ingestion webhook
uvicorn ingestion.slack_listener:app --port 8001

# Meeting (Zoom/Meet) ingestion webhook
uvicorn ingestion.meeting_listener:app --port 8002

# Voice agent worker (connects to LiveKit Cloud/self-hosted)
python -m voice_agent.agent dev
```

## Before production

Read `security/governance.md` §6 ("Known gaps to close") — it lists specific
unfinished items: fail-closed redaction error handling, real Recall.ai webhook
signature verification, audit log write-through, jurisdiction-aware meeting
consent, rate limiting, encryption-at-rest, and data retention policy. None of
these are optional for a real enterprise deployment. Separately, swap off the
NVIDIA free tier before production — a shared rate-limited endpoint is not
appropriate for a live enterprise workload.
