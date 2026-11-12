# Security & Governance Strategy

## 1. PII Redaction

- **Where**: applied in `security/pii_redaction.py`, invoked inside `IngestionPipeline` before any text reaches embedding, entity extraction, or storage. Unredacted text is never persisted — only a SHA-256 hash of the original is kept, for audit correlation, not recovery.
- **What's caught by regex (fast path, always on)**: emails, phone numbers, SSNs, credit-card-like sequences, IP addresses, common API key/token formats (AWS, GitHub, Stripe-style, Slack, generic Bearer tokens).
- **What regex misses**: full names in prose, physical addresses, indirect identifiers. Optional LLM-based secondary pass (`use_llm_pass=True`) trades latency for recall — recommended for meeting transcripts (higher PII density) and optional for Slack technical chat (usually low PII density).
- **Failure mode**: if redaction throws (including a failed optional LLM pass), `IngestionPipeline._redact_or_drop` logs the error and drops the message. Nothing is written to Qdrant or Neo4j. The optional LLM pass raises `RedactionError` instead of returning the original text.

## 2. Workspace & Channel Access Control

- **Workspace isolation**: every stored object (Qdrant payload, Neo4j node, Postgres row) carries `workspace_id`. Every read query filters on it as a hard `must` condition, not a post-filter — cross-tenant leakage is structurally prevented, not just discouraged.
- **Channel-level ACL**: `channel_acl` table (Postgres) mirrors Slack/Teams channel membership. Membership sync from Slack/Teams events is still an operator job (the table is the source of truth). At retrieval time, `PostgresStore.get_allowed_channels()` resolves the caller's channels. When several people are on a voice call, `HybridRetriever` intersects those sets and passes the result into `VectorStore.search()` as a hard filter. A lookup error or an empty intersection returns no chunks.
- **Voice calls**: the LiveKit agent's `workspace_id` and channel scope should come from room metadata set when your backend dispatches the agent to a call — never inferred from voice content. If per-participant channel scoping isn't resolvable for a voice call (common — a call may span people with different channel access), default to the **most restrictive shared scope**, not the broadest.

## 3. Audit Logging

- Every retrieval writes a row to `retrieval_audit_log` from `HybridRetriever.retrieve()`: query text, resolved channel scope, and which chunk_ids were returned. The voice tool passes `source_interface=voice`. Set `AUDIT_FAIL_CLOSED=true` to withhold results when that insert fails. With the flag off, a down database is logged and the answer is still returned, so local development works without Postgres.

## 4. Meeting Consent & Disclosure

- The meeting bot announces itself on every join, in chat (`chat.on_bot_join` and `chat.on_participant_join`) and as the voice agent's opening line. The sentence is fixed: `"AI Knowledge Agent joined and is recording for knowledge capture"`.
- `meeting_consent_log` records the announcement and any objection. `MEETING_ON_OBJECTION` is `stop` (remove the bot via Recall `leave_call`) or `exclude_speaker` (drop that speaker's segments and keep the meeting). `MEETING_JURISDICTION` selects the rule:
  - `one_party` — segments may be stored once the bot is running; an objection still follows `MEETING_ON_OBJECTION`.
  - `two_party` (default) — nothing is stored until the disclosure is recorded.
  - `all_party` — same as two-party, and any objection stops the whole meeting.
- Counsel still has to pick the jurisdiction and the objection policy for each workspace. The default is the stricter pair (`two_party` + `stop`) so an unset environment does not record first and ask later.

## 5. Secrets & Credential Handling

- All API keys (Slack signing secret, Recall.ai key, Deepgram, Cartesia, NVIDIA, Qdrant, Neo4j, LiveKit) are read from environment variables — never hardcoded. See `.env.example`.
- Webhook signature verification is implemented for Slack (`slack_listener.py`), the Teams gateway (`teams_listener.py`, same HMAC construction), and Recall.ai (`security/webhook_signature.py`). Recall.ai uses Standard Webhooks: HMAC-SHA256 over `{id}.{timestamp}.{raw body}` with the base64 key after `whsec_`. Both `webhook-*` and legacy `svix-*` headers are accepted, and timestamps older than five minutes are rejected. A missing secret rejects the request.

## 6. Production controls

- [x] Fail-closed error handling in the redaction step (`IngestionPipeline._redact_or_drop`)
- [x] Real Recall.ai webhook signature verification (`verify_recall_signature`)
- [x] Audit log write-through in `HybridRetriever.retrieve()` (`AUDIT_FAIL_CLOSED` withholds results)
- [x] Jurisdiction-aware meeting consent logic (`security/consent.py`, `MEETING_JURISDICTION`)
- [x] Rate limiting on the Slack, Teams, and Recall.ai webhook endpoints (`WEBHOOK_RATE_LIMIT`, per process)
- [x] Encryption at rest: Postgres community and Neo4j community do not encrypt data files. Production startup (`APP_ENV=production`) refuses to run until `ENCRYPTION_AT_REST` is `disk` or `managed`. Local `docker-compose.yml` volumes still need an encrypted disk (BitLocker or the cloud volume). This is the infrastructure control; the app only attests it.
- [x] Data retention: `CHUNK_TTL_DAYS` (Slack/Teams, default 365), `MEETING_TTL_DAYS` (transcripts and Meeting nodes, default 90), `AUDIT_TTL_DAYS` (audit and consent rows, default 730). Run `python -m ingestion.retention` on a schedule.

Still required before a live enterprise rollout, outside this repository:

- Confirm `MEETING_JURISDICTION` with counsel for each workspace.
- Sync `channel_acl` from Slack/Teams membership events.
- Put Qdrant, Postgres, and Neo4j on encrypted volumes or managed TDE, then set `ENCRYPTION_AT_REST`.
- Replace the NVIDIA free tier with a dedicated endpoint.
- Run the retention job on a scheduler (cron or the platform equivalent). The limiters are in-process; put a shared gateway in front if you run more than one worker.
