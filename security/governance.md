# Security & Governance Strategy

## 1. PII Redaction

- **Where**: applied in `security/pii_redaction.py`, invoked inside `IngestionPipeline` before any text reaches embedding, entity extraction, or storage. Unredacted text is never persisted — only a SHA-256 hash of the original is kept, for audit correlation, not recovery.
- **What's caught by regex (fast path, always on)**: emails, phone numbers, SSNs, credit-card-like sequences, IP addresses, common API key/token formats (AWS, GitHub, Stripe-style, Slack, generic Bearer tokens).
- **What regex misses**: full names in prose, physical addresses, indirect identifiers. Optional LLM-based secondary pass (`use_llm_pass=True`) trades latency for recall — recommended for meeting transcripts (higher PII density) and optional for Slack technical chat (usually low PII density).
- **Failure mode**: if redaction throws, the pipeline should fail closed — drop the message rather than store it unredacted. (Not yet wired as fail-closed in the scaffold; add a try/except around `redactor.redact()` in `pipeline.py` before production.)

## 2. Workspace & Channel Access Control

- **Workspace isolation**: every stored object (Qdrant payload, Neo4j node, Postgres row) carries `workspace_id`. Every read query filters on it as a hard `must` condition, not a post-filter — cross-tenant leakage is structurally prevented, not just discouraged.
- **Channel-level ACL**: `channel_acl` table (Postgres) mirrors Slack/Teams channel membership, kept in sync via periodic membership polling or real-time `member_joined_channel` / `member_left_channel` events. At retrieval time, the caller's `user_id` resolves to `allowed_channel_ids` via `ACLStore.get_allowed_channels()`, which is passed into `VectorStore.search()` as an additional filter.
- **Voice calls**: the LiveKit agent's `workspace_id` and channel scope should come from room metadata set when your backend dispatches the agent to a call — never inferred from voice content. If per-participant channel scoping isn't resolvable for a voice call (common — a call may span people with different channel access), default to the **most restrictive shared scope**, not the broadest.

## 3. Audit Logging

- Every retrieval (Slack reply, Teams reply, or voice KB tool call) writes a row to `retrieval_audit_log`: query text, resolved channel scope, and which chunk_ids were returned. This is for governance review (e.g. "did the agent ever surface content from a restricted channel to the wrong person") and is not currently wired into the scaffold's `HybridRetriever` — add a write-through call in `retrieve()` before production.

## 4. Meeting Consent & Disclosure

- The meeting bot must announce its presence on joining every call (`"AI Knowledge Agent joined and is recording for knowledge capture"` — implement as a bot chat message or the LiveKit agent's opening spoken greeting, both are stubbed as TODOs in `agent.py` / `meeting_listener.py`).
- `meeting_consent_log` records the announcement and any objection. Define your organization's policy for what happens on objection (stop recording entirely vs. exclude that participant's audio) — this is a legal/HR policy decision, not a technical one, and varies by jurisdiction (many US states require two-party consent for recording).
- **This scaffold does not implement jurisdiction-aware consent logic.** Treat this as a hard requirement to resolve with legal counsel before deploying to any real meeting.

## 5. Secrets & Credential Handling

- All API keys (Slack signing secret, Recall.ai key, Deepgram, Cartesia, NVIDIA, Qdrant, Neo4j, LiveKit) are read from environment variables — never hardcoded. See `.env.example`.
- Webhook signature verification is implemented for Slack (`slack_listener.py`) and stubbed for Recall.ai (`meeting_listener.py`) — the Recall.ai stub currently returns `True` unconditionally when a secret is configured, which is a placeholder, not real verification. **Do not deploy without implementing real signature verification against Recall.ai's current signing scheme.**

## 6. Known gaps to close before production

- [ ] Fail-closed error handling in the redaction step
- [ ] Real Recall.ai webhook signature verification
- [ ] Audit log write-through in `HybridRetriever.retrieve()`
- [ ] Jurisdiction-aware meeting consent logic
- [ ] Rate limiting on the Slack/Teams webhook endpoints
- [ ] Encryption at rest for Postgres (raw_text_hash + ACL tables) and Neo4j — infra-level (managed DB encryption), not application-level
- [ ] Data retention policy: define TTL for vector chunks, meeting transcripts, and audit logs; implement scheduled purge jobs
