-- =============================================================
-- Postgres schema: workspace scoping, channel ACL, audit log
-- =============================================================

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id    TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    platform        TEXT NOT NULL CHECK (platform IN ('slack', 'teams')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS channel_acl (
    id              BIGSERIAL PRIMARY KEY,
    workspace_id    TEXT NOT NULL REFERENCES workspaces(workspace_id),
    channel_id      TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('member', 'admin')),
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, channel_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_channel_acl_lookup
    ON channel_acl (workspace_id, user_id);

-- Retrieval-time membership check: given a workspace + user, get every
-- channel_id they're allowed to see. The vector store query filters on
-- this result set before ranking — never after.

-- Audit log: every retrieval (text or voice) is logged for governance review.
CREATE TABLE IF NOT EXISTS retrieval_audit_log (
    id              BIGSERIAL PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    user_id         TEXT,
    query_text      TEXT NOT NULL,
    channel_scope   TEXT[],
    result_chunk_ids TEXT[],
    source_interface TEXT NOT NULL CHECK (source_interface IN ('slack', 'teams', 'voice')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_workspace_time
    ON retrieval_audit_log (workspace_id, created_at DESC);

-- Meeting consent/disclosure log — records that the bot announced itself
-- and whether any participant objected (governance requirement, see
-- security/governance.md).
CREATE TABLE IF NOT EXISTS meeting_consent_log (
    id              BIGSERIAL PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    meeting_id      TEXT NOT NULL,
    announced_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    objection_raised BOOLEAN NOT NULL DEFAULT false,
    recording_stopped_at TIMESTAMPTZ
);
