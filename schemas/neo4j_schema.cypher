// =============================================================
// Neo4j Knowledge Graph Schema
// Enterprise Omnichannel AI Knowledge & Voice Agent
// =============================================================
// Node labels:
//   :Project, :Issue, :DeploySctipt(sic-fixed below), :Person, :Decision,
//   :Document, :Meeting, :Workspace
//
// Every node carries `workspace_id` for tenant isolation — always filter
// on it in application queries, constraints alone do not enforce isolation.
// =============================================================

// ---- Constraints (uniqueness + existence) ----
CREATE CONSTRAINT project_id IF NOT EXISTS
FOR (p:Project) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT issue_id IF NOT EXISTS
FOR (i:Issue) REQUIRE i.id IS UNIQUE;

CREATE CONSTRAINT deployscript_id IF NOT EXISTS
FOR (d:DeployScript) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT person_id IF NOT EXISTS
FOR (p:Person) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT decision_id IF NOT EXISTS
FOR (d:Decision) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (doc:Document) REQUIRE doc.id IS UNIQUE;

CREATE CONSTRAINT meeting_id IF NOT EXISTS
FOR (m:Meeting) REQUIRE m.id IS UNIQUE;

CREATE CONSTRAINT workspace_id IF NOT EXISTS
FOR (w:Workspace) REQUIRE w.id IS UNIQUE;

// ---- Indexes for lookup performance ----
CREATE INDEX project_workspace IF NOT EXISTS FOR (p:Project) ON (p.workspace_id);
CREATE INDEX issue_workspace IF NOT EXISTS FOR (i:Issue) ON (i.workspace_id);
CREATE INDEX issue_status IF NOT EXISTS FOR (i:Issue) ON (i.status);
CREATE INDEX person_email IF NOT EXISTS FOR (p:Person) ON (p.email);

// ---- Node property contracts (documented, not enforced by Neo4j) ----
// :Project     { id, workspace_id, name, description, created_at }
// :Issue       { id, workspace_id, title, description, status, severity,
//                first_seen_at, resolved_at, source_chunk_id }
// :DeployScript{ id, workspace_id, name, repo_path, description }
// :Person      { id, workspace_id, email, display_name, role }
// :Decision    { id, workspace_id, summary, rationale, decided_at, source_chunk_id }
// :Document    { id, workspace_id, title, url, doc_type }
// :Meeting     { id, workspace_id, title, started_at, ended_at, platform }

// ---- Relationship types ----
// (:Project)-[:HAS_ISSUE]->(:Issue)
// (:Issue)-[:RESOLVED_BY]->(:DeployScript)
// (:Issue)-[:RESOLVED_BY]->(:Decision)
// (:Person)-[:REPORTED]->(:Issue)
// (:Person)-[:OWNS]->(:Project)
// (:Person)-[:MADE_DECISION]->(:Decision)
// (:Decision)-[:AFFECTS]->(:Project)
// (:Meeting)-[:DISCUSSED]->(:Issue)
// (:Meeting)-[:DISCUSSED]->(:Decision)
// (:Document)-[:REFERENCED_IN]->(:Issue)
// (:Project)-[:BELONGS_TO]->(:Workspace)

// =============================================================
// Example write pattern — MERGE, never blind CREATE, to avoid duplicates
// when the same entity is mentioned across multiple messages.
// =============================================================

// Example: an issue gets reported and later resolved by a deploy script,
// as extracted from two different Slack messages.

MERGE (w:Workspace {id: $workspace_id})

MERGE (proj:Project {id: $project_id})
  ON CREATE SET proj.name = $project_name,
                proj.workspace_id = $workspace_id,
                proj.created_at = datetime()
MERGE (proj)-[:BELONGS_TO]->(w)

MERGE (issue:Issue {id: $issue_id})
  ON CREATE SET issue.title = $issue_title,
                issue.workspace_id = $workspace_id,
                issue.status = 'open',
                issue.severity = $severity,
                issue.first_seen_at = datetime(),
                issue.source_chunk_id = $source_chunk_id
MERGE (proj)-[:HAS_ISSUE]->(issue)

MERGE (reporter:Person {id: $reporter_id})
  ON CREATE SET reporter.display_name = $reporter_name,
                reporter.workspace_id = $workspace_id
MERGE (reporter)-[:REPORTED]->(issue)

// Later, when a resolution is detected in a follow-up message:
MERGE (script:DeployScript {id: $script_id})
  ON CREATE SET script.name = $script_name,
                script.workspace_id = $workspace_id,
                script.repo_path = $repo_path
MERGE (issue)-[:RESOLVED_BY]->(script)
SET issue.status = 'resolved', issue.resolved_at = datetime();

// =============================================================
// Example traversal — the query the voice agent's tool calls at runtime.
// Given an issue mention, find how it was resolved and by whom.
// =============================================================

MATCH (issue:Issue {workspace_id: $workspace_id})
WHERE toLower(issue.title) CONTAINS toLower($search_term)
OPTIONAL MATCH (issue)-[:RESOLVED_BY]->(resolution)
OPTIONAL MATCH (reporter:Person)-[:REPORTED]->(issue)
OPTIONAL MATCH (proj:Project)-[:HAS_ISSUE]->(issue)
RETURN issue.title AS issue_title,
       issue.status AS status,
       issue.severity AS severity,
       reporter.display_name AS reported_by,
       proj.name AS project,
       labels(resolution) AS resolution_type,
       coalesce(resolution.name, resolution.summary) AS resolution_detail
LIMIT 10;
