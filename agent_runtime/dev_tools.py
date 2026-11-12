"""Read-only development tools every agent surface can call.

Paths stay inside AGENT_WORKSPACE. Secret files are refused. Git commands
are argument lists, never a shell. Nothing here commits, pushes, or edits an issue.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
_SKIP_NAMES = {".env", ".env.local", ".env.production", "credentials.json", "id_rsa", "id_ed25519"}
_SKIP_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
_ISSUE_KEY = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./^{}~@-]{0,80}$")


def workspace_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    raw = os.environ.get("AGENT_WORKSPACE", "").strip()
    return Path(raw).resolve() if raw else Path.cwd().resolve()


def resolve_inside(root: Path, relative: str) -> Path:
    if not relative or relative.strip() in (".", ""):
        raise PermissionError("Path is required.")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError("Path is outside the workspace.")
    return candidate


def _blocked(path: Path) -> bool:
    if path.name in _SKIP_NAMES or path.suffix.lower() in _SKIP_SUFFIXES:
        return True
    return any(part in _SKIP_NAMES for part in path.parts)


def read_repository_file(relative: str, root: Path, max_lines: int = 200) -> str:
    try:
        path = resolve_inside(root, relative)
    except PermissionError as exc:
        return str(exc)
    if _blocked(path):
        return "That file is not available to the agent."
    if not path.is_file():
        return "File not found."
    limit = max(1, min(int(max_lines or 200), 400))
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Could not read the file: {exc}"
    body = "\n".join(lines[:limit])
    if len(lines) > limit:
        body += f"\n... {len(lines) - limit} more lines"
    return body or "(empty file)"


def search_repository(query: str, root: Path, limit: int = 20) -> str:
    needle = (query or "").strip()
    if len(needle) < 2:
        return "Provide a search query of at least 2 characters."
    cap = max(1, min(int(limit or 20), 30))
    hits: list[str] = []
    lowered = needle.lower()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS and not name.startswith(".")]
        for name in filenames:
            if name in _SKIP_NAMES or Path(name).suffix.lower() in _SKIP_SUFFIXES:
                continue
            path = Path(dirpath) / name
            if path.stat().st_size > 200_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if lowered not in text.lower():
                continue
            rel = path.relative_to(root).as_posix()
            for index, line in enumerate(text.splitlines(), start=1):
                if lowered in line.lower():
                    hits.append(f"{rel}:{index}: {line.strip()[:200]}")
                    if len(hits) >= cap:
                        return "\n".join(hits)
                    break
    return "\n".join(hits) if hits else "No matches."


def list_repository(root: Path, path: str = ".", limit: int = 80) -> str:
    relative = path or "."
    try:
        directory = root if relative.strip() in (".", "") else resolve_inside(root, relative)
    except PermissionError as exc:
        return str(exc)
    if _blocked(directory):
        return "That directory is not available to the agent."
    if not directory.is_dir():
        return "Directory not found."
    cap = max(1, min(int(limit or 80), 200))
    names: list[str] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if child.name in _SKIP_DIRS or child.name.startswith(".") or _blocked(child):
            continue
        names.append(child.name + ("/" if child.is_dir() else ""))
        if len(names) >= cap:
            break
    return "\n".join(names) if names else "(empty directory)"


def _git(root: Path, command: list[str], timeout: int = 10) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"git failed: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git failed").strip()
        return detail[:500]
    return completed.stdout.strip() or "(no output)"


def _optional_path(root: Path, path: str | None) -> str | None:
    if not path:
        return None
    resolved = resolve_inside(root, path)
    if _blocked(resolved):
        raise PermissionError("That path is not available to the agent.")
    return resolved.relative_to(root).as_posix()


def git_log(root: Path, limit: int = 10, path: str | None = None) -> str:
    count = max(1, min(int(limit or 10), 30))
    command = ["git", "log", f"-n{count}", "--pretty=format:%h %ad %s", "--date=short"]
    if path:
        try:
            resolved = resolve_inside(root, path)
        except PermissionError as exc:
            return str(exc)
        command.extend(["--", resolved.relative_to(root).as_posix()])
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"git log failed: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git log failed").strip()
        return detail[:500]
    return completed.stdout.strip() or "No commits."


def git_status(root: Path) -> str:
    return _git(root, ["git", "status", "--short", "--branch"])


def git_diff(root: Path, path: str | None = None, staged: bool = False) -> str:
    command = ["git", "diff", "--unified=1"]
    if staged:
        command.append("--cached")
    try:
        relative = _optional_path(root, path)
    except PermissionError as exc:
        return str(exc)
    if relative:
        command.extend(["--", relative])
    return _git(root, command)


def git_show(root: Path, revision: str) -> str:
    rev = (revision or "").strip()
    if ".." in rev or not _REVISION.fullmatch(rev):
        return "Revision is not valid."
    return _git(root, ["git", "show", "--stat", "--format=%h %ad %s", "--date=short", rev])


def run_tests(root: Path, path: str | None = None) -> str:
    """Run pytest. The model may name a file or directory inside the workspace and nothing else."""
    command = [sys.executable, "-m", "pytest", "-q", "--tb=line"]
    if path:
        try:
            relative = _optional_path(root, path)
        except PermissionError as exc:
            return str(exc)
        if relative is None:
            return "Path is required."
        target = root / relative
        if target.suffix.lower() not in {"", ".py"}:
            return "Tests must be a directory or a Python file."
        command.append(relative)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"pytest failed: {exc}"
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return output[:4000] or f"pytest exited {completed.returncode}"


def jira_get_issue(issue_key: str, *, client=None) -> str:
    key = (issue_key or "").strip().upper()
    if not _ISSUE_KEY.fullmatch(key):
        return "Issue key is not valid."
    base, email, token = _jira_auth()
    if not base:
        return "Jira is not configured."
    import httpx

    http = client or httpx.Client(timeout=20.0)
    owns = client is None
    try:
        response = http.get(f"{base}/rest/api/3/issue/{key}", auth=(email, token))
        response.raise_for_status()
        fields = response.json().get("fields") or {}
    except Exception as exc:  # noqa: BLE001
        return f"Jira request failed: {exc}"
    finally:
        if owns:
            http.close()
    summary = fields.get("summary") or ""
    status = (fields.get("status") or {}).get("name") or ""
    description = fields.get("description")
    from agent_runtime.addressing import adf_to_text

    detail = adf_to_text(description) if not isinstance(description, str) else description
    return f"{key} [{status}] {summary}\n{detail[:1500]}".strip()


def jira_search(jql: str, *, client=None, limit: int = 10) -> str:
    query = (jql or "").strip()
    if not query or len(query) > 500:
        return "Provide a JQL query under 500 characters."
    base, email, token = _jira_auth()
    if not base:
        return "Jira is not configured."
    import httpx

    cap = max(1, min(int(limit or 10), 20))
    http = client or httpx.Client(timeout=20.0)
    owns = client is None
    try:
        response = http.get(
            f"{base}/rest/api/3/search/jql",
            params={"jql": query, "maxResults": cap, "fields": "summary,status"},
            auth=(email, token),
        )
        response.raise_for_status()
        issues = response.json().get("issues") or []
    except Exception as exc:  # noqa: BLE001
        return f"Jira request failed: {exc}"
    finally:
        if owns:
            http.close()
    if not issues:
        return "No Jira issues matched."
    lines = []
    for issue in issues:
        fields = issue.get("fields") or {}
        status = (fields.get("status") or {}).get("name") or ""
        lines.append(f"{issue.get('key')} [{status}] {fields.get('summary') or ''}")
    return "\n".join(lines)


def _jira_auth() -> tuple[str, str, str]:
    base = os.environ.get("JIRA_BASE_URL", "").strip().rstrip("/")
    email = os.environ.get("JIRA_EMAIL", "").strip()
    token = os.environ.get("JIRA_API_TOKEN", "").strip()
    if not (base and email and token):
        return "", "", ""
    return base, email, token
