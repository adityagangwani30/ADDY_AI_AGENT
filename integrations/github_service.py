from __future__ import annotations

import base64
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable

import requests

from memory.storage import SQLiteMemoryRepository

LOGGER = logging.getLogger("integrations.github_service")

GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_USER_AGENT = os.getenv("GITHUB_USER_AGENT", "personal-ai-assistant")


@dataclass(frozen=True)
class RepositoryRef:
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


def _truncate(text: str, limit: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _split_lines(text: str, limit: int = 8) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


class GitHubService:
    def __init__(
        self,
        token: str | None = None,
        memory_repo: SQLiteMemoryRepository | None = None,
        base_url: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.token = (token or GITHUB_TOKEN or "").strip()
        self.base_url = (base_url or GITHUB_API_BASE).rstrip("/")
        self.session = session or requests.Session()
        self.memory_repo = memory_repo or SQLiteMemoryRepository()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": GITHUB_USER_AGENT,
            }
        )
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    def is_configured(self) -> bool:
        return bool(self.token)

    def parse_repository_ref(self, value: str) -> RepositoryRef | None:
        text = (value or "").strip()
        if not text:
            return None

        url_match = re.search(r"github\.com/([^/\s]+)/([^/#?\s]+)", text, flags=re.I)
        if url_match:
            return RepositoryRef(owner=url_match.group(1), name=url_match.group(2).removesuffix(".git"))

        slug_match = re.fullmatch(r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", text)
        if slug_match:
            return RepositoryRef(owner=slug_match.group(1), name=slug_match.group(2).removesuffix(".git"))

        return None

    def resolve_repository_reference(self, user_id: str, value: str) -> RepositoryRef | None:
        parsed = self.parse_repository_ref(value)
        if parsed:
            return parsed

        resolved = self.memory_repo.resolve_entity_alias(user_id, value)
        if resolved:
            return self.parse_repository_ref(resolved)

        active = self.memory_repo.get_user_preference(user_id, "github_active_repository")
        if isinstance(active, str):
            return self.parse_repository_ref(active)

        return None

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, timeout: int = 15) -> Any:
        url = f"{self.base_url}{path}"
        retries = 0
        last_error: Exception | None = None

        while retries < 3:
            started = time.perf_counter()
            response = self.session.request(method, url, params=params, timeout=timeout)
            latency_ms = int((time.perf_counter() - started) * 1000)
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            LOGGER.info(
                "github_request method=%s path=%s status=%s latency_ms=%s remaining=%s reset=%s",
                method,
                path,
                response.status_code,
                latency_ms,
                remaining,
                reset,
            )

            if response.status_code in {200, 201}:
                if response.text.strip():
                    return response.json()
                return None

            if response.status_code in {403, 429, 500, 502, 503, 504}:
                message = response.text.lower()
                if "rate limit" in message or response.status_code in {429, 403}:
                    retries += 1
                    sleep_seconds = min(2 ** retries, 8)
                    if reset and remaining == "0":
                        try:
                            sleep_seconds = max(1, int(reset) - int(time.time()))
                        except Exception:
                            pass
                    time.sleep(max(1, sleep_seconds))
                    continue

            try:
                response.raise_for_status()
            except Exception as exc:
                last_error = exc
                break

        if last_error is not None:
            raise RuntimeError(f"GitHub API request failed for {path}: {last_error}") from last_error
        raise RuntimeError(f"GitHub API request failed for {path}")

    def fetch_repositories(self, per_page: int = 10) -> dict[str, Any]:
        count = max(1, min(int(per_page), 100))
        data = self._request("GET", "/user/repos", params={"per_page": count, "sort": "updated", "direction": "desc"})
        repos = [self._normalize_repository(repo) for repo in (data or [])]
        return {"count": len(repos), "repositories": repos}

    def fetch_repository(self, owner: str, repo: str) -> dict[str, Any]:
        data = self._request("GET", f"/repos/{owner}/{repo}")
        return {"repository": self._normalize_repository(data)}

    def fetch_commits(self, owner: str, repo: str, per_page: int = 10) -> dict[str, Any]:
        count = max(1, min(int(per_page), 50))
        data = self._request("GET", f"/repos/{owner}/{repo}/commits", params={"per_page": count})
        commits = [self._normalize_commit_summary(item) for item in (data or [])]
        return {"count": len(commits), "commits": commits}

    def fetch_commit_details(self, owner: str, repo: str, sha: str) -> dict[str, Any]:
        data = self._request("GET", f"/repos/{owner}/{repo}/commits/{sha}")
        return self._normalize_commit_detail(data)

    def fetch_issues(self, owner: str, repo: str, state: str = "open", per_page: int = 10) -> dict[str, Any]:
        count = max(1, min(int(per_page), 50))
        data = self._request(
            "GET",
            f"/repos/{owner}/{repo}/issues",
            params={"state": state, "per_page": count, "sort": "updated", "direction": "desc"},
        )
        issues = [self._normalize_issue(item) for item in (data or []) if not item.get("pull_request")]
        return {"count": len(issues), "issues": issues}

    def fetch_pull_requests(self, owner: str, repo: str, state: str = "open", per_page: int = 10) -> dict[str, Any]:
        count = max(1, min(int(per_page), 50))
        data = self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls",
            params={"state": state, "per_page": count, "sort": "updated", "direction": "desc"},
        )
        pulls = [self._normalize_pull_request(item) for item in (data or [])]
        return {"count": len(pulls), "pull_requests": pulls}

    def fetch_readme(self, owner: str, repo: str) -> dict[str, Any]:
        data = self._request("GET", f"/repos/{owner}/{repo}/readme")
        content = ""
        if isinstance(data, dict):
            encoded = data.get("content") or ""
            if isinstance(encoded, str):
                try:
                    content = base64.b64decode(encoded.replace("\n", "")).decode("utf-8", errors="replace")
                except Exception:
                    content = ""
        return {"readme": content, "name": data.get("name") if isinstance(data, dict) else None}

    def fetch_repository_contents(self, owner: str, repo: str, path: str = "") -> dict[str, Any]:
        suffix = f"/{path.lstrip('/') }" if path else ""
        data = self._request("GET", f"/repos/{owner}/{repo}/contents{suffix}")
        items = data if isinstance(data, list) else [data] if data else []
        return {"contents": [self._normalize_content_item(item) for item in items]}

    def summarize_repository(self, owner: str, repo: str, user_id: str | None = None) -> dict[str, Any]:
        metadata = self.fetch_repository(owner, repo)["repository"]
        readme = self.fetch_readme(owner, repo)
        contents = self.fetch_repository_contents(owner, repo).get("contents", [])
        stack = self._detect_stack(metadata, readme.get("readme") or "", contents)
        structure = self._summarize_structure(contents)
        purpose = self._infer_purpose(metadata, readme.get("readme") or "")
        summary = self._build_repository_summary(metadata, purpose, stack, structure, readme.get("readme") or "")

        payload = {
            "repository": metadata,
            "summary": summary,
            "purpose": purpose,
            "tech_stack": stack,
            "structure": structure,
            "readme_excerpt": _truncate(readme.get("readme") or "", 600),
        }

        if user_id:
            self.memory_repo.set_active_repository(user_id, metadata["full_name"])
            self.memory_repo.save_memory_entry(
                user_id,
                "project",
                metadata["full_name"],
                payload,
                importance=4,
            )

        return payload

    def summarize_commits(self, owner: str, repo: str, per_page: int = 5, user_id: str | None = None) -> dict[str, Any]:
        commits = self.fetch_commits(owner, repo, per_page=per_page).get("commits", [])
        details = []
        for commit in commits[: min(len(commits), 3)]:
            sha = str(commit.get("sha") or "")
            if not sha:
                continue
            try:
                details.append(self.fetch_commit_details(owner, repo, sha))
            except Exception:
                details.append(commit)

        summary = self._build_commit_summary(commits, details)
        payload = {
            "repository": f"{owner}/{repo}",
            "summary": summary,
            "commits": commits,
            "details": details,
        }
        if user_id:
            self.memory_repo.save_memory_entry(user_id, "project_activity", f"{owner}/{repo}:commits", payload, importance=3)
        return payload

    def summarize_issues(self, owner: str, repo: str, state: str = "open", per_page: int = 10, user_id: str | None = None) -> dict[str, Any]:
        issues = self.fetch_issues(owner, repo, state=state, per_page=per_page).get("issues", [])
        stale = [issue for issue in issues if self._is_stale(issue)]
        blockers = [issue for issue in issues if self._is_blocker(issue)]
        summary = self._build_issue_summary(issues, blockers, stale)
        payload = {
            "repository": f"{owner}/{repo}",
            "summary": summary,
            "issues": issues,
            "blockers": blockers,
            "stale": stale,
        }
        if user_id:
            self.memory_repo.save_memory_entry(user_id, "project_activity", f"{owner}/{repo}:issues", payload, importance=3)
        return payload

    def summarize_pull_requests(self, owner: str, repo: str, state: str = "open", per_page: int = 10, user_id: str | None = None) -> dict[str, Any]:
        pulls = self.fetch_pull_requests(owner, repo, state=state, per_page=per_page).get("pull_requests", [])
        blockers = [pull for pull in pulls if self._is_pr_blocked(pull)]
        summary = self._build_pr_summary(pulls, blockers)
        payload = {
            "repository": f"{owner}/{repo}",
            "summary": summary,
            "pull_requests": pulls,
            "blockers": blockers,
        }
        if user_id:
            self.memory_repo.save_memory_entry(user_id, "project_activity", f"{owner}/{repo}:prs", payload, importance=3)
        return payload

    def generate_changelog(self, owner: str, repo: str, per_page: int = 10, user_id: str | None = None) -> dict[str, Any]:
        commits = self.fetch_commits(owner, repo, per_page=per_page).get("commits", [])
        details = []
        for commit in commits[: min(5, len(commits))]:
            sha = str(commit.get("sha") or "")
            if sha:
                try:
                    details.append(self.fetch_commit_details(owner, repo, sha))
                except Exception:
                    details.append(commit)
        changelog = self._build_changelog(commits, details)
        payload = {
            "repository": f"{owner}/{repo}",
            "changelog": changelog,
            "commits": commits,
        }
        if user_id:
            self.memory_repo.save_memory_entry(user_id, "project_activity", f"{owner}/{repo}:changelog", payload, importance=2)
        return payload

    def draft_commit_message(self, changes: str, repo_name: str | None = None) -> dict[str, Any]:
        text = _truncate(changes, 400)
        if re.search(r"fix|bug|error|traceback", text, flags=re.I):
            category = "fix"
            subject = f"fix: {self._first_meaningful_phrase(text)}"
        elif re.search(r"add|implement|create|introduce", text, flags=re.I):
            category = "feat"
            subject = f"feat: {self._first_meaningful_phrase(text)}"
        elif re.search(r"refactor|cleanup|restructure", text, flags=re.I):
            category = "refactor"
            subject = f"refactor: {self._first_meaningful_phrase(text)}"
        else:
            category = "chore"
            subject = f"chore: {self._first_meaningful_phrase(text)}"
        if repo_name:
            subject = f"{subject} ({repo_name})"
        body = self._build_commit_body(text)
        return {"category": category, "subject": subject[:72], "body": body}

    def summarize_code_snippet(self, code: str, language: str | None = None) -> dict[str, Any]:
        lines = _split_lines(code, limit=80)
        functions = [line.strip() for line in lines if re.match(r"^(async\s+def|def|class)\s+", line.strip())]
        imports = [line.strip() for line in lines if re.match(r"^(from\s+\S+\s+import\s+|import\s+)", line.strip())]
        snippet = _truncate(code, 500)
        summary = self._build_code_summary(language, functions, imports, snippet)
        return {
            "language": language or self.detect_language(code),
            "summary": summary,
            "functions": functions[:10],
            "imports": imports[:10],
            "preview": snippet,
        }

    def detect_language(self, text: str) -> str:
        if re.search(r"^\s*def\s+\w+\(|^\s*class\s+\w+", text, flags=re.M):
            return "python"
        if re.search(r"^\s*function\s+\w+\(|=>\s*\{", text, flags=re.M):
            return "javascript"
        if re.search(r"^\s*public\s+class\s+\w+|^\s*import\s+java\.", text, flags=re.M):
            return "java"
        if re.search(r"^\s*package\s+main|func\s+\w+\(", text, flags=re.M):
            return "go"
        if re.search(r"^\s*<\w+[^>]*>", text, flags=re.M):
            return "html"
        return "text"

    def explain_traceback(self, traceback_text: str) -> dict[str, Any]:
        lines = _split_lines(traceback_text, limit=80)
        error_line = next((line for line in reversed(lines) if re.search(r"Error|Exception", line)), "")
        file_line = next((line for line in lines if "File " in line and ", line " in line), "")
        probable_cause = self._infer_traceback_cause(traceback_text)
        return {
            "summary": probable_cause,
            "error_line": error_line,
            "location": file_line,
            "preview": _truncate(traceback_text, 600),
        }

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, timeout: int = 15) -> Any:
        url = f"{self.base_url}{path}"
        retries = 0
        last_error: Exception | None = None

        while retries < 3:
            started = time.perf_counter()
            response = self.session.request(method, url, params=params, timeout=timeout)
            latency_ms = int((time.perf_counter() - started) * 1000)
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            LOGGER.info(
                "github_request method=%s path=%s status=%s latency_ms=%s remaining=%s reset=%s",
                method,
                path,
                response.status_code,
                latency_ms,
                remaining,
                reset,
            )

            if response.status_code in {200, 201}:
                if response.text.strip():
                    return response.json()
                return None

            if response.status_code in {403, 429, 500, 502, 503, 504}:
                message = response.text.lower()
                if "rate limit" in message or response.status_code in {429, 403}:
                    retries += 1
                    sleep_seconds = min(2 ** retries, 8)
                    if reset and remaining == "0":
                        try:
                            sleep_seconds = max(1, int(reset) - int(time.time()))
                        except Exception:
                            pass
                    time.sleep(max(1, sleep_seconds))
                    continue

            try:
                response.raise_for_status()
            except Exception as exc:
                last_error = exc
                break

        if last_error is not None:
            raise RuntimeError(f"GitHub API request failed for {path}: {last_error}") from last_error
        raise RuntimeError(f"GitHub API request failed for {path}")

    def _normalize_repository(self, repo: dict[str, Any] | None) -> dict[str, Any]:
        repo = repo or {}
        owner = repo.get("owner") or {}
        return {
            "id": repo.get("id"),
            "name": repo.get("name"),
            "full_name": repo.get("full_name") or f"{owner.get('login', '')}/{repo.get('name', '')}".strip("/"),
            "description": repo.get("description") or "",
            "html_url": repo.get("html_url"),
            "default_branch": repo.get("default_branch"),
            "language": repo.get("language"),
            "topics": repo.get("topics") or [],
            "private": repo.get("private", False),
            "fork": repo.get("fork", False),
            "archived": repo.get("archived", False),
            "updated_at": repo.get("updated_at"),
            "pushed_at": repo.get("pushed_at"),
            "open_issues_count": repo.get("open_issues_count"),
            "stargazers_count": repo.get("stargazers_count"),
            "forks_count": repo.get("forks_count"),
            "owner": owner.get("login"),
        }

    def _normalize_commit_summary(self, item: dict[str, Any] | None) -> dict[str, Any]:
        item = item or {}
        commit = item.get("commit") or {}
        author = commit.get("author") or {}
        return {
            "sha": item.get("sha"),
            "message": _truncate(commit.get("message") or "", 140),
            "author": author.get("name") or (item.get("author") or {}).get("login") or "unknown",
            "date": author.get("date"),
            "url": item.get("html_url"),
        }

    def _normalize_commit_detail(self, item: dict[str, Any] | None) -> dict[str, Any]:
        item = item or {}
        commit = item.get("commit") or {}
        files = item.get("files") or []
        return {
            "sha": item.get("sha"),
            "message": commit.get("message") or "",
            "author": (commit.get("author") or {}).get("name") or "unknown",
            "date": (commit.get("author") or {}).get("date"),
            "additions": item.get("additions", 0),
            "deletions": item.get("deletions", 0),
            "changes": item.get("changes", 0),
            "files": [self._normalize_content_item(file_item) for file_item in files],
        }

    def _normalize_issue(self, item: dict[str, Any] | None) -> dict[str, Any]:
        item = item or {}
        return {
            "number": item.get("number"),
            "title": item.get("title"),
            "state": item.get("state"),
            "labels": [label.get("name") for label in item.get("labels", []) if isinstance(label, dict)],
            "updated_at": item.get("updated_at"),
            "created_at": item.get("created_at"),
            "comments": item.get("comments", 0),
            "html_url": item.get("html_url"),
            "body": _truncate(item.get("body") or "", 240),
            "assignee": (item.get("assignee") or {}).get("login"),
        }

    def _normalize_pull_request(self, item: dict[str, Any] | None) -> dict[str, Any]:
        item = item or {}
        return {
            "number": item.get("number"),
            "title": item.get("title"),
            "state": item.get("state"),
            "draft": item.get("draft", False),
            "merged_at": item.get("merged_at"),
            "updated_at": item.get("updated_at"),
            "created_at": item.get("created_at"),
            "comments": item.get("comments", 0),
            "review_comments": item.get("review_comments", 0),
            "commits": item.get("commits", 0),
            "additions": item.get("additions", 0),
            "deletions": item.get("deletions", 0),
            "changed_files": item.get("changed_files", 0),
            "html_url": item.get("html_url"),
            "body": _truncate(item.get("body") or "", 240),
        }

    def _normalize_content_item(self, item: dict[str, Any] | None) -> dict[str, Any]:
        item = item or {}
        return {
            "name": item.get("name"),
            "path": item.get("path"),
            "type": item.get("type"),
            "size": item.get("size", 0),
            "download_url": item.get("download_url"),
            "html_url": item.get("html_url"),
            "language_hint": self._guess_language_from_name(item.get("name") or ""),
        }

    def _guess_language_from_name(self, filename: str) -> str:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mapping = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "jsx": "javascript",
            "go": "go",
            "rs": "rust",
            "java": "java",
            "rb": "ruby",
            "php": "php",
            "sh": "shell",
            "md": "markdown",
            "json": "json",
            "yml": "yaml",
            "yaml": "yaml",
            "toml": "toml",
            "cs": "csharp",
        }
        return mapping.get(ext, "text")

    def _detect_stack(self, metadata: dict[str, Any], readme_text: str, contents: list[dict[str, Any]]) -> list[str]:
        stack: list[str] = []
        seen: set[str] = set()

        def add(label: str) -> None:
            key = label.strip().lower()
            if not key or key in seen:
                return
            seen.add(key)
            stack.append(label)

        language = metadata.get("language")
        if isinstance(language, str) and language:
            add(language)

        readme_lower = readme_text.lower()
        heuristics = [
            ("FastAPI", ["fastapi", "uvicorn", "pydantic"]),
            ("Python", ["python", "pip", "pytest", "requirements.txt"]),
            ("Node.js", ["node", "package.json", "npm", "yarn"]),
            ("TypeScript", ["typescript", "tsconfig", "tsx"]),
            ("React", ["react", "next.js", "vite", "jsx"]),
            ("Docker", ["docker", "dockerfile"]),
            ("SQLite", ["sqlite", "sqlite3"]),
            ("Google APIs", ["google workspace", "gmail", "calendar", "drive"]),
        ]
        root_names = {item.get("name", "").lower() for item in contents}
        for label, needles in heuristics:
            if any(needle in readme_lower or any(needle in name for name in root_names) for needle in needles):
                add(label)

        topics = metadata.get("topics") or []
        for topic in topics:
            if isinstance(topic, str) and topic:
                add(topic)

        return stack[:8]

    def _summarize_structure(self, contents: list[dict[str, Any]]) -> dict[str, Any]:
        directories = [item["name"] for item in contents if item.get("type") == "dir" and item.get("name")]
        files = [item["name"] for item in contents if item.get("type") == "file" and item.get("name")]
        notable_files = [name for name in files if name.lower() in {"readme.md", "requirements.txt", "package.json", "pyproject.toml", "dockerfile", "main.py"}]
        return {
            "directories": directories[:12],
            "files": files[:12],
            "notable_files": notable_files[:8],
        }

    def _infer_purpose(self, metadata: dict[str, Any], readme_text: str) -> str:
        description = metadata.get("description") or ""
        first_para = " ".join(_split_lines(readme_text, limit=4))
        source = description or first_para
        if not source:
            return "Project purpose not described in fetched metadata."
        return _truncate(source, 260)

    def _build_repository_summary(self, metadata: dict[str, Any], purpose: str, stack: list[str], structure: dict[str, Any], readme_text: str) -> str:
        parts = [f"{metadata.get('full_name')}", purpose]
        if stack:
            parts.append("Tech stack: " + ", ".join(stack))
        if structure.get("directories"):
            parts.append("Structure: " + ", ".join(structure["directories"][:6]))
        if readme_text:
            parts.append("README: " + _truncate(readme_text, 180))
        return "\n".join(parts)

    def _build_commit_summary(self, commits: list[dict[str, Any]], details: list[dict[str, Any]]) -> str:
        if not commits:
            return "No recent commits found."
        authors = sorted({commit.get("author", "unknown") for commit in commits if commit.get("author")})
        messages = [commit.get("message", "") for commit in commits[:5]]
        file_paths = []
        for detail in details:
            for file_item in detail.get("files", []):
                path = file_item.get("path") or file_item.get("name")
                if path:
                    file_paths.append(str(path))
        risky = self._detect_risky_changes(file_paths, messages)
        lines = [f"{len(commits)} recent commits by {', '.join(authors[:4]) or 'unknown authors'}. "]
        if messages:
            lines.append("Recent messages: " + "; ".join(_truncate(msg, 70) for msg in messages[:4]))
        if risky:
            lines.append("Potentially sensitive areas: " + ", ".join(risky))
        return "\n".join(lines)

    def _build_issue_summary(self, issues: list[dict[str, Any]], blockers: list[dict[str, Any]], stale: list[dict[str, Any]]) -> str:
        if not issues:
            return "No matching issues found."
        open_count = sum(1 for issue in issues if issue.get("state") == "open")
        parts = [f"{len(issues)} issues fetched, {open_count} open."]
        if blockers:
            parts.append("Blockers: " + "; ".join(_truncate(issue.get("title") or "", 80) for issue in blockers[:4]))
        if stale:
            parts.append("Stale: " + "; ".join(_truncate(issue.get("title") or "", 80) for issue in stale[:4]))
        return "\n".join(parts)

    def _build_pr_summary(self, pulls: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> str:
        if not pulls:
            return "No matching pull requests found."
        draft_count = sum(1 for pull in pulls if pull.get("draft"))
        merged_count = sum(1 for pull in pulls if pull.get("merged_at"))
        parts = [f"{len(pulls)} pull requests fetched, {draft_count} draft, {merged_count} merged."]
        if blockers:
            parts.append("Blocked or review-heavy: " + "; ".join(_truncate(pull.get("title") or "", 80) for pull in blockers[:4]))
        return "\n".join(parts)

    def _build_changelog(self, commits: list[dict[str, Any]], details: list[dict[str, Any]]) -> str:
        if not commits:
            return "No commits to summarize."
        items = []
        for commit, detail in zip(commits[:5], details[:5], strict=False):
            message = _truncate(commit.get("message") or detail.get("message") or "", 90)
            additions = int(detail.get("additions") or 0)
            deletions = int(detail.get("deletions") or 0)
            files = detail.get("files") or []
            path_sample = ", ".join(str(item.get("path") or item.get("name") or "") for item in files[:3] if item)
            items.append(f"- {message} (+{additions}/-{deletions}) {path_sample}".strip())
        return "\n".join(items)

    def _build_code_summary(self, language: str | None, functions: list[str], imports: list[str], snippet: str) -> str:
        parts = []
        if language:
            parts.append(f"Detected language: {language}.")
        if functions:
            parts.append("Top symbols: " + "; ".join(functions[:5]))
        if imports:
            parts.append("Imports: " + "; ".join(imports[:5]))
        parts.append("Preview: " + snippet)
        return " ".join(parts)

    def _build_commit_body(self, text: str) -> str:
        bullet_points = [line for line in _split_lines(text, limit=12) if len(line) > 8]
        if not bullet_points:
            return ""
        return "\n".join(f"- {line[:100]}" for line in bullet_points[:5])

    def _first_meaningful_phrase(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9\s._-]", " ", text)
        words = [word for word in cleaned.split() if len(word) > 2]
        if not words:
            return "update repository"
        return " ".join(words[:6])[:50].strip()

    def _detect_risky_changes(self, file_paths: Iterable[str], messages: Iterable[str]) -> list[str]:
        risk_keywords = {
            "auth": ["auth", "oauth", "token", "secret"],
            "deployment": ["deploy", "infra", "docker", "render", "k8s"],
            "database": ["db", "database", "migration", "schema", "sqlite"],
            "api": ["api", "route", "endpoint", "webhook"],
        }
        risky = []
        joined = " ".join(list(file_paths) + list(messages)).lower()
        for label, keywords in risk_keywords.items():
            if any(keyword in joined for keyword in keywords):
                risky.append(label)
        return risky

    def _is_stale(self, issue: dict[str, Any]) -> bool:
        updated_at = issue.get("updated_at") or ""
        if not updated_at:
            return False
        try:
            updated = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        except Exception:
            return False
        return (datetime.now(timezone.utc) - updated).days >= 21 and issue.get("state") == "open"

    def _is_blocker(self, issue: dict[str, Any]) -> bool:
        labels = [str(label).lower() for label in issue.get("labels", [])]
        title = str(issue.get("title") or "").lower()
        body = str(issue.get("body") or "").lower()
        return any(label in {"blocker", "blocked", "critical", "p0", "priority-high"} for label in labels) or "block" in title or "block" in body

    def _is_pr_blocked(self, pull: dict[str, Any]) -> bool:
        if pull.get("draft"):
            return True
        body = str(pull.get("body") or "").lower()
        title = str(pull.get("title") or "").lower()
        return "wip" in title or "blocked" in title or "review" in body

    def _infer_traceback_cause(self, traceback_text: str) -> str:
        text = traceback_text.lower()
        if "module not found" in text or "importerror" in text:
            return "Import or dependency issue detected."
        if "typeerror" in text:
            return "A value was likely passed with the wrong type or missing argument."
        if "keyerror" in text:
            return "A dictionary lookup likely missed an expected key."
        if "attributeerror" in text:
            return "An object likely does not expose the expected attribute or method."
        if "indentationerror" in text or "syntaxerror" in text:
            return "There is a syntax or indentation problem in the code."
        return "Traceback suggests a runtime failure in the referenced code path."


@lru_cache(maxsize=1)
def get_github_service() -> GitHubService:
    return GitHubService()


__all__ = ["GitHubService", "RepositoryRef", "get_github_service"]
