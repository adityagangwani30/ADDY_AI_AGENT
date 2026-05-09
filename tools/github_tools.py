from __future__ import annotations

from typing import Any

from integrations.github_service import GitHubService, get_github_service


def _service() -> GitHubService:
    return get_github_service()


def github_list_repositories(account: str, page_size: int = 10, session_id: str | None = None) -> dict[str, Any]:
    return _service().fetch_repositories(per_page=page_size)


def github_repository_summary(account: str, repository: str = "", owner: str = "", name: str = "", session_id: str | None = None) -> dict[str, Any]:
    service = _service()
    repo_ref = service.resolve_repository_reference(session_id or account or "", repository or f"{owner}/{name}")
    if not repo_ref:
        raise ValueError("repository is required")
    return service.summarize_repository(repo_ref.owner, repo_ref.name, user_id=session_id or account)


def github_commits(account: str, repository: str = "", owner: str = "", name: str = "", per_page: int = 10, session_id: str | None = None) -> dict[str, Any]:
    service = _service()
    repo_ref = service.resolve_repository_reference(session_id or account or "", repository or f"{owner}/{name}")
    if not repo_ref:
        raise ValueError("repository is required")
    return service.summarize_commits(repo_ref.owner, repo_ref.name, per_page=per_page, user_id=session_id or account)


def github_issues(account: str, repository: str = "", owner: str = "", name: str = "", state: str = "open", per_page: int = 10, session_id: str | None = None) -> dict[str, Any]:
    service = _service()
    repo_ref = service.resolve_repository_reference(session_id or account or "", repository or f"{owner}/{name}")
    if not repo_ref:
        raise ValueError("repository is required")
    return service.summarize_issues(repo_ref.owner, repo_ref.name, state=state, per_page=per_page, user_id=session_id or account)


def github_pull_requests(account: str, repository: str = "", owner: str = "", name: str = "", state: str = "open", per_page: int = 10, session_id: str | None = None) -> dict[str, Any]:
    service = _service()
    repo_ref = service.resolve_repository_reference(session_id or account or "", repository or f"{owner}/{name}")
    if not repo_ref:
        raise ValueError("repository is required")
    return service.summarize_pull_requests(repo_ref.owner, repo_ref.name, state=state, per_page=per_page, user_id=session_id or account)


def github_changelog(account: str, repository: str = "", owner: str = "", name: str = "", per_page: int = 10, session_id: str | None = None) -> dict[str, Any]:
    service = _service()
    repo_ref = service.resolve_repository_reference(session_id or account or "", repository or f"{owner}/{name}")
    if not repo_ref:
        raise ValueError("repository is required")
    return service.generate_changelog(repo_ref.owner, repo_ref.name, per_page=per_page, user_id=session_id or account)


def github_draft_commit_message(account: str, changes: str, repository: str = "", session_id: str | None = None) -> dict[str, Any]:
    service = _service()
    repo_ref = service.resolve_repository_reference(session_id or account or "", repository) if repository else None
    repo_name = repo_ref.full_name if repo_ref else None
    return service.draft_commit_message(changes, repo_name=repo_name)


def github_code_summary(account: str, code: str, language: str = "", session_id: str | None = None) -> dict[str, Any]:
    return _service().summarize_code_snippet(code, language=language or None)


def github_traceback_explain(account: str, traceback_text: str, session_id: str | None = None) -> dict[str, Any]:
    return _service().explain_traceback(traceback_text)


def github_project_dashboard(account: str, repository: str = "", owner: str = "", name: str = "", session_id: str | None = None) -> dict[str, Any]:
    service = _service()
    repo_ref = service.resolve_repository_reference(session_id or account or "", repository or f"{owner}/{name}")
    if not repo_ref:
        raise ValueError("repository is required")
    repo_summary = service.summarize_repository(repo_ref.owner, repo_ref.name, user_id=session_id or account)
    commits = service.summarize_commits(repo_ref.owner, repo_ref.name, per_page=5, user_id=session_id or account)
    issues = service.summarize_issues(repo_ref.owner, repo_ref.name, per_page=5, user_id=session_id or account)
    pulls = service.summarize_pull_requests(repo_ref.owner, repo_ref.name, per_page=5, user_id=session_id or account)
    return {
        "repository": repo_summary,
        "recent_commits": commits,
        "open_issues": issues,
        "open_pull_requests": pulls,
    }


__all__ = [
    "github_list_repositories",
    "github_repository_summary",
    "github_commits",
    "github_issues",
    "github_pull_requests",
    "github_changelog",
    "github_draft_commit_message",
    "github_code_summary",
    "github_traceback_explain",
    "github_project_dashboard",
]
