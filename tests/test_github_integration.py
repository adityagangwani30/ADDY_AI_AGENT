from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from integrations.github_service import GitHubService
from services.document_processor import process_file


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}
        self.text = "{}" if payload is not None else ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


class _Session:
    def __init__(self, routes):
        self.routes = routes
        self.headers = {}

    def request(self, method, url, params=None, timeout=None):
        key = (method, url)
        response = self.routes.get(key)
        if response is None:
            raise AssertionError(f"Unexpected request: {key}")
        return response


def test_repository_summary_uses_readme_and_structure(tmp_path):
    base = "https://api.github.com"
    session = _Session(
        {
            ("GET", f"{base}/repos/octo/demo"): _Response({
                "id": 1,
                "name": "demo",
                "full_name": "octo/demo",
                "description": "Demo repository",
                "html_url": "https://github.com/octo/demo",
                "default_branch": "main",
                "language": "Python",
                "topics": ["fastapi"],
                "owner": {"login": "octo"},
            }),
            ("GET", f"{base}/repos/octo/demo/readme"): _Response({
                "name": "README.md",
                "content": "# Demo\n\nFastAPI service with SQLite storage.",
            }),
            ("GET", f"{base}/repos/octo/demo/contents"): _Response([
                {"name": "app.py", "type": "file", "path": "app.py"},
                {"name": "services", "type": "dir", "path": "services"},
                {"name": "README.md", "type": "file", "path": "README.md"},
            ]),
        }
    )
    service = GitHubService(token="test-token", session=session)
    summary = service.summarize_repository("octo", "demo", user_id="user-1")
    assert summary["repository"]["full_name"] == "octo/demo"
    assert "FastAPI" in " ".join(summary["tech_stack"])
    assert summary["structure"]["directories"] == ["services"]


def test_code_summary_detects_language():
    service = GitHubService(token="test-token", session=_Session({}))
    result = service.summarize_code_snippet("import os\n\ndef hello():\n    return os.getcwd()", language=None)
    assert result["language"] == "python"
    assert any("def hello" in item for item in result["functions"])


def test_document_processor_handles_code_files(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("from fastapi import FastAPI\n\napp = FastAPI()\n\ndef ping():\n    return {'ok': True}\n", encoding="utf-8")
    result = process_file(str(path))
    assert result["language"] == "python"
    assert result["code_summary"] is not None
    assert "FastAPI" in result["code_summary"]["preview"]
