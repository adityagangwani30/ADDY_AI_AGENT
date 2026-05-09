from __future__ import annotations

import logging
import mimetypes
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

try:
    from PyPDF2 import PdfReader  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

try:
    from docx import Document as DocxDocument  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    DocxDocument = None

LOGGER = logging.getLogger("services.document_processor")


STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "in",
    "for",
    "is",
    "on",
}

CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _read_docx(path: str) -> str:
    if DocxDocument is None:
        # fallback: read as text
        try:
            return _read_txt(path)
        except Exception:
            return ""
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paragraphs)


def _read_pdf(path: str) -> List[str]:
    if PdfReader is None:
        return []
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def infer_topic(text: str) -> str | None:
    words = re.findall(r"\w+", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w in STOPWORDS or len(w) < 3:
            continue
        freq[w] = freq.get(w, 0) + 1
    if not freq:
        return None
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    return ", ".join([w for w, _ in top])


def detect_language_from_name(filename: str, text: str = "") -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in CODE_EXTENSIONS:
        return CODE_EXTENSIONS[ext]
    if re.search(r"^\s*def\s+\w+\(|^\s*class\s+\w+", text, flags=re.M):
        return "python"
    if re.search(r"^\s*function\s+\w+\(|=>\s*\{", text, flags=re.M):
        return "javascript"
    if re.search(r"^\s*package\s+main|func\s+\w+\(", text, flags=re.M):
        return "go"
    return "text"


def summarize_code(text: str, language: str | None = None) -> Dict[str, List[str] | str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    functions = [line.strip() for line in lines if re.match(r"^(async\s+def|def|class|function|const\s+\w+\s*=\s*\(|export\s+function)\b", line.strip())]
    imports = [line.strip() for line in lines if re.match(r"^(from\s+\S+\s+import\s+|import\s+|const\s+\w+\s*=\s+require\()", line.strip())]
    preview = "\n".join(lines[:12])
    return {
        "language": language or "text",
        "functions": functions[:12],
        "imports": imports[:12],
        "preview": preview[:600],
    }


def chunk_text_paragraphs(text: str, max_chunk_chars: int = 2000) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = []
    cur_len = 0
    for p in parts:
        if cur_len + len(p) > max_chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
            current = [p]
            cur_len = len(p)
        else:
            current.append(p)
            cur_len += len(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def process_file(path: str, meta: Optional[Dict] = None) -> Dict:
    """Extract text, metadata and chunks from a supported file.

    Returns a dict with keys: filename, filetype, upload_date, text_preview,
    inferred_topic, chunks (list), extracted_text (full up to some cap)
    """
    meta = meta or {}
    filename = os.path.basename(path)
    filetype, _ = mimetypes.guess_type(path)
    ext = os.path.splitext(filename)[1].lower()

    result: Dict = {
        "filename": filename,
        "filetype": filetype or ext,
        "upload_date": meta.get("upload_date") or datetime.utcnow().isoformat(),
        "drive_id": meta.get("drive_id"),
        "owner": meta.get("owner") or meta.get("source"),
        "text_preview": None,
        "inferred_topic": None,
        "chunks": [],
        "extracted_text": "",
        "language": None,
        "code_summary": None,
    }

    try:
        if ext in {".txt"}:
            text = _read_txt(path)
            chunks = chunk_text_paragraphs(text)
            result.update({"chunks": chunks, "extracted_text": text})

        elif ext in {".docx", ".doc"}:
            text = _read_docx(path)
            chunks = chunk_text_paragraphs(text)
            result.update({"chunks": chunks, "extracted_text": text})

        elif ext in {".pdf"}:
            pages = _read_pdf(path)
            # page-based chunking
            chunks = [p for p in pages if p and p.strip()]
            extracted = "\n\n".join(chunks)
            result.update({"chunks": chunks, "extracted_text": extracted})

        elif ext in CODE_EXTENSIONS:
            text = _read_txt(path)
            language = detect_language_from_name(filename, text)
            result["language"] = language
            result["code_summary"] = summarize_code(text, language=language)
            result["chunks"] = chunk_text_paragraphs(text)
            result["extracted_text"] = text

        else:
            # attempt plain read as fallback
            text = _read_txt(path)
            chunks = chunk_text_paragraphs(text)
            result.update({"chunks": chunks, "extracted_text": text})

        preview = (result["extracted_text"] or "")[:400]
        result["text_preview"] = preview
        result["inferred_topic"] = infer_topic(result["extracted_text"] or preview)
        if not result.get("language") and ext in CODE_EXTENSIONS:
            result["language"] = CODE_EXTENSIONS.get(ext)

    except Exception as exc:
        LOGGER.exception("process_file: failed for %s: %s", path, exc)

    return result


__all__ = ["process_file", "chunk_text_paragraphs", "infer_topic"]
