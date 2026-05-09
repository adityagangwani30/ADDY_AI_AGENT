from __future__ import annotations

import logging
import re
from typing import List

from brain.llm_provider import call_llm
from memory.file_index import FileIndex
from services.document_processor import chunk_text_paragraphs

LOGGER = logging.getLogger("services.document_qa")


def _score_chunk(chunk: str, query: str) -> int:
    qwords = re.findall(r"\w+", query.lower())
    cw = re.findall(r"\w+", chunk.lower())
    score = 0
    for w in qwords:
        score += cw.count(w)
    return score


def answer_question(user_id: str, question: str, max_chars: int = 1800, request_id: str = "unknown") -> str:
    """Retrieve top chunks from indexed files and ask the LLM to answer.

    Deterministic-first: uses `FileIndex.search` to find matching files,
    then extracts highest-scoring chunks and constructs a compact prompt.
    """
    idx = FileIndex()
    files = idx.search(question, limit=5)

    # collect scored chunks
    candidates: List[tuple[int, str]] = []
    for f in files:
        text = f.get("extracted_text") or ""
        if not text:
            continue
        chunks = chunk_text_paragraphs(text, max_chunk_chars=1000)
        for i, c in enumerate(chunks):
            s = _score_chunk(c, question)
            if s > 0:
                header = f"[FILE: {f.get('filename')} | score={s} | idx={i}]\n"
                candidates.append((s, header + c))

    # if no deterministic matches, fall back to returning an informative message
    if not candidates:
        return "I couldn't find relevant content in your indexed files."

    # pick top chunks until max_chars
    candidates.sort(key=lambda x: x[0], reverse=True)
    assembled = []
    cur_len = 0
    for _, txt in candidates:
        if cur_len + len(txt) > max_chars:
            continue
        assembled.append(txt)
        cur_len += len(txt)

    prompt_parts = [
        "You are an assistant that answers questions using only the provided document excerpts.",
        "If the answer cannot be found in the excerpts, say you couldn't find it and be concise.",
        "Do not hallucinate facts.",
        "\\n\\n--- BEGIN EXCERPTS ---\\n",
    ]
    prompt_parts.extend(assembled)
    prompt_parts.append("\\n--- END EXCERPTS ---\\n")
    prompt_parts.append(f"Question: {question}")
    prompt = "\n\n".join(prompt_parts)

    try:
        resp = call_llm(prompt=prompt, system_prompt="You are a concise document QA assistant.", request_id=request_id, phase="document_qa")
        if not resp:
            return "I'm having trouble accessing the reasoning model right now."
        return resp
    except Exception as exc:
        LOGGER.exception("document_qa failed: %s", exc)
        return "An error occurred while answering the question."


__all__ = ["answer_question"]
