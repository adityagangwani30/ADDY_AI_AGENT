import os
import tempfile

from services.document_processor import process_file, chunk_text_paragraphs


def test_txt_processing():
    t = "Hello world.\n\nThis is a test paragraph.\n\nAnother paragraph."
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(t)

    res = process_file(path)
    assert "extracted_text" in res
    assert "Hello world" in res["extracted_text"]
    chunks = chunk_text_paragraphs(res["extracted_text"], max_chunk_chars=50)
    assert isinstance(chunks, list)
    os.remove(path)
