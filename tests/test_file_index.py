import os
import tempfile

from memory.file_index import FileIndex


def test_file_index_add_search():
    tmp = tempfile.mkstemp()[1]
    os.close(os.open(tmp, os.O_RDONLY))
    db_path = tmp + ".sqlite"
    idx = FileIndex(db_path=db_path)
    idx.add_file(file_id="f1", filename="Test Document.pdf", extracted_text="This is about RL and PPO", keywords="rl,ppo", source="local")
    results = idx.search("RL")
    assert isinstance(results, list)
    assert any("Test Document" in r.get("filename", "") for r in results)
    try:
        os.remove(db_path)
    except Exception:
        pass
