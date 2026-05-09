import tempfile
from pathlib import Path
import unittest

from memory.memory_manager import MemoryManager


def _make_mgr():
    td = tempfile.TemporaryDirectory()
    db = Path(td.name) / "memtest.db"
    mgr = MemoryManager(db_path=str(db))
    return mgr, td


class MemoryManagerTests(unittest.TestCase):
    def test_memory_save_and_retrieve(self):
        mgr, td = _make_mgr()
        uid = "user_test"
        entry_id = mgr.save_memory(uid, "preference", "email_style", {"style": "formal"}, importance=5)
        self.assertIsInstance(entry_id, int)
        items = mgr.get_memories(uid, "preference")
        self.assertTrue(any(i["key"] == "email_style" for i in items))
        td.cleanup()

    def test_alias_set_and_resolve(self):
        mgr, td = _make_mgr()
        uid = "u_alias"
        mgr.set_alias(uid, "sir", "prof@example.com", entity_type="contact")
        resolved = mgr.resolve_alias(uid, "sir")
        self.assertEqual(resolved, "prof@example.com")
        td.cleanup()

    def test_recent_context_and_cleanup(self):
        mgr, td = _make_mgr()
        uid = "u_ctx"
        mgr.add_context(uid, "user", "Hello there")
        mgr.add_context(uid, "assistant", "Hi")
        ctx = mgr.get_recent_context(uid, limit=10)
        self.assertGreaterEqual(len(ctx), 2)
        removed = mgr.clear_context_older_than(0)
        self.assertIsInstance(removed, int)
        td.cleanup()

    def test_preference_set_get(self):
        mgr, td = _make_mgr()
        uid = "u_pref"
        mgr.set_preference(uid, "tone", "formal")
        val = mgr.get_preference(uid, "tone")
        self.assertEqual(val, "formal")
        td.cleanup()

    def test_search_memory(self):
        mgr, td = _make_mgr()
        uid = "u_search"
        mgr.save_memory(uid, "reference", "resume", {"filename": "CV_2026.pdf"}, importance=3)
        results = mgr.search(uid, "resume", limit=5)
        self.assertTrue(any(r.get("key") == "resume" or "resume" in str(r.get("value", "")).lower() for r in results))
        td.cleanup()


if __name__ == '__main__':
    unittest.main()
