from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.assistant import PhaseOneAssistant
from agent.intent_router import IntentRouter
from api import security as webhook_security
from memory.storage import SQLiteMemoryRepository
from services.account_manager import resolve_account_for_session


class SafetyAndRoutingTests(unittest.TestCase):
    def test_intent_router_quick_routes_email_read(self) -> None:
        route = IntentRouter().route("list my emails", "request-1")
        self.assertEqual(route.intent, "gmail_read")

    def test_webhook_signature_verification(self) -> None:
        with patch.object(webhook_security, "WEBHOOK_SECRET", "secret-token"):
            self.assertTrue(webhook_security.verify_webhook_signature(b"{}", "secret-token"))
            self.assertFalse(webhook_security.verify_webhook_signature(b"{}", "wrong-token"))

    def test_account_resolution_prefers_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = SQLiteMemoryRepository(f"{tmp_dir}/memory.db")
            repo.set_account_alias("work", "work@example.com")
            with patch("services.account_manager.list_available_accounts", return_value=["work@example.com", "other@example.com"]):
                account = resolve_account_for_session("session-1", text="use my work account", memory_repo=repo)
        self.assertEqual(account, "work@example.com")

    def test_risky_action_requires_confirmation_before_execution(self) -> None:
        assistant = PhaseOneAssistant()
        assistant.router.route = MagicMock(return_value=SimpleNamespace(intent="calendar_delete", parameters={"event_id": "evt-1"}, confidence=1.0))
        assistant.executor.execute = MagicMock(return_value={"ok": True, "result": {"deleted": True}, "latency_ms": 1})

        with patch("services.account_manager.list_available_accounts", return_value=["test@example.com"]):
            first_result = assistant.run("delete the meeting", "session-1", "request-1")
            self.assertEqual(first_result.status, "confirmation_required")
            assistant.executor.execute.assert_not_called()

            second_result = assistant.run("confirm", "session-1", "request-2")
            self.assertEqual(second_result.status, "ok")
            assistant.executor.execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
