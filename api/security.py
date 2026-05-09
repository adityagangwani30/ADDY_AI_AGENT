from __future__ import annotations

import hmac

from config import WEBHOOK_SECRET


def verify_webhook_signature(body: bytes, signature: str | None) -> bool:
    """Verify the Telegram webhook secret token."""
    if not WEBHOOK_SECRET:
        return False
    if not signature:
        return False

    return hmac.compare_digest(WEBHOOK_SECRET.strip(), signature.strip())
