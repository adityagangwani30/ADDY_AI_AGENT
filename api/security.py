from __future__ import annotations

import hashlib
import hmac

from config import WEBHOOK_SECRET


def verify_webhook_signature(body: bytes, signature: str | None) -> bool:
    """
    Placeholder verification layer for webhook authentication.
    Replace header format and signing mechanism as per provider specs.
    """
    if not WEBHOOK_SECRET:
        return True
    if not signature:
        return False

    expected = hmac.new(WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip())
