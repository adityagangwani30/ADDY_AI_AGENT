from __future__ import annotations

from memory.storage import MemoryRepository, SQLiteMemoryRepository
from auth.google_auth_manager import list_available_accounts


def resolve_account_for_session(
    session_id: str,
    text: str = "",
    memory_repo: MemoryRepository | None = None,
) -> str | None:
    repo = memory_repo or SQLiteMemoryRepository()
    available = list_available_accounts()
    if not available:
        return None

    aliases = repo.list_account_aliases()
    low = (text or "").lower()
    for alias, mapped in aliases.items():
        if alias and alias in low:
            for acc in available:
                if acc.lower() == mapped.lower():
                    repo.set_account_preference(session_id, acc)
                    return acc

    preferred = repo.get_account_preference(session_id)
    if preferred and preferred in available:
        return preferred

    default_alias = aliases.get("personal")
    if default_alias:
        for acc in available:
            if acc.lower() == default_alias.lower():
                repo.set_account_preference(session_id, acc)
                return acc

    repo.set_account_preference(session_id, available[0])
    return available[0]
