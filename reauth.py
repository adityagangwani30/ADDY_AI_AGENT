"""Google OAuth token regeneration and validation CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from auth.google_auth_manager import (
    ACCOUNTS_FILE,
    SCOPES,
    AuthReauthRequired,
    authenticate_account as core_authenticate_account,
)
from auth.token_validator import validate_account_health, validate_oauth_health

CREDENTIALS_FILE = ROOT / "credentials.json"
authenticate_account = core_authenticate_account


def _load_accounts_local() -> dict:
    if ACCOUNTS_FILE.exists():
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, IOError) as exc:
            print(f"Failed to load {ACCOUNTS_FILE}: {exc}")
    return {}


def check_account(email: str, account_data: dict) -> tuple[bool, str]:
    health = validate_account_health(email)
    if health.get("status") == "healthy":
        return True, f"✅ Scopes active: {', '.join(health.get('active_scopes', [])[:4])}"
    if health.get("status") == "degraded":
        return False, f"⚠️ Token refreshed, but validation is degraded: {health}"
    return False, f"❌ {health.get('error', 'Token validation failed')}"


def check_all_accounts() -> dict[str, tuple[bool, str]]:
    accounts = _load_accounts_local()
    return {email: check_account(email, data) for email, data in accounts.items()}


def reauthenticate_account(email: str) -> bool:
    try:
        authenticate_account(email)
    except (RuntimeError, FileNotFoundError, AuthReauthRequired) as exc:
        print(f"\n❌ {exc}")
        return False
    except Exception as exc:
        print(f"\n❌ Unexpected error during authentication: {exc}")
        return False

    health = validate_account_health(email)
    if health.get("status") == "healthy":
        print("\n✅ Account connected successfully with Gmail, Calendar, and Drive access")
        return True

    print(f"\n⚠️ Account saved, but validation is {health.get('status')}: {health}")
    return False


def reauthenticate_all_accounts(force: bool = False) -> None:
    accounts = _load_accounts_local()
    if not accounts:
        print("No accounts found in accounts.json or ACCOUNTS_JSON env.")
        return

    print(f"\n📋 Found {len(accounts)} account(s):\n")

    results: dict[str, tuple[bool, str]] = {}
    for email in accounts:
        healthy, msg = check_account(email, accounts[email])
        results[email] = (healthy, msg)
        print(f"  {email}: {msg}")

    if force:
        to_reauth = list(accounts.keys())
        print(f"\n🔄 Force mode: re-authenticating ALL {len(to_reauth)} account(s)")
    else:
        to_reauth = [email for email, (healthy, _) in results.items() if not healthy]
        if not to_reauth:
            print("\n✅ All accounts are healthy! No re-authentication needed.")
            return
        print(f"\n🔧 {len(to_reauth)} account(s) need re-authentication")

    success = 0
    for email in to_reauth:
        if reauthenticate_account(email):
            success += 1
        else:
            print(f"\n⚠️ Failed to re-authenticate {email}")

    print(f"\n{'='*60}")
    print(f"  Results: {success}/{len(to_reauth)} accounts re-authenticated")
    print(f"{'='*60}")

    if success > 0:
        updated = _load_accounts_local()
        print("\n📋 Updated ACCOUNTS_JSON for Render (copy this):\n")
        print(json.dumps(updated, separators=(",", ":")))
        print()


def validate_bootstrap_tokens() -> dict:
    health = validate_oauth_health()
    bootstrap = health.get("bootstrap", {})
    print(f"Bootstrap token status: {bootstrap.get('status')}")

    bootstrap_tokens = health.get("bootstrap_tokens", {})
    for slot_name, token_health in bootstrap_tokens.items():
        print(f"  {slot_name}: {token_health.get('status')}")

    return health


def main() -> None:
    parser = argparse.ArgumentParser(description="Google OAuth Re-authentication Tool (InstalledAppFlow)")
    parser.add_argument("--check", action="store_true", help="Only check account health, don't re-authenticate")
    parser.add_argument("--all", action="store_true", help="Force re-authenticate ALL accounts (even healthy ones)")
    parser.add_argument("--email", type=str, default="", help="Re-authenticate a specific account by email")
    parser.add_argument("--add", type=str, default="", help="Add and authenticate a NEW account by email")
    args = parser.parse_args()

    if not args.check and not CREDENTIALS_FILE.exists():
        print(f"❌ credentials.json not found at {CREDENTIALS_FILE}")
        print("   Download it from Google Cloud Console > APIs & Services > Credentials.")
        sys.exit(1)

    if args.check:
        print("\n🔍 Checking account health...\n")
        validate_bootstrap_tokens()
        results = check_all_accounts()
        for email, (healthy, msg) in results.items():
            print(f"  {email}: {msg}")
        all_ok = all(healthy for healthy, _ in results.values())
        print(f"\n{'✅ All accounts healthy' if all_ok else '⚠️ Some accounts need attention'}")
        sys.exit(0 if all_ok else 1)

    if args.add:
        sys.exit(0 if reauthenticate_account(args.add) else 1)

    if args.email:
        sys.exit(0 if reauthenticate_account(args.email) else 1)

    reauthenticate_all_accounts(force=args.all)


if __name__ == "__main__":
    main()
