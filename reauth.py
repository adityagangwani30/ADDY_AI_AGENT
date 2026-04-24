"""
Google OAuth Re-authentication Tool.

Uses google-auth-oauthlib InstalledAppFlow for a reliable, automated OAuth
flow that opens the browser, captures tokens via local redirect, and updates
accounts.json — no manual URL or code copying required.

Usage:
    python reauth.py                  # Check all, re-auth broken ones
    python reauth.py --all            # Force re-auth ALL accounts
    python reauth.py --email X        # Re-auth a specific account
    python reauth.py --check          # Only check, don't re-auth
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console encoding for emoji output
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# ── Project imports ────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

# Load env before importing config
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_TOKEN_URI
from auth.google_auth_manager import ACCOUNTS_FILE, SCOPES

# Path to the OAuth client secrets file (installed app type)
CREDENTIALS_FILE = ROOT / "credentials.json"

# Local server port for the OAuth redirect
OAUTH_PORT = 8080


def _load_accounts_local() -> dict:
    """Load accounts from the local file (ignores ACCOUNTS_JSON env for re-auth)."""
    if ACCOUNTS_FILE.exists():
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, IOError) as exc:
            print(f"Failed to load {ACCOUNTS_FILE}: {exc}")
    return {}


def _save_accounts_local(accounts: dict) -> None:
    """Save accounts to the local file."""
    ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCOUNTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(accounts, fh, indent=2)


# ── Health check ───────────────────────────────────────────────────────

def check_account(email: str, account_data: dict) -> tuple[bool, str]:
    """
    Test if an account's credentials are valid by attempting a token refresh.

    Returns:
        (is_healthy, status_message)
    """
    refresh_token = account_data.get("refresh_token")
    if not refresh_token:
        return False, "❌ No refresh_token stored"

    client_id = account_data.get("client_id", GOOGLE_CLIENT_ID)
    client_secret = account_data.get("client_secret", GOOGLE_CLIENT_SECRET)
    token_uri = account_data.get("token_uri", GOOGLE_TOKEN_URI)

    if not client_id or not client_secret:
        return False, "❌ Missing client_id or client_secret"

    try:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=token_uri,
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        creds.refresh(Request())

        if creds.valid and creds.token:
            return True, "✅ Token refreshed successfully"
        return False, "❌ Refresh returned invalid token"

    except Exception as exc:
        return False, f"❌ Refresh failed: {exc}"


def check_all_accounts() -> dict[str, tuple[bool, str]]:
    """Check health of all accounts. Returns {email: (healthy, msg)}."""
    accounts = _load_accounts_local()
    results = {}
    for email, data in accounts.items():
        healthy, msg = check_account(email, data)
        results[email] = (healthy, msg)
    return results


# ── Core OAuth flow using InstalledAppFlow ─────────────────────────────

def authenticate_account(email: str) -> dict:
    """
    Authenticate a Google account using the official InstalledAppFlow.

    Opens browser automatically, handles redirect internally on localhost,
    captures both access_token and refresh_token, and stores them in
    accounts.json.

    Args:
        email: The email address to authenticate.

    Returns:
        Dict with email, access_token, refresh_token, and expiry.

    Raises:
        RuntimeError: If authentication fails or refresh_token is missing.
        FileNotFoundError: If credentials.json is not found.
    """
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"credentials.json not found at {CREDENTIALS_FILE}.\n"
            "Download it from Google Cloud Console > APIs & Services > Credentials."
        )

    print(f"\n{'='*60}")
    print(f"  Authenticating: {email}")
    print(f"{'='*60}\n")

    # Initialize the InstalledAppFlow from client secrets
    flow = InstalledAppFlow.from_client_secrets_file(
        str(CREDENTIALS_FILE),
        SCOPES,
    )

    print("Opening browser for Google sign-in...")
    print(f"If the browser doesn't open, check http://localhost:{OAUTH_PORT}\n")

    # Run the local server flow — opens browser, captures redirect automatically
    creds = flow.run_local_server(
        port=OAUTH_PORT,
        prompt="consent",           # Force consent to ensure refresh_token
        access_type="offline",      # Request offline access for refresh_token
        login_hint=email,           # Pre-fill email in Google sign-in
    )

    # Extract tokens
    access_token = creds.token
    refresh_token = creds.refresh_token
    expiry = creds.expiry.isoformat() if creds.expiry else None

    # CRITICAL: Handle missing refresh_token
    if not refresh_token:
        print("\n⚠️  Refresh token missing. This usually means Google didn't")
        print("   issue a new refresh token because prior consent still exists.\n")
        print("   To fix this:")
        print("   1. Go to https://myaccount.google.com/permissions")
        print("   2. Find and remove this app's access")
        print("   3. Run this script again\n")
        raise RuntimeError(
            "Refresh token missing. Revoke app access at "
            "https://myaccount.google.com/permissions and retry."
        )

    # Build the account entry
    account_entry = {
        "token": access_token,
        "refresh_token": refresh_token,
        "token_uri": GOOGLE_TOKEN_URI,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "expiry": expiry,
    }

    # Update accounts.json (replace existing, don't duplicate)
    accounts = _load_accounts_local()
    accounts[email] = account_entry
    _save_accounts_local(accounts)
    print(f"\n✅ Tokens saved to {ACCOUNTS_FILE}")

    return {
        "email": email,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expiry": expiry,
    }


# ── Re-authentication (wraps authenticate_account with verification) ──

def reauthenticate_account(email: str) -> bool:
    """
    Re-authenticate a single account with full verification.

    1. Runs OAuth flow via InstalledAppFlow
    2. Stores tokens in accounts.json
    3. Verifies with a token refresh
    4. Tests Gmail API access

    Returns:
        True if successful, False otherwise.
    """
    try:
        result = authenticate_account(email)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\n❌ {exc}")
        return False
    except Exception as exc:
        print(f"\n❌ Unexpected error during authentication: {exc}")
        return False

    # Verify the new tokens work
    print("\nVerifying new tokens...")
    accounts = _load_accounts_local()
    account_data = accounts.get(email, {})
    healthy, msg = check_account(email, account_data)
    print(f"  {msg}")

    if healthy:
        # Try a real API call
        try:
            from tools.gmail_tools import list_emails
            api_result = list_emails(email, max_results=1)
            count = api_result.get("count", 0) if isinstance(api_result, dict) else "?"
            print(f"  ✅ Account connected successfully — {count} email(s) accessible")
        except Exception as exc:
            print(f"  ⚠️ Gmail test failed (tokens may still be valid): {exc}")

    return healthy


def reauthenticate_all_accounts(force: bool = False) -> None:
    """
    Re-authenticate all accounts.

    Args:
        force: If True, re-auth all accounts even if healthy.
               If False, only re-auth broken accounts.
    """
    accounts = _load_accounts_local()
    if not accounts:
        print("No accounts found in accounts.json or ACCOUNTS_JSON env.")
        return

    print(f"\n📋 Found {len(accounts)} account(s):\n")

    # Check health first
    results = {}
    for email in accounts:
        healthy, msg = check_account(email, accounts[email])
        results[email] = (healthy, msg)
        print(f"  {email}: {msg}")

    # Determine which to re-auth
    to_reauth = []
    if force:
        to_reauth = list(accounts.keys())
        print(f"\n🔄 Force mode: re-authenticating ALL {len(to_reauth)} account(s)")
    else:
        to_reauth = [e for e, (h, _) in results.items() if not h]
        if not to_reauth:
            print("\n✅ All accounts are healthy! No re-authentication needed.")
            return
        print(f"\n🔧 {len(to_reauth)} account(s) need re-authentication")

    # Re-auth each
    success = 0
    for email in to_reauth:
        if reauthenticate_account(email):
            success += 1
        else:
            print(f"\n⚠️ Failed to re-authenticate {email}")

    print(f"\n{'='*60}")
    print(f"  Results: {success}/{len(to_reauth)} accounts re-authenticated")
    print(f"{'='*60}")

    # Print updated ACCOUNTS_JSON for Render
    if success > 0:
        updated = _load_accounts_local()
        print("\n📋 Updated ACCOUNTS_JSON for Render (copy this):\n")
        compact = json.dumps(updated, separators=(",", ":"))
        print(compact)
        print()


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google OAuth Re-authentication Tool (InstalledAppFlow)"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only check account health, don't re-authenticate"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Force re-authenticate ALL accounts (even healthy ones)"
    )
    parser.add_argument(
        "--email", type=str, default="",
        help="Re-authenticate a specific account by email"
    )
    parser.add_argument(
        "--add", type=str, default="",
        help="Add and authenticate a NEW account by email"
    )
    args = parser.parse_args()

    # Validate credentials file exists
    if not args.check and not CREDENTIALS_FILE.exists():
        print(f"❌ credentials.json not found at {CREDENTIALS_FILE}")
        print("   Download it from Google Cloud Console > APIs & Services > Credentials.")
        sys.exit(1)

    if args.check:
        print("\n🔍 Checking account health...\n")
        results = check_all_accounts()
        for email, (healthy, msg) in results.items():
            print(f"  {email}: {msg}")
        all_ok = all(h for h, _ in results.values())
        print(f"\n{'✅ All accounts healthy' if all_ok else '⚠️ Some accounts need attention'}")
        sys.exit(0 if all_ok else 1)

    if args.add:
        ok = reauthenticate_account(args.add)
        sys.exit(0 if ok else 1)

    if args.email:
        ok = reauthenticate_account(args.email)
        sys.exit(0 if ok else 1)

    reauthenticate_all_accounts(force=args.all)


if __name__ == "__main__":
    main()
