from __future__ import annotations

import base64
from email.mime.text import MIMEText

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from auth.google_auth_manager import get_credentials


def list_emails(account: str, max_results: int = 5) -> dict:
    if max_results < 1:
        raise ValueError("max_results must be >= 1")

    try:
        creds = get_credentials(account)
        service = build("gmail", "v1", credentials=creds)
        results = service.users().messages().list(userId="me", maxResults=max_results).execute()
        messages = results.get("messages", [])
        return {"count": len(messages), "messages": messages}
    except HttpError as exc:
        raise RuntimeError(f"Gmail list_emails API error: {exc}") from exc


def send_email(account: str, to: str, subject: str, body: str) -> dict:
    if not to or not subject:
        raise ValueError("Both 'to' and 'subject' are required.")

    try:
        creds = get_credentials(account)
        service = build("gmail", "v1", credentials=creds)

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {"id": sent.get("id"), "thread_id": sent.get("threadId")}
    except HttpError as exc:
        raise RuntimeError(f"Gmail send_email API error: {exc}") from exc


def search_email(account: str, query: str, max_results: int = 10) -> dict:
    if not query:
        raise ValueError("query is required")

    try:
        creds = get_credentials(account)
        service = build("gmail", "v1", credentials=creds)
        results = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        messages = results.get("messages", [])
        return {"count": len(messages), "messages": messages}
    except HttpError as exc:
        raise RuntimeError(f"Gmail search_email API error: {exc}") from exc


def delete_email(account: str, message_id: str) -> dict:
    if not message_id:
        raise ValueError("message_id is required")

    try:
        creds = get_credentials(account)
        service = build("gmail", "v1", credentials=creds)
        service.users().messages().delete(userId="me", id=message_id).execute()
        return {"deleted": True, "message_id": message_id}
    except HttpError as exc:
        raise RuntimeError(f"Gmail delete_email API error: {exc}") from exc
