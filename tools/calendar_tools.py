from __future__ import annotations

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from auth.google_auth_manager import get_credentials


def list_events(account: str, max_results: int = 10) -> dict:
    if max_results < 1:
        raise ValueError("max_results must be >= 1")

    try:
        creds = get_credentials(account)
        service = build("calendar", "v3", credentials=creds)
        events = service.events().list(
            calendarId="primary",
            maxResults=max_results,
            orderBy="startTime",
            singleEvents=True,
        ).execute()
        items = events.get("items", [])
        return {"count": len(items), "events": items}
    except HttpError as exc:
        raise RuntimeError(f"Calendar list_events API error: {exc}") from exc


def create_event(
    account: str,
    summary: str,
    start_time: str,
    end_time: str,
    time_zone: str = "UTC",
) -> dict:
    if not summary:
        raise ValueError("summary is required")
    if not start_time or not end_time:
        raise ValueError("start_time and end_time are required")

    event_payload = {
        "summary": summary,
        "start": {"dateTime": start_time, "timeZone": time_zone},
        "end": {"dateTime": end_time, "timeZone": time_zone},
    }

    try:
        creds = get_credentials(account)
        service = build("calendar", "v3", credentials=creds)
        created = service.events().insert(calendarId="primary", body=event_payload).execute()
        return {"id": created.get("id"), "html_link": created.get("htmlLink")}
    except HttpError as exc:
        raise RuntimeError(f"Calendar create_event API error: {exc}") from exc


def delete_event(account: str, event_id: str) -> dict:
    if not event_id:
        raise ValueError("event_id is required")

    try:
        creds = get_credentials(account)
        service = build("calendar", "v3", credentials=creds)
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return {"deleted": True, "event_id": event_id}
    except HttpError as exc:
        raise RuntimeError(f"Calendar delete_event API error: {exc}") from exc
