from __future__ import annotations

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from auth.google_auth_manager import get_credentials


def list_events(account: str, max_results: int = 10) -> dict:
    """
    Fetch upcoming events from the primary Google Calendar.

    Args:
        account: The account identifier used to retrieve OAuth credentials.
        max_results: Maximum number of events to return (must be >= 1).

    Returns:
        A dict with ``count`` (int) and ``events`` (list of event objects).

    Raises:
        ValueError: If max_results is less than 1.
        RuntimeError: If the Calendar API returns an HTTP error.
    """
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
    """
    Create a new event on the primary Google Calendar.

    Args:
        account: The account identifier used to retrieve OAuth credentials.
        summary: Title/name of the event.
        start_time: ISO 8601 datetime string for the event start (e.g. ``2024-01-15T10:00:00``).
        end_time: ISO 8601 datetime string for the event end.
        time_zone: IANA timezone name (default ``"UTC"``).

    Returns:
        A dict with ``id`` and ``html_link`` of the created event.

    Raises:
        ValueError: If summary, start_time, or end_time are missing.
        RuntimeError: If the Calendar API returns an HTTP error.
    """
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
    """
    Permanently delete an event from the primary Google Calendar.

    Args:
        account: The account identifier used to retrieve OAuth credentials.
        event_id: The Google Calendar event ID to delete.

    Returns:
        A dict with ``deleted`` (True) and ``event_id``.

    Raises:
        ValueError: If event_id is empty.
        RuntimeError: If the Calendar API returns an HTTP error.
    """
    if not event_id:
        raise ValueError("event_id is required")

    try:
        creds = get_credentials(account)
        service = build("calendar", "v3", credentials=creds)
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return {"deleted": True, "event_id": event_id}
    except HttpError as exc:
        raise RuntimeError(f"Calendar delete_event API error: {exc}") from exc
