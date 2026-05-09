from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from auth.google_auth_manager import get_credentials


def _build_calendar_service(account: str):
    creds = get_credentials(account)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _normalize_time(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw
    if raw.endswith("Z"):
        return raw
    if "+" in raw or raw.count("-") > 2:
        return raw
    return f"{raw}Z"


def _event_matches_title(event: dict[str, Any], title: str) -> bool:
    if not title:
        return False
    summary = str(event.get("summary") or "").lower()
    query = title.lower().strip()
    return bool(query) and query in summary


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
        service = _build_calendar_service(account)
        now_utc = datetime.now(timezone.utc).isoformat()
        events = service.events().list(
            calendarId="primary",
            timeMin=now_utc,
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

    start_time = _normalize_time(start_time)
    end_time = _normalize_time(end_time)

    event_payload = {
        "summary": summary,
        "start": {"dateTime": start_time, "timeZone": time_zone},
        "end": {"dateTime": end_time, "timeZone": time_zone},
    }

    try:
        service = _build_calendar_service(account)

        # Avoid accidental duplicate creation with identical summary/start.
        existing = service.events().list(
            calendarId="primary",
            timeMin=start_time,
            timeMax=end_time,
            maxResults=20,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        for item in existing.get("items", []):
            item_summary = str(item.get("summary") or "")
            item_start = str(item.get("start", {}).get("dateTime") or "")
            if item_summary.strip().lower() == summary.strip().lower() and item_start == start_time:
                return {
                    "id": item.get("id"),
                    "html_link": item.get("htmlLink"),
                    "duplicate": True,
                }

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
        service = _build_calendar_service(account)
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return {"deleted": True, "event_id": event_id}
    except HttpError as exc:
        raise RuntimeError(f"Calendar delete_event API error: {exc}") from exc


def edit_event(
    account: str,
    event_id: str = "",
    title: str = "",
    start_time: str = "",
    end_time: str = "",
    time_zone: str = "UTC",
) -> dict:
    """
    Update an event by event_id, or by best-effort title match if event_id is not provided.
    """
    if not event_id and not title:
        raise ValueError("event_id or title is required")

    try:
        service = _build_calendar_service(account)

        target_id = event_id
        if not target_id:
            now_utc = datetime.now(timezone.utc).isoformat()
            future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            events = service.events().list(
                calendarId="primary",
                timeMin=now_utc,
                timeMax=future,
                maxResults=50,
                singleEvents=True,
                orderBy="startTime",
            ).execute().get("items", [])
            for item in events:
                if _event_matches_title(item, title):
                    target_id = str(item.get("id") or "")
                    break

        if not target_id:
            raise ValueError("No matching event found to edit")

        existing = service.events().get(calendarId="primary", eventId=target_id).execute()
        if title:
            existing["summary"] = title
        if start_time:
            existing.setdefault("start", {})["dateTime"] = _normalize_time(start_time)
            existing.setdefault("start", {})["timeZone"] = time_zone
        if end_time:
            existing.setdefault("end", {})["dateTime"] = _normalize_time(end_time)
            existing.setdefault("end", {})["timeZone"] = time_zone

        updated = service.events().update(calendarId="primary", eventId=target_id, body=existing).execute()
        return {"updated": True, "id": updated.get("id"), "html_link": updated.get("htmlLink")}
    except HttpError as exc:
        raise RuntimeError(f"Calendar edit_event API error: {exc}") from exc


def calendar_create(
    account: str,
    title: str,
    start_time: str,
    end_time: str,
    time_zone: str = "UTC",
    participants: list[str] | None = None,
) -> dict:
    payload = create_event(account, summary=title, start_time=start_time, end_time=end_time, time_zone=time_zone)
    if participants:
        payload["participants"] = participants
    return payload


def calendar_edit(
    account: str,
    event_id: str = "",
    title: str = "",
    start_time: str = "",
    end_time: str = "",
    time_zone: str = "UTC",
) -> dict:
    return edit_event(
        account=account,
        event_id=event_id,
        title=title,
        start_time=start_time,
        end_time=end_time,
        time_zone=time_zone,
    )


def calendar_delete(account: str, event_id: str = "", title: str = "", date_hint: str = "") -> dict:
    if event_id:
        return delete_event(account, event_id)

    if not title:
        raise ValueError("event_id or title is required")

    try:
        service = _build_calendar_service(account)
        now_utc = datetime.now(timezone.utc).isoformat()
        events = service.events().list(
            calendarId="primary",
            timeMin=now_utc,
            maxResults=50,
            singleEvents=True,
            orderBy="startTime",
        ).execute().get("items", [])

        matched_id = ""
        hint = (date_hint or "").lower()
        for item in events:
            if not _event_matches_title(item, title):
                continue
            if hint:
                start_text = str(item.get("start", {}).get("dateTime") or item.get("start", {}).get("date") or "").lower()
                if hint not in start_text:
                    continue
            matched_id = str(item.get("id") or "")
            break

        if not matched_id:
            raise ValueError("No matching event found to delete")
        return delete_event(account, matched_id)
    except HttpError as exc:
        raise RuntimeError(f"Calendar calendar_delete API error: {exc}") from exc


def calendar_list(account: str, max_results: int = 10) -> dict:
    return list_events(account=account, max_results=max_results)
