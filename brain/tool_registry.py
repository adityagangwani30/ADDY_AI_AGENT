from __future__ import annotations

from domain.schemas import (
    CreateEventParams,
    DeleteEmailParams,
    DeleteEventParams,
    DeleteFileParams,
    ListEmailsParams,
    ListEventsParams,
    ListFilesParams,
    SearchEmailParams,
    SendEmailParams,
    UploadFileParams,
)
from tools.calendar_tools import create_event, delete_event, list_events
from tools.drive_tools import delete_file, list_files, upload_file
from tools.gmail_tools import delete_email, list_emails, search_email, send_email

TOOLS = {
    "send_email": send_email,
    "list_emails": list_emails,
    "search_email": search_email,
    "delete_email": delete_email,
    "create_event": create_event,
    "list_events": list_events,
    "delete_event": delete_event,
    "list_files": list_files,
    "upload_file": upload_file,
    "delete_file": delete_file,
}

DESTRUCTIVE_TOOLS = {"send_email", "delete_email", "delete_file", "delete_event"}

TOOL_PARAMETER_MODELS = {
    "send_email": SendEmailParams,
    "list_emails": ListEmailsParams,
    "search_email": SearchEmailParams,
    "delete_email": DeleteEmailParams,
    "create_event": CreateEventParams,
    "list_events": ListEventsParams,
    "delete_event": DeleteEventParams,
    "list_files": ListFilesParams,
    "upload_file": UploadFileParams,
    "delete_file": DeleteFileParams,
}
