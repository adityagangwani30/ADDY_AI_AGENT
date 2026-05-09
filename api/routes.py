from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from brain.ai_brain import GLOBAL_FALLBACK_RESPONSE, run_agent
from brain.llm_provider import get_provider_health
from brain.tool_registry import TOOLS
from auth.token_validator import validate_oauth_health
from api.security import verify_webhook_signature
from config import TELEGRAM_BOT_TOKEN
from config import WEBHOOK_SECRET
from services.account_manager import resolve_account_for_session
from memory.storage import SQLiteMemoryRepository
from services.document_processor import process_file
from memory.file_index import FileIndex
from memory.memory_manager import MemoryManager
from services import ocr_service

router = APIRouter()
LOGGER = logging.getLogger(__name__)
MEMORY_REPO = SQLiteMemoryRepository()

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


# ────────────────────────────────────────────
# Telegram helper functions
# ────────────────────────────────────────────


def _log(level: int, **payload: object) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


async def _telegram_post(endpoint: str, payload: dict[str, object]) -> dict[str, object] | None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{TELEGRAM_API}/{endpoint}", json=payload)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None


async def send_text(chat_id: int, text: str, parse_mode: str | None = None) -> None:
    """Send a plain text message with keyboard removed."""
    try:
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": text,
            "reply_markup": {"remove_keyboard": True},
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        await _telegram_post(
            "sendMessage",
            payload,
        )
    except Exception as exc:
        _log(logging.ERROR, event="telegram_send_error", error_type=type(exc).__name__, message=str(exc))


async def send_typing(chat_id: int) -> None:
    try:
        await _telegram_post(
            "sendChatAction",
            {"chat_id": chat_id, "action": "typing"},
        )
    except Exception as exc:
        _log(logging.ERROR, event="telegram_typing_error", error_type=type(exc).__name__, message=str(exc))


async def _download_telegram_file(file_id: str) -> Path:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id})
        response.raise_for_status()
        info = response.json()
    if not info.get("ok"):
        raise RuntimeError("Telegram getFile failed")

    file_path = str((info.get("result") or {}).get("file_path") or "")
    if not file_path:
        raise RuntimeError("Telegram file path missing")

    download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(download_url)
        response.raise_for_status()
        content = response.content

    folder = Path(tempfile.gettempdir()) / "ai-assistant-telegram"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(file_path).name
    target.write_bytes(content)
    return target


async def _upload_telegram_document_to_drive(chat_id: int, session_id: str, document_payload: dict[str, object]) -> str:
    file_id = str(document_payload.get("file_id") or "")
    if not file_id:
        return "❌ Could not read the incoming file."

    account = resolve_account_for_session(session_id, memory_repo=MEMORY_REPO)
    if not account:
        return "❌ No connected Google account found for Drive upload."

    local_path: Path | None = None
    try:
        local_path = await _download_telegram_file(file_id)
        # Lightweight local processing & indexing before upload
        try:
            processed = process_file(str(local_path), meta={"source": "telegram", "upload_date": None})
            index = FileIndex()
            idx_file_id = f"telegram:{file_id}"
            index.add_file(
                file_id=idx_file_id,
                filename=processed.get("filename") or local_path.name,
                extracted_text=(processed.get("extracted_text") or "")[:200000],
                keywords=processed.get("inferred_topic") or "",
                source="telegram",
                category=None,
                aliases=None,
            )
            # attempt simple OCR for images/PDFs if text empty
            if not processed.get("extracted_text"):
                try:
                    # ocr_service may be optional
                    if local_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
                        ocr_text = ocr_service.ocr_image(str(local_path))
                    elif local_path.suffix.lower() == ".pdf":
                        ocr_text = ocr_service.ocr_pdf(str(local_path))
                    else:
                        ocr_text = ""
                    if ocr_text:
                        index.add_file(
                            file_id=idx_file_id,
                            filename=processed.get("filename") or local_path.name,
                            extracted_text=ocr_text[:200000],
                            keywords=processed.get("inferred_topic") or "",
                            source="telegram",
                        )
                except Exception:
                    pass
        except Exception:
            LOGGER.exception("telegram:processing failed for %s", file_id)
        drive_upload = TOOLS.get("drive_upload")
        if drive_upload is None:
            return "❌ Drive upload tool is not available."
        result = drive_upload(account, file_path=str(local_path))
        # If upload succeeded, update index entry with drive id
        try:
            drive_id = result.get("id") or result.get("fileId")
            if drive_id:
                idx = FileIndex()
                rec = idx.get_by_file_id(f"telegram:{file_id}")
                if rec:
                    idx.add_file(
                        file_id=str(drive_id),
                        filename=rec.get("filename"),
                        extracted_text=rec.get("extracted_text"),
                        keywords=rec.get("keywords"),
                        source="drive",
                        upload_ts=rec.get("upload_ts"),
                    )
        except Exception:
            LOGGER.exception("telegram: failed to update index with drive id for %s", file_id)

        return "📂 File uploaded to Drive\n" f"Name: {result.get('name', local_path.name)}"
    except Exception as exc:
        return f"❌ Drive upload failed: {exc}"
    finally:
        if local_path and local_path.exists():
            try:
                os.remove(local_path)
            except Exception:
                pass
# ────────────────────────────────────────────
# Webhook route
# ────────────────────────────────────────────


@router.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    raw_body = getattr(request.state, "raw_body", b"") or await request.body()
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")

    if not verify_webhook_signature(raw_body, secret_token):
        return JSONResponse({"ok": False, "message": "Invalid webhook signature"}, status_code=401)

    try:
        data = json.loads(raw_body.decode("utf-8")) if raw_body else await request.json()
    except Exception:
        return JSONResponse({"ok": True})

    # Ignore non-message updates (edits, joins, etc.)
    if "message" not in data:
        return JSONResponse({"ok": True})

    message_data = data["message"]
    chat = message_data.get("chat", {})
    chat_id = chat.get("id")
    user_text = (message_data.get("text") or "").strip()
    document = message_data.get("document")

    if not chat_id:
        return JSONResponse({"ok": True})

    # Telegram file flow: document upload to Drive.
    if isinstance(document, dict):
        await send_typing(chat_id)
        reply = await _upload_telegram_document_to_drive(chat_id, str(chat_id), document)
        await send_text(chat_id, reply)
        return JSONResponse({"ok": True})

    if not user_text:
        return JSONResponse({"ok": True})

    # ── /start command ──────────────────────
    if user_text == "/start":
        await send_text(chat_id, (
            "Hey! 👋 I'm your personal AI assistant.\n\n"
            "Just type what you need — here are some things I can help with:\n\n"
            "📩 *Email* — \"list my emails\", \"send email to X\", \"any urgent emails?\"\n"
            "📅 *Calendar* — \"what meetings do I have today?\", \"create a meeting\"\n"
            "📁 *Drive* — \"list my drive files\"\n"
            "🤖 *Anything else* — \"summarize this\", ask me a question\n\n"
            "Just type naturally and I'll handle the rest!"
        ), parse_mode="Markdown")
        return JSONResponse({"ok": True})

    # ── All input goes straight to the AI agent ──
    _log(
        logging.INFO,
        event="agent_execution_start",
        request_id=request_id,
        tool_name=None,
        latency_ms=None,
        error_type=None,
    )

    await send_typing(chat_id)
    reply_text = await _run_agent_safe(user_text, str(chat_id), request_id, started)
    await send_text(chat_id, reply_text)
    return JSONResponse({"ok": True})


# ────────────────────────────────────────────
# Agent execution wrapper
# ────────────────────────────────────────────


async def _run_agent_safe(
    user_message: str,
    session_id: str,
    request_id: str,
    started: float,
) -> str:
    """Run the agent in a thread with timeout, returning the reply string."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_agent,
                user_message=user_message,
                session_id=session_id,
                request_id=request_id,
            ),
            timeout=25,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.INFO,
            event="agent_execution_finish",
            request_id=request_id,
            tool_name=result.tool_name,
            latency_ms=latency_ms,
            error_type=result.error_type,
        )
        return result.message

    except asyncio.TimeoutError:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="agent_execution_timeout",
            request_id=request_id,
            tool_name=None,
            latency_ms=latency_ms,
            error_type="TimeoutError",
        )
        return "⏱ Request timed out. Please try again."

    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()

        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="agent_execution_error",
            request_id=request_id,
            tool_name=None,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return GLOBAL_FALLBACK_RESPONSE


@router.get("/health")
async def health() -> JSONResponse:
    database_ok = False
    try:
        database_ok = MEMORY_REPO.health_check()
    except Exception as exc:
        _log(logging.ERROR, event="health_database_error", error_type=type(exc).__name__, message=str(exc))

    oauth_health = validate_oauth_health()
    provider_health = get_provider_health()
    telegram_ok = bool(TELEGRAM_BOT_TOKEN and WEBHOOK_SECRET)

    overall = "healthy"
    if not database_ok or oauth_health.get("status") != "healthy" or provider_health.get("status") not in {"healthy", "degraded"} or not telegram_ok:
        overall = "degraded"

    payload = {
        "status": overall,
        "database": "ok" if database_ok else "error",
        "llm": provider_health,
        "google_auth": oauth_health,
        "telegram": "ok" if telegram_ok else "missing",
    }
    return JSONResponse(payload)