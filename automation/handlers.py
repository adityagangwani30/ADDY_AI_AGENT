import logging
import json
from typing import Dict, Any, Callable

logger = logging.getLogger("automation.handlers")

# Simple handler registry for event-driven hooks and task handlers
_HANDLERS: Dict[str, Callable[[Dict[str, Any]], bool]] = {}


def register_handler(task_type: str, fn: Callable[[Dict[str, Any]], bool]):
    _HANDLERS[task_type] = fn


def run_handler(task_type: str, payload: Dict[str, Any]) -> bool:
    fn = _HANDLERS.get(task_type)
    if not fn:
        logger.warning("No handler registered for task_type=%s", task_type)
        return False
    try:
        return bool(fn(payload))
    except Exception as e:
        logger.exception("Handler %s failed: %s", task_type, e)
        return False


def send_reminder(payload: Dict[str, Any]) -> bool:
    # Minimal reminder action: log and (TODO) integrate with notification channels
    who = payload.get("user_id") or payload.get("target")
    message = payload.get("message") or payload.get("text") or "Reminder"
    logger.info("Delivering reminder to %s: %s", who, message)
    # Integrations (telegram, email, etc.) can be wired here using existing services
    return True


def github_issue_reminder(payload: Dict[str, Any]) -> bool:
    message = payload.get("message") or payload.get("summary") or "GitHub issue update"
    payload = dict(payload)
    payload["message"] = message
    logger.info("GitHub issue event received for %s", payload.get("repository"))
    return send_reminder(payload)


def github_pr_merged(payload: Dict[str, Any]) -> bool:
    message = payload.get("message") or payload.get("summary") or "GitHub pull request merged"
    payload = dict(payload)
    payload["message"] = message
    logger.info("GitHub PR event received for %s", payload.get("repository"))
    return send_reminder(payload)


def github_repo_update(payload: Dict[str, Any]) -> bool:
    message = payload.get("message") or payload.get("summary") or "GitHub repository updated"
    payload = dict(payload)
    payload["message"] = message
    logger.info("GitHub repo update for %s", payload.get("repository"))
    return send_reminder(payload)


# register built-in handlers
register_handler("send_reminder", send_reminder)
register_handler("github_issue_reminder", github_issue_reminder)
register_handler("github_pr_merged", github_pr_merged)
register_handler("github_repo_update", github_repo_update)
