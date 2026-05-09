"""Lightweight decision engine for Phase 5.

Provides deterministic-first reasoning for:
- Tool selection
- Execution order
- Parameter resolution
- Ambiguity detection
- Fallback handling

Minimal LLM usage - mostly rule-based decisions.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from brain.llm_provider import call_llm
from memory.storage import SQLiteMemoryRepository

LOGGER = logging.getLogger(__name__)

# Tool capability mappings
TOOL_CAPABILITIES = {
    "gmail_send": {"sends": "email", "category": "communication", "risky": True},
    "gmail_read": {"reads": "email", "category": "information", "risky": False},
    "gmail_draft": {"creates": "draft", "category": "communication", "risky": False},
    "calendar_create": {"creates": "event", "category": "scheduling", "risky": False},
    "calendar_delete": {"deletes": "event", "category": "scheduling", "risky": True},
    "calendar_edit": {"edits": "event", "category": "scheduling", "risky": True},
    "drive_search": {"searches": "files", "category": "information", "risky": False},
    "drive_retrieve": {"retrieves": "files", "category": "information", "risky": False},
    "drive_upload": {"uploads": "files", "category": "storage", "risky": False},
    "drive_share": {"shares": "files", "category": "sharing", "risky": True},
}

# Tool chaining templates: what tools work well in sequence
TOOL_CHAINS = {
    "find_and_send": {
        "description": "Find file and send via email",
        "chain": ["drive_search", "gmail_send"],
    },
    "create_and_share": {
        "description": "Create file and share",
        "chain": ["drive_upload", "drive_share"],
    },
    "email_and_schedule": {
        "description": "Send email and create calendar event",
        "chain": ["gmail_send", "calendar_create"],
    },
}

# Ambiguity patterns
AMBIGUITY_PATTERNS = {
    "multiple_recipients": re.compile(r"(?:send|mail).+?(?:to|cc).+?(?:,|and|or)", re.I),
    "multiple_files": re.compile(r"(?:file|files|document).+?(?:,|and|or)", re.I),
    "time_ambiguity": re.compile(r"(?:schedule|remind).+?(?:today|tomorrow|next|this|time)", re.I),
}


class DecisionEngine:
    """Lightweight AI reasoning for workflow decisions."""

    def __init__(self, memory_repo: SQLiteMemoryRepository | None = None):
        self.memory_repo = memory_repo or SQLiteMemoryRepository()

    def select_tools(
        self,
        user_intent: str,
        context: Dict[str, Any],
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Select tools needed to accomplish the user's intent.

        Returns list of {tool_name, confidence, reasoning}
        """
        tools = []

        # Rule-based tool selection (deterministic first)
        if self._matches_pattern(user_intent, ["send", "mail", "email"]):
            tools.append(
                {
                    "tool": "gmail_send",
                    "confidence": 0.95,
                    "reasoning": "User wants to send email",
                }
            )
            if self._mentions(user_intent, ["file", "document", "attach"]):
                tools.insert(0, {
                    "tool": "drive_search",
                    "confidence": 0.90,
                    "reasoning": "User wants to attach a file",
                })

        elif self._matches_pattern(user_intent, ["search", "find", "look for", "get"]):
            if self._matches_pattern(user_intent, ["file", "document"]):
                tools.append({
                    "tool": "drive_search",
                    "confidence": 0.95,
                    "reasoning": "User searching for files",
                })
            elif self._matches_pattern(user_intent, ["email", "mail", "message"]):
                tools.append({
                    "tool": "gmail_read",
                    "confidence": 0.95,
                    "reasoning": "User searching for emails",
                })

        elif self._matches_pattern(user_intent, ["create", "schedule", "add", "new"]):
            if self._matches_pattern(user_intent, ["event", "meeting", "appointment", "reminder"]):
                tools.append({
                    "tool": "calendar_create",
                    "confidence": 0.95,
                    "reasoning": "User wants to create calendar event",
                })
            elif self._matches_pattern(user_intent, ["email", "draft", "message"]):
                tools.append({
                    "tool": "gmail_draft",
                    "confidence": 0.95,
                    "reasoning": "User wants to draft email",
                })

        elif self._matches_pattern(user_intent, ["delete", "remove", "cancel"]):
            if self._matches_pattern(user_intent, ["event", "meeting"]):
                tools.append({
                    "tool": "calendar_delete",
                    "confidence": 0.95,
                    "reasoning": "User wants to delete calendar event",
                })

        elif self._matches_pattern(user_intent, ["share", "upload"]):
            tools.append({
                "tool": "drive_upload",
                "confidence": 0.90,
                "reasoning": "User wants to upload or share file",
            })
            if self._mentions(user_intent, ["share"]):
                tools.append({
                    "tool": "drive_share",
                    "confidence": 0.85,
                    "reasoning": "User wants to share file",
                })

        # If no deterministic match, use lightweight LLM reasoning
        if not tools and len(user_intent) > 5:
            tools = self._llm_select_tools(user_intent)

        return tools

    def resolve_parameters(
        self,
        tool_name: str,
        user_intent: str,
        context: Dict[str, Any],
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Resolve tool parameters from user intent and context.

        Returns dict of parameters for the tool.
        """
        params = {}

        # Extract from context first (memory, recent activity)
        recipient = self._resolve_recipient(user_intent, user_id)
        if recipient:
            params["to"] = recipient

        file_info = self._resolve_file_reference(user_intent, user_id)
        if file_info:
            params.update(file_info)

        time_info = self._resolve_time_reference(user_intent)
        if time_info:
            params.update(time_info)

        # Extract from text patterns
        subject = self._extract_subject(user_intent, tool_name)
        if subject:
            params["subject"] = subject

        body = self._extract_body(user_intent, tool_name)
        if body:
            params["body"] = body

        return params

    def detect_ambiguities(
        self,
        user_intent: str,
        context: Dict[str, Any],
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect ambiguities in user request that need clarification.

        Returns list of {ambiguity_type, question, options, severity}
        """
        ambiguities = []

        for amb_type, pattern in AMBIGUITY_PATTERNS.items():
            if pattern.search(user_intent):
                ambiguities.append({
                    "type": amb_type,
                    "severity": "medium",
                    "requires_resolution": True,
                })

        # Check recipient ambiguity
        if self._has_recipient_ambiguity(user_intent, user_id):
            ambiguities.append({
                "type": "recipient",
                "severity": "high",
                "requires_resolution": True,
                "question": "Who should I send this to?",
            })

        # Check file ambiguity
        if self._has_file_ambiguity(user_intent, user_id):
            ambiguities.append({
                "type": "file",
                "severity": "high",
                "requires_resolution": True,
                "question": "Which file should I use?",
            })

        return ambiguities

    def plan_tool_chain(
        self,
        tools: List[Dict[str, Any]],
        user_intent: str,
    ) -> List[Dict[str, Any]]:
        """
        Determine execution order for multiple tools.

        Returns ordered list of tools with dependency information.
        """
        if not tools:
            return []

        # Single tool - no ordering needed
        if len(tools) == 1:
            return [
                {
                    **tools[0],
                    "step": 1,
                    "depends_on": None,
                }
            ]

        # Multi-tool: check for known chains
        tool_names = [t["tool"] for t in tools]
        chain = self._find_applicable_chain(tool_names)
        if chain:
            return self._build_ordered_chain(chain, tools)

        # Default: topological sort by capability dependencies
        return self._topological_sort_tools(tools)

    def should_confirm(self, tool_name: str) -> bool:
        """Determine if a tool execution requires user confirmation."""
        capabilities = TOOL_CAPABILITIES.get(tool_name, {})
        return capabilities.get("risky", False)

    def can_retry(self, tool_name: str, attempt: int = 1, max_attempts: int = 2) -> bool:
        """Determine if a failed tool execution should be retried."""
        # Most tools can retry once on failure
        # Risky operations generally should not auto-retry
        return attempt < max_attempts and not TOOL_CAPABILITIES.get(tool_name, {}).get("risky", False)

    def fallback_strategy(
        self,
        failed_tool: str,
        user_intent: str,
        error: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest fallback action when a tool fails.

        Returns alternative tool or clarification needed.
        """
        # gmail_send fails -> draft instead
        if failed_tool == "gmail_send" and "draft" not in error.lower():
            return {
                "fallback_tool": "gmail_draft",
                "reason": "Send failed; suggest drafting instead",
            }

        # drive_share fails -> suggest retrieve instead
        if failed_tool == "drive_share":
            return {
                "fallback_tool": "drive_retrieve",
                "reason": "Cannot share; retrieving file info instead",
            }

        return None

    # ─────────────────── Private helpers ───────────────────

    def _matches_pattern(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords (case-insensitive)."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in keywords)

    def _mentions(self, text: str, keywords: List[str]) -> bool:
        """Alias for _matches_pattern."""
        return self._matches_pattern(text, keywords)

    def _resolve_recipient(self, user_intent: str, user_id: str) -> Optional[str]:
        """Try to resolve email recipient from intent and memory."""
        # Pattern: "to someone@email.com"
        match = re.search(r"to\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", user_intent, re.I)
        if match:
            return match.group(1)

        # Check memory for common recipients
        aliases = self.memory_repo.list_entity_aliases(user_id)
        for alias_entry in aliases:
            if self._mentions(user_intent, [alias_entry.get("alias", "")]):
                return alias_entry.get("actual_value")

        return None

    def _resolve_file_reference(self, user_intent: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Try to resolve file reference from intent and memory."""
        # Simple pattern: "resume", "report", "document"
        common_files = ["resume", "cv", "report", "proposal", "document"]
        for file_type in common_files:
            if file_type in user_intent.lower():
                return {
                    "file_type": file_type,
                    "search_hint": file_type,
                }
        return None

    def _resolve_time_reference(self, user_intent: str) -> Optional[Dict[str, Any]]:
        """Extract time references from intent."""
        # Simple time pattern matching
        if "tomorrow" in user_intent.lower():
            return {"time_hint": "tomorrow"}
        if "today" in user_intent.lower():
            return {"time_hint": "today"}
        if "next week" in user_intent.lower():
            return {"time_hint": "next_week"}
        return None

    def _extract_subject(self, user_intent: str, tool_name: str) -> Optional[str]:
        """Extract email subject from intent."""
        if tool_name not in ["gmail_send", "gmail_draft"]:
            return None

        # Pattern: "subject: ..." or just use first few words
        match = re.search(r"subject\s*[:=]\s*(.+?)(?:\.|$)", user_intent, re.I)
        if match:
            return match.group(1).strip()

        return None

    def _extract_body(self, user_intent: str, tool_name: str) -> Optional[str]:
        """Extract email body from intent."""
        if tool_name not in ["gmail_send", "gmail_draft"]:
            return None

        # For now, return None - body usually comes from separate elaboration
        return None

    def _has_recipient_ambiguity(self, user_intent: str, user_id: str) -> bool:
        """Check if recipient is ambiguous."""
        # Ambiguous if: no email pattern AND no alias match
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_intent)
        if email_match:
            return False

        aliases = self.memory_repo.list_entity_aliases(user_id)
        found_alias = any(self._mentions(user_intent, [a.get("alias", "")]) for a in aliases)
        return not found_alias

    def _has_file_ambiguity(self, user_intent: str, user_id: str) -> bool:
        """Check if file reference is ambiguous."""
        # Ambiguous if multiple file references without specifics
        file_matches = len(re.findall(r"(?:file|document|resume|report)", user_intent, re.I))
        return file_matches > 1

    def _find_applicable_chain(self, tool_names: List[str]) -> Optional[List[str]]:
        """Find a known tool chain matching the tools."""
        for chain_name, chain_config in TOOL_CHAINS.items():
            chain_tools = chain_config.get("chain", [])
            if all(t in tool_names for t in chain_tools):
                return chain_tools
        return None

    def _build_ordered_chain(
        self,
        chain: List[str],
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build ordered chain with dependency info."""
        result = []
        for step_num, tool_name in enumerate(chain, 1):
            tool_info = next((t for t in tools if t["tool"] == tool_name), None)
            if tool_info:
                result.append({
                    **tool_info,
                    "step": step_num,
                    "depends_on": chain[step_num - 2] if step_num > 1 else None,
                })
        return result

    def _topological_sort_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort tools by capability dependencies."""
        # Simple heuristic: read/search before send/modify
        def priority(tool):
            t = tool["tool"]
            if "search" in t or "read" in t or "retrieve" in t:
                return 0
            elif "draft" in t:
                return 1
            else:
                return 2

        sorted_tools = sorted(tools, key=priority)
        return [
            {**t, "step": i + 1, "depends_on": None}
            for i, t in enumerate(sorted_tools)
        ]

    def _llm_select_tools(self, user_intent: str) -> List[Dict[str, Any]]:
        """Use lightweight LLM reasoning for tool selection when rules don't match."""
        tool_descriptions = "\n".join([f"- {t}: {c}" for t, c in [
            ("gmail_send", "Send an email"),
            ("gmail_read", "Read/search emails"),
            ("calendar_create", "Create calendar event"),
            ("calendar_delete", "Delete calendar event"),
            ("drive_search", "Search for files"),
            ("drive_retrieve", "Get file details"),
            ("drive_upload", "Upload a file"),
            ("drive_share", "Share a file"),
        ]])

        prompt = f"""Given this user intent, which tools should be used?

User intent: {user_intent}

Available tools:
{tool_descriptions}

Respond with JSON: {{"tools": ["tool1", "tool2"], "reasoning": "..."}}
Only include tools that are clearly needed. Keep it concise."""

        try:
            response = call_llm(
                prompt=prompt,
                system_prompt="You are a tool selection agent. Return valid JSON only.",
            )

            import json
            result = json.loads(response)
            return [
                {
                    "tool": t,
                    "confidence": 0.70,
                    "reasoning": result.get("reasoning", "LLM selection"),
                }
                for t in result.get("tools", [])
            ]
        except Exception as e:
            LOGGER.warning(f"LLM tool selection failed: {e}")
            return []
