"""Lightweight planner agent for Phase 5.

Breaks user requests into multi-step workflow plans.
Deterministic-first planning with lightweight LLM assistance.

Example:
  "Find my resume and send it to HR"
  →
  1. Search Drive for resume
  2. Get file details
  3. Draft email to HR with attachment
  4. Send email
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from brain.llm_provider import call_llm
from memory.storage import SQLiteMemoryRepository
from agent.decision_engine import DecisionEngine

LOGGER = logging.getLogger(__name__)


class WorkflowStep:
    """Represents a single step in a workflow."""

    def __init__(
        self,
        step_number: int,
        tool_name: str,
        description: str,
        parameters: Dict[str, Any],
        requires_confirmation: bool = False,
        requires_clarification: bool = False,
        depends_on: Optional[int] = None,
    ):
        self.step_number = step_number
        self.tool_name = tool_name
        self.description = description
        self.parameters = parameters
        self.requires_confirmation = requires_confirmation
        self.requires_clarification = requires_clarification
        self.depends_on = depends_on

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "tool_name": self.tool_name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation,
            "requires_clarification": self.requires_clarification,
            "depends_on": self.depends_on,
        }


class WorkflowPlan:
    """Complete execution plan for a user request."""

    def __init__(
        self,
        plan_id: str,
        user_id: str,
        user_intent: str,
        steps: List[WorkflowStep],
        ambiguities: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None,
    ):
        self.plan_id = plan_id
        self.user_id = user_id
        self.user_intent = user_intent
        self.steps = steps
        self.ambiguities = ambiguities or []
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "user_intent": self.user_intent,
            "steps": [s.to_dict() for s in self.steps],
            "ambiguities": self.ambiguities,
            "context": self.context,
            "total_steps": len(self.steps),
            "has_risky_steps": any(s.requires_confirmation for s in self.steps),
            "requires_clarification": any(a.get("requires_resolution") for a in self.ambiguities),
        }

    def summarize(self) -> str:
        """Generate human-readable plan summary."""
        lines = [f"Plan: {self.user_intent}\n"]
        for step in self.steps:
            lines.append(f"{step.step_number}. {step.description}")
            if step.requires_confirmation:
                lines.append("   ⚠️ Requires confirmation")
            if step.requires_clarification:
                lines.append("   ❓ Needs clarification")
        return "\n".join(lines)


class TaskPlanner:
    """Multi-step task planner for Phase 5 workflows."""

    def __init__(self, memory_repo: SQLiteMemoryRepository | None = None):
        self.memory_repo = memory_repo or SQLiteMemoryRepository()
        self.decision_engine = DecisionEngine(memory_repo)

    def plan(
        self,
        user_intent: str,
        user_id: str,
        plan_id: str,
        context: Dict[str, Any] = None,
    ) -> WorkflowPlan:
        """
        Create an execution plan for a user request.

        Returns WorkflowPlan with ordered steps and requirements.
        """
        context = context or {}

        # 1. Detect ambiguities first
        ambiguities = self.decision_engine.detect_ambiguities(user_intent, context, user_id)
        LOGGER.info(f"Detected {len(ambiguities)} ambiguities for intent: {user_intent}")

        # 2. Select tools needed
        tools = self.decision_engine.select_tools(user_intent, context, user_id)
        LOGGER.info(f"Selected {len(tools)} tools: {[t['tool'] for t in tools]}")

        # If no tools found, ask for clarification
        if not tools:
            plan = WorkflowPlan(
                plan_id=plan_id,
                user_id=user_id,
                user_intent=user_intent,
                steps=[],
                ambiguities=[{
                    "type": "unclear_intent",
                    "severity": "high",
                    "requires_resolution": True,
                    "question": "I'm not sure what you'd like me to do. Could you clarify?",
                }],
                context=context,
            )
            return plan

        # 3. Plan tool execution chain
        ordered_tools = self.decision_engine.plan_tool_chain(tools, user_intent)

        # 4. Build workflow steps
        steps: List[WorkflowStep] = []
        for tool_info in ordered_tools:
            step_number = tool_info.get("step", len(steps) + 1)
            tool_name = tool_info["tool"]

            # Resolve parameters
            parameters = self.decision_engine.resolve_parameters(
                tool_name=tool_name,
                user_intent=user_intent,
                context=context,
                user_id=user_id,
            )

            # Check if risky
            requires_confirmation = self.decision_engine.should_confirm(tool_name)

            # Check if ambiguous
            requires_clarification = any(
                a.get("type") == tool_name or
                (a.get("type") in ["recipient", "file"] and not parameters.get(self._param_for_ambiguity(a["type"])))
                for a in ambiguities
            )

            # Build step description
            description = self._build_step_description(tool_name, parameters)

            step = WorkflowStep(
                step_number=step_number,
                tool_name=tool_name,
                description=description,
                parameters=parameters,
                requires_confirmation=requires_confirmation,
                requires_clarification=requires_clarification,
                depends_on=tool_info.get("depends_on"),
            )
            steps.append(step)

        # 5. Return complete plan
        plan = WorkflowPlan(
            plan_id=plan_id,
            user_id=user_id,
            user_intent=user_intent,
            steps=steps,
            ambiguities=ambiguities,
            context=context,
        )

        LOGGER.info(f"Created plan with {len(steps)} steps: {plan.summarize()}")
        return plan

    def refine_with_clarifications(
        self,
        plan: WorkflowPlan,
        clarifications: Dict[str, str],
    ) -> WorkflowPlan:
        """
        Refine a plan based on user's clarification responses.

        Returns updated WorkflowPlan.
        """
        LOGGER.info(f"Refining plan with clarifications: {clarifications}")

        # Update context with clarifications
        plan.context.update(clarifications)

        # Recalculate steps with new context
        updated_steps = []
        for step in plan.steps:
            # Re-resolve parameters with clarifications
            new_params = self.decision_engine.resolve_parameters(
                tool_name=step.tool_name,
                user_intent=plan.user_intent,
                context=plan.context,
                user_id=plan.user_id,
            )

            updated_step = WorkflowStep(
                step_number=step.step_number,
                tool_name=step.tool_name,
                description=self._build_step_description(step.tool_name, new_params),
                parameters=new_params,
                requires_confirmation=step.requires_confirmation,
                requires_clarification=False,  # Marked as resolved
                depends_on=step.depends_on,
            )
            updated_steps.append(updated_step)

        plan.steps = updated_steps
        plan.ambiguities = [a for a in plan.ambiguities if a.get("requires_resolution")]

        return plan

    def estimate_complexity(self, plan: WorkflowPlan) -> str:
        """Estimate workflow complexity: simple, moderate, complex."""
        step_count = len(plan.steps)
        has_risky = plan.to_dict()["has_risky_steps"]
        has_ambiguities = plan.to_dict()["requires_clarification"]

        score = step_count
        if has_risky:
            score += 2
        if has_ambiguities:
            score += 1

        if score <= 2:
            return "simple"
        elif score <= 5:
            return "moderate"
        else:
            return "complex"

    # ─────────────── Private helpers ───────────────────

    def _param_for_ambiguity(self, ambiguity_type: str) -> str:
        """Map ambiguity type to parameter key."""
        mapping = {
            "recipient": "to",
            "file": "file_id",
            "time": "start_time",
        }
        return mapping.get(ambiguity_type, "")

    def _build_step_description(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Build human-readable description of a workflow step."""
        descriptions = {
            "gmail_send": lambda p: f"Send email to {p.get('to', 'recipient')} about '{p.get('subject', 'matter')}'",
            "gmail_draft": lambda p: f"Draft email to {p.get('to', 'recipient')}",
            "gmail_read": lambda p: f"Read/search emails (limit: {p.get('max_results', 5)})",
            "calendar_create": lambda p: f"Create event '{p.get('summary', 'event')}' at {p.get('start_time', 'specified time')}",
            "calendar_delete": lambda p: f"Delete event '{p.get('event_id', 'event')}'",
            "calendar_edit": lambda p: f"Edit event",
            "drive_search": lambda p: f"Search Drive for '{p.get('search_hint', 'files')}'",
            "drive_retrieve": lambda p: f"Get file details",
            "drive_upload": lambda p: f"Upload file '{p.get('file_path', 'file')}'",
            "drive_share": lambda p: f"Share file with {p.get('email', 'recipient')}",
        }

        builder = descriptions.get(tool_name)
        if builder:
            try:
                return builder(parameters)
            except Exception as e:
                LOGGER.warning(f"Failed to build description for {tool_name}: {e}")

        return f"Execute {tool_name}"

    def _llm_plan_steps(self, user_intent: str, tools: List[str]) -> List[Dict[str, Any]]:
        """Use LLM for complex step planning when needed."""
        tool_list = ", ".join(tools)
        prompt = f"""Break down this task into execution steps.

Task: {user_intent}
Available tools: {tool_list}

Return JSON: {{"steps": [{{"tool": "tool_name", "description": "...", "parameters": {{}}}}]}}
Keep steps simple and deterministic."""

        try:
            response = call_llm(
                prompt=prompt,
                system_prompt="You are a workflow planner. Return valid JSON only.",
            )
            result = json.loads(response)
            return result.get("steps", [])
        except Exception as e:
            LOGGER.warning(f"LLM step planning failed: {e}")
            return []
