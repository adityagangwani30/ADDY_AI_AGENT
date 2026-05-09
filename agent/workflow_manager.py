"""Workflow orchestration manager for Phase 5.

Coordinates:
- Task planning
- Workflow execution
- Clarification and confirmation flows
- State persistence and recovery
- Integration with existing assistant

This is the main entry point for Phase 5 autonomous workflows.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from agent.planner import TaskPlanner, WorkflowPlan
from agent.workflow_executor import WorkflowExecutor
from memory.storage import SQLiteMemoryRepository
from automation.workflow_db import WorkflowDatabase

LOGGER = logging.getLogger(__name__)


class WorkflowManager:
    """Orchestrates Phase 5 autonomous workflows."""

    def __init__(
        self,
        db_path: str,
        memory_repo: SQLiteMemoryRepository | None = None,
    ):
        self.db = WorkflowDatabase(db_path)
        self.memory_repo = memory_repo or SQLiteMemoryRepository()
        self.planner = TaskPlanner(memory_repo)
        self.executor = WorkflowExecutor(db_path, memory_repo)
        self.active_workflows = {}  # workflow_run_id -> workflow state

    def start_workflow(
        self,
        user_intent: str,
        user_id: str,
        account: str,
        context: Dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Start a new autonomous workflow.

        Returns workflow result with status and details.
        """
        request_id = request_id or str(uuid.uuid4())
        plan_id = str(uuid.uuid4())
        context = context or {}

        LOGGER.info(f"Starting workflow: user_intent='{user_intent}' user_id={user_id}")

        # Phase 1: Plan the workflow
        plan = self.planner.plan(
            user_intent=user_intent,
            user_id=user_id,
            plan_id=plan_id,
            context=context,
        )

        if not plan.steps and plan.ambiguities:
            # Can't proceed without clarification
            LOGGER.info(f"Workflow requires clarification before proceeding")
            return {
                "status": "clarification_required",
                "plan_id": plan_id,
                "message": "I need clarification before I can proceed.",
                "ambiguities": plan.ambiguities,
                "request_id": request_id,
            }

        # Phase 2: Estimate complexity and provide preview
        complexity = self.planner.estimate_complexity(plan)
        preview_message = f"I'll help you with this in {len(plan.steps)} steps:\n\n{plan.summarize()}"

        LOGGER.info(f"Workflow plan: {complexity} complexity, {len(plan.steps)} steps")

        # Phase 3: Execute the workflow
        status, execution_result = self.executor.execute(
            plan=plan,
            account=account,
            request_id=request_id,
        )

        workflow_run_id = execution_result.get("workflow_run_id")

        # Store active workflow
        if workflow_run_id:
            self.active_workflows[workflow_run_id] = {
                "plan": plan,
                "status": status,
                "created_at": plan_id,
            }

        # Return result
        if status == "completed":
            self.memory_repo.add_conversation(
                user_id,
                "assistant",
                f"✅ Completed: {user_intent}"
            )
            return {
                "status": "completed",
                "message": f"✅ Successfully completed: {user_intent}",
                "steps_executed": execution_result.get("steps_executed", 0),
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
            }

        elif status == "confirmation_required":
            action = execution_result.get("pending_confirmation_action", "This action")
            return {
                "status": "confirmation_required",
                "message": f"⚠️ {action}\n\nPlease confirm by replying YES or NO.",
                "confirmation_id": execution_result.get("pending_confirmation_id"),
                "action": action,
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
            }

        elif status == "clarification_required":
            clarifications = execution_result.get("clarifications", [])
            questions = [c.get("question") for c in clarifications]
            return {
                "status": "clarification_required",
                "message": "I need clarification to proceed:\n\n" + "\n".join(questions),
                "clarifications": clarifications,
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
            }

        else:  # error
            error_msg = execution_result.get("error") or execution_result.get("message", "Unknown error")
            LOGGER.error(f"Workflow failed: {error_msg}")
            return {
                "status": "error",
                "message": f"❌ Something went wrong: {error_msg}",
                "error": error_msg,
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
            }

    def confirm_action(
        self,
        workflow_run_id: int,
        confirmation_id: int,
        confirmed: bool,
        account: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Confirm or reject a pending workflow action.

        Returns updated workflow result.
        """
        LOGGER.info(f"Confirmation received: workflow_id={workflow_run_id}, confirmed={confirmed}")

        # Record confirmation
        self.db.confirm_action(confirmation_id, confirmed)

        if not confirmed:
            # Cancel workflow
            self.executor.cancel(workflow_run_id)
            return {
                "status": "cancelled",
                "message": "Action cancelled.",
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
            }

        # Resume workflow
        status, result = self.executor.resume(
            workflow_run_id=workflow_run_id,
            account=account,
            request_id=request_id,
        )

        if status == "completed":
            return {
                "status": "completed",
                "message": "✅ Workflow completed.",
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
            }
        elif status == "error":
            return {
                "status": "error",
                "message": f"❌ Workflow failed: {result.get('error')}",
                "error": result.get("error"),
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
            }
        else:
            # Still waiting for something
            return {
                "status": status,
                "message": f"Workflow status: {status}",
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
            }

    def respond_to_clarification(
        self,
        workflow_run_id: int,
        clarification_id: int,
        response: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Respond to a clarification request.

        Returns updated plan or continues workflow.
        """
        LOGGER.info(f"Clarification response: workflow_id={workflow_run_id}, response={response}")

        # Record response
        self.db.respond_to_clarification(clarification_id, response)

        # Get workflow
        workflow_run = self.db.get_workflow_run(workflow_run_id)
        if not workflow_run:
            return {
                "status": "error",
                "message": "Workflow not found",
            }

        # Retrieve cached plan
        cached = self.active_workflows.get(workflow_run_id)
        if not cached:
            return {
                "status": "error",
                "message": "Workflow context not available",
            }

        plan = cached["plan"]

        # Update plan with clarification
        plan = self.planner.refine_with_clarifications(
            plan,
            {f"clarification_{clarification_id}": response}
        )

        # Check if all ambiguities resolved
        if any(a.get("requires_resolution") for a in plan.ambiguities):
            return {
                "status": "clarification_required",
                "message": "I still need clarification on a few more things.",
                "ambiguities": plan.ambiguities,
            }

        # All clarified - return to user for preview/confirmation
        return {
            "status": "plan_ready",
            "message": "Got it! Here's my updated plan:\n\n" + plan.summarize(),
            "workflow_run_id": workflow_run_id,
        }

    def list_workflows(self, user_id: str, status: Optional[str] = None) -> list:
        """List workflows for a user."""
        return self.db.list_workflow_runs(user_id, status=status)

    def get_workflow_details(self, workflow_run_id: int) -> Dict[str, Any]:
        """Get detailed information about a workflow."""
        run = self.db.get_workflow_run(workflow_run_id)
        if not run:
            return {}

        steps = self.db.get_workflow_steps(workflow_run_id)

        return {
            "run": run,
            "steps": steps,
            "workflow_type": run.get("workflow_type"),
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at"),
        }

    def cancel_workflow(self, workflow_run_id: int) -> Dict[str, Any]:
        """Cancel an active workflow."""
        LOGGER.info(f"Cancelling workflow: {workflow_run_id}")
        self.executor.cancel(workflow_run_id)

        return {
            "status": "cancelled",
            "message": "Workflow cancelled.",
            "workflow_run_id": workflow_run_id,
        }

    def cleanup_old_workflows(self, days: int = 30) -> int:
        """Clean up completed workflows older than specified days."""
        removed = self.db.cleanup_old_workflows(days)
        LOGGER.info(f"Cleaned up {removed} old workflows")
        return removed
