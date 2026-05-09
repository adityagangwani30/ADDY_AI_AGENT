"""Workflow execution engine for Phase 5.

Handles:
- Sequential execution of workflow steps
- Conditional branching
- Context passing between tools
- Retry logic
- Workflow cancellation
- State persistence
- Resumable execution
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.tool_executor import ToolExecutor
from agent.decision_engine import DecisionEngine
from automation.workflow_db import WorkflowDatabase
from memory.storage import SQLiteMemoryRepository
from agent.planner import WorkflowPlan

LOGGER = logging.getLogger(__name__)

EXECUTION_TIMEOUT_SECONDS = 300  # 5 minute timeout per workflow
STEP_TIMEOUT_SECONDS = 30  # 30 seconds per step


class ExecutionContext:
    """Runtime context for workflow execution."""

    def __init__(self, user_id: str, plan: WorkflowPlan):
        self.user_id = user_id
        self.plan = plan
        self.step_results = {}  # step_number -> result
        self.step_errors = {}  # step_number -> error
        self.variables = {}  # shared variables between steps
        self.clarifications = {}  # clarification_id -> response
        self.confirmations = {}  # confirmation_id -> confirmed bool
        self.started_at = time.time()
        self.last_activity = time.time()

    def get_step_result(self, step_number: int) -> Optional[Any]:
        """Get result from a previous step."""
        return self.step_results.get(step_number)

    def set_step_result(self, step_number: int, result: Any) -> None:
        """Store result from a step execution."""
        self.step_results[step_number] = result
        self.last_activity = time.time()

    def set_step_error(self, step_number: int, error: str) -> None:
        """Store error from a step execution."""
        self.step_errors[step_number] = error
        self.last_activity = time.time()

    def is_timed_out(self) -> bool:
        """Check if execution has exceeded timeout."""
        elapsed = time.time() - self.started_at
        return elapsed > EXECUTION_TIMEOUT_SECONDS

    def is_idle(self, idle_threshold: int = 60) -> bool:
        """Check if execution is idle (waiting for user input)."""
        idle_time = time.time() - self.last_activity
        return idle_time > idle_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for persistence."""
        return {
            "user_id": self.user_id,
            "step_results": self.step_results,
            "step_errors": self.step_errors,
            "variables": self.variables,
            "clarifications": self.clarifications,
            "confirmations": self.confirmations,
        }


class WorkflowExecutor:
    """Executes workflow plans with state management."""

    def __init__(
        self,
        db_path: str,
        memory_repo: SQLiteMemoryRepository | None = None,
    ):
        self.db = WorkflowDatabase(db_path)
        self.memory_repo = memory_repo or SQLiteMemoryRepository()
        self.tool_executor = ToolExecutor(self.memory_repo)
        self.decision_engine = DecisionEngine(self.memory_repo)

    def execute(
        self,
        plan: WorkflowPlan,
        account: str,
        request_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute a workflow plan.

        Returns (status, results) where status is one of:
        - "completed": workflow finished successfully
        - "confirmation_required": waiting for user confirmation
        - "clarification_required": waiting for user to clarify ambiguities
        - "error": workflow failed
        """
        LOGGER.info(f"Starting workflow execution: plan_id={plan.plan_id}")

        # Create workflow run
        workflow_run_id = self.db.create_workflow_run(
            user_id=plan.user_id,
            workflow_type="multi_step",
            workflow_state={"plan": plan.to_dict()},
            planned_steps=[s.to_dict() for s in plan.steps],
        )

        # Create execution context
        context = ExecutionContext(plan.user_id, plan)

        # Handle ambiguities first
        if plan.ambiguities and any(a.get("requires_resolution") for a in plan.ambiguities):
            LOGGER.info(f"Workflow has unresolved ambiguities: {len(plan.ambiguities)}")
            return self._handle_ambiguities(workflow_run_id, plan, context)

        # Execute steps
        return self._execute_steps(
            workflow_run_id=workflow_run_id,
            plan=plan,
            context=context,
            account=account,
            request_id=request_id,
        )

    def resume(
        self,
        workflow_run_id: int,
        account: str,
        request_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Resume a paused workflow (after confirmation or clarification)."""
        LOGGER.info(f"Resuming workflow: workflow_run_id={workflow_run_id}")

        workflow_run = self.db.get_workflow_run(workflow_run_id)
        if not workflow_run:
            return "error", {"message": "Workflow not found"}

        if workflow_run["status"] not in ["confirmation_required", "clarification_required"]:
            return "error", {"message": f"Cannot resume workflow in status: {workflow_run['status']}"}

        # Recreate plan and context
        plan_data = workflow_run.get("workflow_state", {}).get("plan", {})
        plan = WorkflowPlan(
            plan_id=workflow_run["id"],
            user_id=workflow_run["user_id"],
            user_intent=plan_data.get("user_intent", ""),
            steps=plan_data.get("steps", []),
            ambiguities=plan_data.get("ambiguities", []),
            context=plan_data.get("context", {}),
        )

        context = ExecutionContext(plan.user_id, plan)
        context.step_results = workflow_run.get("execution_context", {}).get("step_results", {})
        context.step_errors = workflow_run.get("execution_context", {}).get("step_errors", {})

        # Continue execution
        return self._execute_steps(
            workflow_run_id=workflow_run_id,
            plan=plan,
            context=context,
            account=account,
            request_id=request_id,
        )

    def cancel(self, workflow_run_id: int) -> None:
        """Cancel a running workflow."""
        LOGGER.info(f"Cancelling workflow: workflow_run_id={workflow_run_id}")
        self.db.cancel_workflow(workflow_run_id)
        self.db.update_workflow_status(workflow_run_id, "cancelled")

    # ────────────── Private helpers ──────────────────

    def _execute_steps(
        self,
        workflow_run_id: int,
        plan: WorkflowPlan,
        context: ExecutionContext,
        account: str,
        request_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute workflow steps sequentially."""
        self.db.update_workflow_status(workflow_run_id, "running", current_step=0)

        results = {"workflow_run_id": workflow_run_id, "steps_executed": 0, "steps_failed": 0}

        for step in plan.steps:
            # Check for cancellation
            workflow_run = self.db.get_workflow_run(workflow_run_id)
            if workflow_run and workflow_run.get("cancel_requested"):
                LOGGER.info("Workflow cancellation requested")
                self.db.update_workflow_status(workflow_run_id, "cancelled")
                return "error", {"message": "Workflow cancelled by user", **results}

            # Check timeout
            if context.is_timed_out():
                LOGGER.warning("Workflow execution timeout")
                self.db.update_workflow_status(
                    workflow_run_id,
                    "failed",
                    error_message="Execution timeout",
                )
                return "error", {"message": "Workflow timeout", **results}

            # Handle risky steps (require confirmation)
            if step.requires_confirmation:
                LOGGER.info(f"Step {step.step_number} requires confirmation")
                confirmation_id = self.db.add_confirmation(
                    workflow_run_id=workflow_run_id,
                    step_number=step.step_number,
                    action_description=step.description,
                    required=True,
                )
                self.db.update_workflow_status(workflow_run_id, "confirmation_required", current_step=step.step_number)
                results["pending_confirmation_id"] = confirmation_id
                results["pending_confirmation_action"] = step.description
                return "confirmation_required", results

            # Execute step
            LOGGER.info(f"Executing step {step.step_number}: {step.tool_name}")
            step_id = self.db.add_workflow_step(
                workflow_run_id=workflow_run_id,
                step_number=step.step_number,
                tool_name=step.tool_name,
                parameters=step.parameters,
            )

            status, result = self._execute_step(
                step_id=step_id,
                step=step,
                context=context,
                account=account,
                request_id=request_id,
            )

            if status == "success":
                context.set_step_result(step.step_number, result)
                results["steps_executed"] += 1
            else:
                context.set_step_error(step.step_number, result.get("error", "Unknown error"))
                results["steps_failed"] += 1

                # Check if we should retry
                if self.decision_engine.can_retry(step.tool_name, attempt=1):
                    LOGGER.info(f"Retrying step {step.step_number}")
                    status, result = self._execute_step(
                        step_id=step_id,
                        step=step,
                        context=context,
                        account=account,
                        request_id=request_id,
                    )
                    if status == "success":
                        context.set_step_result(step.step_number, result)
                        results["steps_executed"] += 1
                        results["steps_failed"] -= 1

                # If still failed and not risky, continue; otherwise abort
                if status != "success" and step.requires_confirmation:
                    LOGGER.warning(f"Critical step {step.step_number} failed: {result.get('error')}")
                    self.db.update_workflow_status(
                        workflow_run_id,
                        "failed",
                        error_message=result.get("error"),
                    )
                    return "error", {**results, "error": result.get("error")}

        # All steps completed successfully
        LOGGER.info(f"Workflow completed: {results}")
        self.db.update_workflow_status(workflow_run_id, "completed")
        return "completed", results

    def _execute_step(
        self,
        step_id: int,
        step: Any,
        context: ExecutionContext,
        account: str,
        request_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute a single workflow step."""
        started = time.time()

        try:
            # Update step status
            self.db.update_workflow_step(step_id, "running")

            # Execute tool
            result = self.tool_executor.execute(
                intent=step.tool_name,
                account=account,
                parameters=step.parameters or {},
                request_id=request_id,
            )

            elapsed_ms = int((time.time() - started) * 1000)

            if result.get("ok"):
                self.db.update_workflow_step(
                    step_id,
                    "completed",
                    result={"data": result.get("result"), "latency_ms": elapsed_ms},
                )
                LOGGER.info(f"Step {step.step_number} completed in {elapsed_ms}ms")
                return "success", result

            else:
                error = result.get("error", "Tool execution failed")
                self.db.update_workflow_step(step_id, "failed", error=error)
                LOGGER.error(f"Step {step.step_number} failed: {error}")
                return "failed", {"error": error}

        except Exception as e:
            error = f"Step execution exception: {str(e)}"
            self.db.update_workflow_step(step_id, "failed", error=error)
            LOGGER.exception(f"Step {step.step_number} exception")
            return "failed", {"error": error}

    def _handle_ambiguities(
        self,
        workflow_run_id: int,
        plan: WorkflowPlan,
        context: ExecutionContext,
    ) -> Tuple[str, Dict[str, Any]]:
        """Create clarification requests for ambiguous steps."""
        LOGGER.info(f"Handling {len(plan.ambiguities)} ambiguities")

        clarifications = []
        for amb in plan.ambiguities:
            if not amb.get("requires_resolution"):
                continue

            clarification_id = self.db.add_clarification(
                workflow_run_id=workflow_run_id,
                step_number=0,
                clarification_type=amb.get("type", "unknown"),
                question=amb.get("question", "Please clarify"),
                options=amb.get("options"),
            )
            clarifications.append({
                "id": clarification_id,
                "type": amb.get("type"),
                "question": amb.get("question"),
                "options": amb.get("options"),
            })

        self.db.update_workflow_status(workflow_run_id, "clarification_required", current_step=0)

        return "clarification_required", {
            "workflow_run_id": workflow_run_id,
            "clarifications": clarifications,
        }
