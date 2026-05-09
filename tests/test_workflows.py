"""Tests for Phase 5 autonomous workflow system.

Covers:
- Task planning
- Workflow execution
- Tool chaining
- Clarification/confirmation flows
- Workflow persistence
- Error recovery
"""
import unittest
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from agent.planner import TaskPlanner, WorkflowPlan, WorkflowStep
from agent.decision_engine import DecisionEngine
from agent.workflow_executor import WorkflowExecutor, ExecutionContext
from agent.workflow_manager import WorkflowManager
from automation.workflow_db import WorkflowDatabase
from memory.storage import SQLiteMemoryRepository


class TestDecisionEngine(unittest.TestCase):
    """Test lightweight decision engine."""

    def setUp(self):
        self.memory_repo = Mock(spec=SQLiteMemoryRepository)
        self.memory_repo.list_entity_aliases.return_value = []
        self.engine = DecisionEngine(self.memory_repo)

    def test_select_tools_send_email(self):
        """Should detect email sending intent."""
        tools = self.engine.select_tools(
            user_intent="Send an email to john@example.com",
            context={},
            user_id="user123"
        )
        self.assertTrue(any(t["tool"] == "gmail_send" for t in tools))

    def test_select_tools_find_file(self):
        """Should detect file search intent."""
        tools = self.engine.select_tools(
            user_intent="Find my resume",
            context={},
            user_id="user123"
        )
        self.assertTrue(any(t["tool"] == "drive_search" for t in tools))

    def test_detect_ambiguities_multiple_recipients(self):
        """Should detect when multiple recipients mentioned."""
        ambiguities = self.engine.detect_ambiguities(
            user_intent="Send to john and jane",
            context={},
            user_id="user123"
        )
        self.assertTrue(len(ambiguities) > 0)

    def test_should_confirm_risky_operations(self):
        """Should require confirmation for risky tools."""
        self.assertTrue(self.engine.should_confirm("gmail_send"))
        self.assertTrue(self.engine.should_confirm("calendar_delete"))
        self.assertFalse(self.engine.should_confirm("gmail_read"))

    def test_can_retry_non_risky_operations(self):
        """Should allow retries for non-risky operations."""
        self.assertTrue(self.engine.can_retry("gmail_read", attempt=1))
        self.assertFalse(self.engine.can_retry("gmail_send", attempt=1))


class TestTaskPlanner(unittest.TestCase):
    """Test task planner."""

    def setUp(self):
        self.memory_repo = Mock(spec=SQLiteMemoryRepository)
        self.memory_repo.list_entity_aliases.return_value = []
        self.planner = TaskPlanner(self.memory_repo)

    def test_plan_simple_task(self):
        """Should create simple workflow plan."""
        plan = self.planner.plan(
            user_intent="Send email to test@example.com",
            user_id="user123",
            plan_id="plan123"
        )
        self.assertIsInstance(plan, WorkflowPlan)
        self.assertTrue(len(plan.steps) > 0)
        self.assertTrue(any(s.tool_name == "gmail_send" for s in plan.steps))

    def test_plan_complex_task(self):
        """Should create multi-step workflow plan."""
        plan = self.planner.plan(
            user_intent="Find my resume and send it to HR",
            user_id="user123",
            plan_id="plan123"
        )
        self.assertTrue(len(plan.steps) >= 2)
        # Should have search before send
        search_steps = [s for s in plan.steps if "search" in s.tool_name or "retrieve" in s.tool_name]
        send_steps = [s for s in plan.steps if "send" in s.tool_name]
        self.assertTrue(len(search_steps) > 0)
        self.assertTrue(len(send_steps) > 0)

    def test_plan_with_ambiguities(self):
        """Should identify ambiguous requests."""
        plan = self.planner.plan(
            user_intent="Send resume to someone",
            user_id="user123",
            plan_id="plan123"
        )
        # Should have ambiguities when recipient is vague
        self.assertTrue(any(a.get("requires_resolution") for a in plan.ambiguities))

    def test_estimate_complexity(self):
        """Should estimate plan complexity."""
        # Simple plan
        simple_plan = self.planner.plan(
            user_intent="List my emails",
            user_id="user123",
            plan_id="plan1"
        )
        self.assertEqual(self.planner.estimate_complexity(simple_plan), "simple")

        # Complex plan (multiple steps + risky)
        complex_plan = self.planner.plan(
            user_intent="Find resume and send to multiple people",
            user_id="user123",
            plan_id="plan2"
        )
        complexity = self.planner.estimate_complexity(complex_plan)
        self.assertIn(complexity, ["simple", "moderate", "complex"])


class TestWorkflowDatabase(unittest.TestCase):
    """Test workflow persistence."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db = WorkflowDatabase(self.temp_db.name)

    def tearDown(self):
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    def test_create_workflow_run(self):
        """Should create workflow run."""
        run_id = self.db.create_workflow_run(
            user_id="user123",
            workflow_type="test",
            workflow_state={"test": "data"},
            planned_steps=[{"step": 1, "tool": "test"}]
        )
        self.assertIsNotNone(run_id)
        self.assertGreater(run_id, 0)

    def test_get_workflow_run(self):
        """Should retrieve workflow run."""
        run_id = self.db.create_workflow_run(
            user_id="user123",
            workflow_type="test",
            workflow_state={"key": "value"},
            planned_steps=[]
        )
        run = self.db.get_workflow_run(run_id)
        self.assertEqual(run["user_id"], "user123")
        self.assertEqual(run["workflow_type"], "test")

    def test_workflow_steps(self):
        """Should manage workflow steps."""
        run_id = self.db.create_workflow_run(
            user_id="user123",
            workflow_type="test",
            workflow_state={},
            planned_steps=[]
        )
        step_id = self.db.add_workflow_step(
            workflow_run_id=run_id,
            step_number=1,
            tool_name="gmail_send",
            parameters={"to": "test@example.com"}
        )
        self.assertIsNotNone(step_id)

        steps = self.db.get_workflow_steps(run_id)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["tool_name"], "gmail_send")

    def test_workflow_status_update(self):
        """Should update workflow status."""
        run_id = self.db.create_workflow_run(
            user_id="user123",
            workflow_type="test",
            workflow_state={},
            planned_steps=[]
        )
        self.db.update_workflow_status(run_id, "running")
        run = self.db.get_workflow_run(run_id)
        self.assertEqual(run["status"], "running")

    def test_confirmation_flow(self):
        """Should handle confirmation requests."""
        run_id = self.db.create_workflow_run(
            user_id="user123",
            workflow_type="test",
            workflow_state={},
            planned_steps=[]
        )
        conf_id = self.db.add_confirmation(
            workflow_run_id=run_id,
            step_number=1,
            action_description="Send email"
        )
        self.assertIsNotNone(conf_id)

        self.db.confirm_action(conf_id, True)
        # Verify update (should not raise)

    def test_clarification_flow(self):
        """Should handle clarification requests."""
        run_id = self.db.create_workflow_run(
            user_id="user123",
            workflow_type="test",
            workflow_state={},
            planned_steps=[]
        )
        clar_id = self.db.add_clarification(
            workflow_run_id=run_id,
            step_number=1,
            clarification_type="recipient",
            question="Who should I send this to?"
        )
        self.assertIsNotNone(clar_id)

        self.db.respond_to_clarification(clar_id, "john@example.com")


class TestExecutionContext(unittest.TestCase):
    """Test execution context."""

    def setUp(self):
        plan = WorkflowPlan(
            plan_id="plan1",
            user_id="user123",
            user_intent="test",
            steps=[]
        )
        self.context = ExecutionContext("user123", plan)

    def test_store_result(self):
        """Should store step results."""
        self.context.set_step_result(1, {"data": "result1"})
        result = self.context.get_step_result(1)
        self.assertEqual(result, {"data": "result1"})

    def test_timeout_detection(self):
        """Should detect timeout."""
        self.assertFalse(self.context.is_timed_out())
        # (Can't easily test timeout without mocking time)


class TestWorkflowExecutor(unittest.TestCase):
    """Test workflow executor."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.memory_repo = Mock(spec=SQLiteMemoryRepository)
        self.executor = WorkflowExecutor(self.temp_db.name, self.memory_repo)

    def tearDown(self):
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    @patch('agent.tool_executor.ToolExecutor.execute')
    def test_execute_simple_workflow(self, mock_execute):
        """Should execute simple workflow."""
        mock_execute.return_value = {
            "ok": True,
            "result": {"message": "success"}
        }

        plan = WorkflowPlan(
            plan_id="plan1",
            user_id="user123",
            user_intent="Send email",
            steps=[
                WorkflowStep(
                    step_number=1,
                    tool_name="gmail_send",
                    description="Send email",
                    parameters={"to": "test@example.com"}
                )
            ]
        )

        status, result = self.executor.execute(
            plan=plan,
            account="user123",
            request_id="req123"
        )

        self.assertEqual(status, "completed")
        self.assertGreater(result.get("steps_executed", 0), 0)


class TestWorkflowManager(unittest.TestCase):
    """Test workflow orchestration manager."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.memory_repo = Mock(spec=SQLiteMemoryRepository)
        self.memory_repo.list_entity_aliases.return_value = []
        self.manager = WorkflowManager(self.temp_db.name, self.memory_repo)

    def tearDown(self):
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    @patch('agent.workflow_executor.WorkflowExecutor.execute')
    def test_start_workflow(self, mock_execute):
        """Should start new workflow."""
        mock_execute.return_value = ("completed", {
            "workflow_run_id": 1,
            "steps_executed": 1,
            "steps_failed": 0
        })

        result = self.manager.start_workflow(
            user_intent="Send email",
            user_id="user123",
            account="user123"
        )

        self.assertEqual(result["status"], "completed")

    def test_list_workflows(self):
        """Should list workflows."""
        workflows = self.manager.list_workflows("user123")
        self.assertIsInstance(workflows, list)

    def test_cancel_workflow(self):
        """Should cancel workflow."""
        # Create a workflow first
        run_id = self.manager.db.create_workflow_run(
            user_id="user123",
            workflow_type="test",
            workflow_state={},
            planned_steps=[]
        )

        result = self.manager.cancel_workflow(run_id)
        self.assertEqual(result["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
