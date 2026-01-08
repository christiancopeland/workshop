"""
Workshop Task Manager
Track progress on multi-step tasks with persistence and visibility.

Inspired by Claude Code's TodoRead/TodoWrite pattern - tasks should be
created FIRST before starting any non-trivial work to provide:
- Planning before execution
- Progress visibility for the user
- State tracking across conversation turns
- Resumability of interrupted work
"""

import json
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from logger import get_logger

log = get_logger("task_manager")


class TaskStatus(Enum):
    """Task status values"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Task:
    """
    A single task in the task list.

    Attributes:
        content: What needs to be done (imperative form: "Fix the bug")
        status: Current status (pending, in_progress, completed)
        active_form: Present continuous form for display ("Fixing the bug")
        created_at: When the task was created
        completed_at: When the task was completed (if applicable)
    """
    content: str
    status: TaskStatus
    active_form: str
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "content": self.content,
            "status": self.status.value,
            "active_form": self.active_form,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create Task from dictionary"""
        return cls(
            content=data["content"],
            status=TaskStatus(data["status"]),
            active_form=data["active_form"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        )


@dataclass
class TaskList:
    """
    A complete task list for a work session.

    Attributes:
        session_id: Unique identifier for this work session
        original_request: The user's original request that spawned these tasks
        tasks: List of Task objects
        created_at: When the task list was created
        updated_at: Last modification time
    """
    session_id: str
    tasks: List[Task] = field(default_factory=list)
    original_request: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "original_request": self.original_request,
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskList':
        """Create TaskList from dictionary"""
        return cls(
            session_id=data["session_id"],
            original_request=data.get("original_request", ""),
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now()
        )


class TaskManager:
    """
    Manages the current task list with persistence.

    Usage:
        manager = TaskManager()

        # Write tasks (replaces current list)
        manager.write_tasks([
            {"content": "Research the codebase", "status": "in_progress", "active_form": "Researching codebase"},
            {"content": "Implement the feature", "status": "pending", "active_form": "Implementing feature"},
        ])

        # Get tasks
        tasks = manager.get_tasks()

        # Format for agent context
        context = manager.format_for_context()

        # Clear when done
        manager.clear_tasks()
    """

    def __init__(self, tasks_dir: Optional[Path] = None, dashboard: Any = None):
        """
        Initialize TaskManager.

        Args:
            tasks_dir: Directory for task persistence. Defaults to ~/.workshop/tasks/
            dashboard: Optional DashboardServer for emitting events
        """
        self.tasks_dir = tasks_dir or Path.home() / ".workshop" / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

        self.current_file = self.tasks_dir / "current.json"
        self._task_list: Optional[TaskList] = None
        self._dashboard = dashboard
        self._bound_session_id: Optional[str] = None  # Session binding for isolation

        # Don't auto-load on init - wait for session binding
        # This prevents stale task leakage between sessions
        log.info(f"TaskManager initialized. Tasks dir: {self.tasks_dir}")

    def set_dashboard(self, dashboard: Any):
        """Set the dashboard server for event emission"""
        self._dashboard = dashboard

    def bind_to_session(self, session_id: str):
        """
        Bind this TaskManager to a specific session.

        This is critical for session isolation:
        - Tasks will only be loaded if they match the session
        - New tasks will be tagged with this session_id
        - Prevents stale task leakage between sessions
        """
        self._bound_session_id = session_id
        log.info(f"TaskManager bound to session: {session_id}")

        # Now try to load tasks - but only if they match our session
        self._load_for_session(session_id)

    def _load_for_session(self, expected_session: str):
        """Load tasks only if they belong to the expected session"""
        if not self.current_file.exists():
            log.debug("No existing tasks file")
            return

        try:
            with open(self.current_file, 'r') as f:
                data = json.load(f)

            loaded_session = data.get("session_id")

            # If tasks are from a different session, don't load them
            if loaded_session and loaded_session != expected_session:
                log.info(f"Skipping stale tasks from session {loaded_session} (current: {expected_session})")
                # Don't clear them - SessionManager handles archiving
                self._task_list = None
                return

            # Load the tasks
            self._task_list = TaskList.from_dict(data)
            log.info(f"Loaded {len(self._task_list.tasks)} tasks for session {expected_session}")

        except Exception as e:
            log.warning(f"Failed to load tasks: {e}")
            self._task_list = None

    def get_session_id(self) -> Optional[str]:
        """Get the session ID of the current task list"""
        if self._task_list:
            return self._task_list.session_id
        return self._bound_session_id

    def is_bound_to_session(self, session_id: str) -> bool:
        """Check if tasks belong to the given session"""
        current = self.get_session_id()
        return current == session_id if current else False

    def _emit_task_event_sync(self, event_type: str, **kwargs):
        """Emit a task event to the dashboard (sync wrapper)"""
        if not self._dashboard:
            return

        import asyncio

        async def _emit():
            try:
                if event_type == "task_list_updated":
                    await self._dashboard.emit_task_list_updated(
                        tasks=kwargs.get("tasks", []),
                        original_request=kwargs.get("original_request", "")
                    )
                elif event_type == "task_started":
                    await self._dashboard.emit_task_started(
                        task_content=kwargs.get("content", ""),
                        task_active_form=kwargs.get("active_form", ""),
                        task_index=kwargs.get("index", 0)
                    )
                elif event_type == "task_completed":
                    await self._dashboard.emit_task_completed(
                        task_content=kwargs.get("content", ""),
                        task_index=kwargs.get("index", 0)
                    )
                elif event_type == "task_cleared":
                    await self._dashboard.emit_task_cleared(
                        task_count=kwargs.get("count", 0)
                    )
            except Exception as e:
                log.debug(f"Failed to emit task event: {e}")

        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(_emit())
        except RuntimeError:
            # No running loop, skip dashboard update
            pass

    def _load(self):
        """Load task list from disk"""
        if self.current_file.exists():
            try:
                with open(self.current_file, 'r') as f:
                    data = json.load(f)
                self._task_list = TaskList.from_dict(data)
                log.debug(f"Loaded task list: {self._task_list.session_id}")
            except Exception as e:
                log.warning(f"Failed to load tasks: {e}")
                self._task_list = None

    def _save(self):
        """Save task list to disk"""
        if self._task_list:
            try:
                self._task_list.updated_at = datetime.now()
                with open(self.current_file, 'w') as f:
                    json.dump(self._task_list.to_dict(), f, indent=2)
                log.debug(f"Saved task list: {self._task_list.session_id}")
            except Exception as e:
                log.error(f"Failed to save tasks: {e}")

    def write_tasks(
        self,
        tasks: List[Dict[str, str]],
        original_request: str = "",
        work_evidence: List[str] = None
    ) -> str:
        """
        Create or update the task list with optional work evidence validation.

        This replaces the current task list entirely. The agent should call this:
        1. At the START of any non-trivial work (with initial tasks)
        2. After completing each task (with updated statuses)

        Args:
            tasks: List of task dicts with keys:
                - content: str (imperative: "Fix the bug")
                - status: str ("pending", "in_progress", "completed")
                - active_form: str (present continuous: "Fixing the bug")
            original_request: The user's original request (optional)
            work_evidence: List of tool names that were executed (for validation)

        Returns:
            Confirmation message

        Rules:
            - Only ONE task should be "in_progress" at a time
            - Mark tasks "completed" immediately after finishing
            - Always include active_form for display purposes
        """
        # Validate tasks
        if not tasks:
            return "Error: No tasks provided"

        # Check for multiple in_progress
        in_progress_count = sum(1 for t in tasks if t.get("status") == "in_progress")
        if in_progress_count > 1:
            log.warning(f"Multiple tasks marked in_progress ({in_progress_count}). Should be exactly 1.")

        # === WORK EVIDENCE VALIDATION ===
        # Prevent hallucinated task completion by checking if actual work was done
        previous_tasks = self.read_tasks_raw()

        if previous_tasks and work_evidence is not None:
            completed_count = sum(1 for t in tasks if t.get('status') == 'completed')
            prev_completed = sum(1 for t in previous_tasks if t.get('status') == 'completed')
            newly_completed = completed_count - prev_completed

            # Filter out task management tools - they don't count as "work"
            task_tools = ('task_write', 'task_read', 'task_clear')
            non_task_tools = [t for t in work_evidence if t not in task_tools]

            if newly_completed > 0 and len(non_task_tools) == 0:
                log.warning(f"Rejecting task completion: {newly_completed} tasks marked complete with no work evidence")
                # Revert completion status for tasks being marked complete without evidence
                for task in tasks:
                    if task.get('status') == 'completed':
                        # Find corresponding previous task
                        for prev in previous_tasks:
                            if prev.get('content') == task.get('content') and prev.get('status') != 'completed':
                                task['status'] = prev.get('status', 'pending')
                                log.info(f"Reverted task '{task.get('content')[:30]}...' to '{task['status']}'")
                                break

        # Create or update task list
        # Use bound session_id if available, otherwise fall back to existing or generate new
        session_id = self._bound_session_id or (self._task_list.session_id if self._task_list else datetime.now().strftime("%Y%m%d_%H%M%S"))

        task_objects = []
        for t in tasks:
            # Handle existing tasks (preserve timestamps)
            existing = self._find_existing_task(t.get("content", ""))

            status = TaskStatus(t.get("status", "pending"))

            task = Task(
                content=t.get("content", ""),
                status=status,
                active_form=t.get("active_form", t.get("content", "")),
                created_at=existing.created_at if existing else datetime.now(),
                completed_at=datetime.now() if status == TaskStatus.COMPLETED and (not existing or existing.status != TaskStatus.COMPLETED) else (existing.completed_at if existing else None)
            )
            task_objects.append(task)

        self._task_list = TaskList(
            session_id=session_id,
            tasks=task_objects,
            original_request=original_request or (self._task_list.original_request if self._task_list else ""),
            created_at=self._task_list.created_at if self._task_list else datetime.now()
        )

        self._save()

        # Build confirmation message
        completed = sum(1 for t in task_objects if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in task_objects if t.status == TaskStatus.IN_PROGRESS)
        pending = sum(1 for t in task_objects if t.status == TaskStatus.PENDING)

        # Emit dashboard event
        self._emit_task_event_sync(
            "task_list_updated",
            tasks=[t.to_dict() for t in task_objects],
            original_request=self._task_list.original_request
        )

        return f"Task list updated: {completed} completed, {in_progress} in progress, {pending} pending"

    def _find_existing_task(self, content: str) -> Optional[Task]:
        """Find an existing task by content (for preserving timestamps)"""
        if not self._task_list:
            return None
        for task in self._task_list.tasks:
            if task.content == content:
                return task
        return None

    def get_tasks(self) -> List[Task]:
        """
        Get the current task list.

        Returns:
            List of Task objects, or empty list if no tasks
        """
        if not self._task_list:
            return []
        return self._task_list.tasks

    def get_task_list(self) -> Optional[TaskList]:
        """Get the full TaskList object"""
        return self._task_list

    def clear_tasks(self) -> str:
        """
        Clear the current task list.

        Call this when:
        - All tasks are completed
        - Starting completely fresh work
        - User explicitly requests clearing

        Returns:
            Confirmation message
        """
        if self._task_list and self._task_list.tasks:
            count = len(self._task_list.tasks)

            # Archive to history before clearing (optional)
            self._archive_completed()

            self._task_list = None

            # Remove current file
            if self.current_file.exists():
                self.current_file.unlink()

            # Emit dashboard event
            self._emit_task_event_sync("task_cleared", count=count)

            log.info(f"Cleared {count} tasks")
            return f"Cleared {count} tasks. Ready for new work."

        return "No tasks to clear."

    def _archive_completed(self):
        """Archive completed task list to history (for learning)"""
        if not self._task_list:
            return

        history_dir = self.tasks_dir / "history"
        history_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = history_dir / f"tasks_{timestamp}.json"

        try:
            with open(history_file, 'w') as f:
                json.dump(self._task_list.to_dict(), f, indent=2)
            log.debug(f"Archived task list to {history_file}")
        except Exception as e:
            log.warning(f"Failed to archive tasks: {e}")

    def format_for_context(self) -> str:
        """
        Format current tasks for injection into agent context.

        Returns:
            Formatted string showing task progress, or empty string if no tasks
        """
        if not self._task_list or not self._task_list.tasks:
            return ""

        lines = ["## Current Tasks\n"]

        if self._task_list.original_request:
            lines.append(f"*Working on: {self._task_list.original_request}*\n")

        for i, task in enumerate(self._task_list.tasks, 1):
            if task.status == TaskStatus.COMPLETED:
                marker = "✓"
                suffix = ""
            elif task.status == TaskStatus.IN_PROGRESS:
                marker = "→"
                suffix = "  ← IN PROGRESS"
            else:
                marker = " "
                suffix = ""

            lines.append(f"{i}. [{marker}] {task.content}{suffix}")

        return "\n".join(lines)

    def format_for_display(self) -> str:
        """
        Format tasks for user-facing display.

        Returns:
            Formatted string with status indicators
        """
        if not self._task_list or not self._task_list.tasks:
            return "No active tasks."

        lines = ["📋 **Current Tasks**\n"]

        for i, task in enumerate(self._task_list.tasks, 1):
            if task.status == TaskStatus.COMPLETED:
                status_icon = "✅"
            elif task.status == TaskStatus.IN_PROGRESS:
                status_icon = "🔄"
            else:
                status_icon = "⬜"

            lines.append(f"{status_icon} {i}. {task.content}")

            if task.status == TaskStatus.IN_PROGRESS:
                lines.append(f"   *{task.active_form}...*")

        # Summary
        completed = sum(1 for t in self._task_list.tasks if t.status == TaskStatus.COMPLETED)
        total = len(self._task_list.tasks)
        lines.append(f"\n*Progress: {completed}/{total} completed*")

        return "\n".join(lines)

    def get_current_task(self) -> Optional[Task]:
        """Get the currently in-progress task, if any"""
        if not self._task_list:
            return None
        for task in self._task_list.tasks:
            if task.status == TaskStatus.IN_PROGRESS:
                return task
        return None

    def has_tasks(self) -> bool:
        """Check if there are any active tasks"""
        return bool(self._task_list and self._task_list.tasks)

    def read_tasks_raw(self) -> List[Dict[str, str]]:
        """
        Get current tasks as raw dicts for programmatic manipulation.

        Returns:
            List of task dicts with content, status, active_form
        """
        if not self._task_list or not self._task_list.tasks:
            return []

        return [
            {
                "content": task.content,
                "status": task.status.value,
                "active_form": task.active_form
            }
            for task in self._task_list.tasks
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get task statistics"""
        if not self._task_list or not self._task_list.tasks:
            return {
                "has_tasks": False,
                "total": 0,
                "completed": 0,
                "in_progress": 0,
                "pending": 0
            }

        tasks = self._task_list.tasks
        return {
            "has_tasks": True,
            "total": len(tasks),
            "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            "in_progress": sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS),
            "pending": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "session_id": self._task_list.session_id,
            "original_request": self._task_list.original_request
        }

    def get_context_for_router(self) -> Optional[str]:
        """
        Get task context for the router to help with intent classification.

        Returns a short summary that helps the router understand:
        - There are active tasks
        - What the original request was
        - What task is currently in progress

        This is critical for handling "please continue" type inputs.
        """
        if not self._task_list or not self._task_list.tasks:
            return None

        stats = self.get_stats()
        if stats['completed'] == stats['total']:
            return None  # All tasks done, no active workflow

        current = self.get_current_task()
        pending = [t for t in self._task_list.tasks if t.status == TaskStatus.PENDING]

        lines = [
            f"[ACTIVE WORKFLOW] Original request: \"{self._task_list.original_request}\"",
            f"Progress: {stats['completed']}/{stats['total']} tasks completed",
        ]

        if current:
            lines.append(f"Current task: {current.active_form}")

        if pending:
            lines.append(f"Next: {pending[0].content}")

        return " | ".join(lines)

    def get_context_for_llm(self) -> Optional[str]:
        """
        Get full task context for LLM conversation injection.

        Returns a detailed context block that tells the LLM:
        - The original request being worked on
        - All tasks with their status
        - What to work on next

        This should be injected into the conversation so the LLM
        can continue working on tasks without needing reminders.
        """
        if not self._task_list or not self._task_list.tasks:
            return None

        stats = self.get_stats()
        if stats['completed'] == stats['total']:
            return None  # All done

        lines = [
            "=== ACTIVE TASK CONTEXT ===",
            f"Working on: {self._task_list.original_request}",
            "",
            "Task Progress:"
        ]

        for i, task in enumerate(self._task_list.tasks, 1):
            if task.status == TaskStatus.COMPLETED:
                status = "✅"
            elif task.status == TaskStatus.IN_PROGRESS:
                status = "🔄"
            else:
                status = "⬜"
            lines.append(f"  {status} {i}. {task.content}")

        current = self.get_current_task()
        pending = [t for t in self._task_list.tasks if t.status == TaskStatus.PENDING]

        lines.append("")
        if current:
            lines.append(f"CURRENT: {current.active_form}")
        elif pending:
            lines.append(f"NEXT: {pending[0].content}")
            lines.append("(Mark this task as in_progress and begin working on it)")

        lines.append("=== END TASK CONTEXT ===")
        return "\n".join(lines)


# Singleton instance
_task_manager: Optional[TaskManager] = None


def get_task_manager(tasks_dir: Optional[Path] = None, dashboard: Any = None) -> TaskManager:
    """Get or create the singleton TaskManager instance"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(tasks_dir=tasks_dir, dashboard=dashboard)
    elif dashboard is not None:
        # Update dashboard reference if provided
        _task_manager.set_dashboard(dashboard)
    return _task_manager
