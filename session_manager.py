"""
Workshop Session Manager
Central coordinator for session lifecycle and state isolation.

Sessions ensure that:
- Tasks from one conversation don't bleed into another
- Each new Workshop startup gets a fresh context
- Previous sessions can be resumed if desired
- Auto-continue only works within the same session

Phase 4 Update (Jan 2026):
- Integration with ClaudeBridgeManager for session continuity
- Claude session ID tracking alongside Workshop sessions
- Automatic session binding on start/resume
"""

import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from logger import get_logger

log = get_logger("session_manager")

# Lazy import to avoid circular dependency
def _get_claude_bridge_manager():
    """Lazy import of Claude bridge manager."""
    try:
        from claude_bridge import get_claude_bridge_manager
        return get_claude_bridge_manager()
    except ImportError:
        log.debug("Claude bridge not available")
        return None


@dataclass
class Session:
    """
    Represents a Workshop session - a logical unit of work.

    A session starts when Workshop launches and ends when:
    - User explicitly ends it
    - Workshop shuts down gracefully
    - A new session is started (auto-archives previous)
    """
    session_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    mode: str = "text"  # "text", "voice", "dashboard"
    status: str = "active"  # "active", "archived", "abandoned"
    original_request: str = ""  # First user input that started the session
    task_count: int = 0
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "mode": self.mode,
            "status": self.status,
            "original_request": self.original_request,
            "task_count": self.task_count,
            "message_count": self.message_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        return cls(
            session_id=data["session_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            mode=data.get("mode", "text"),
            status=data.get("status", "active"),
            original_request=data.get("original_request", ""),
            task_count=data.get("task_count", 0),
            message_count=data.get("message_count", 0)
        )


class SessionManager:
    """
    Central coordinator for Workshop sessions.

    Ensures clean separation between different work sessions by:
    - Auto-archiving previous sessions on startup
    - Providing session IDs for task/memory binding
    - Managing session lifecycle (start, end, resume)
    """

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.home() / ".workshop" / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.archive_dir = self.base_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)

        self.current_file = self.base_dir / "current.json"

        self._current_session: Optional[Session] = None
        self._dashboard = None  # Set via set_dashboard()

        log.debug(f"SessionManager initialized at {self.base_dir}")

    def set_dashboard(self, dashboard):
        """Set dashboard reference for emitting session events"""
        self._dashboard = dashboard

    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:4]
        return f"sess_{timestamp}_{unique}"

    def start_session(self, mode: str = "text") -> Session:
        """
        Start a new session.

        If a previous session exists, it will be automatically archived.
        This ensures every Workshop startup gets a fresh context.

        Also binds the Claude bridge to this new session for context continuity.
        """
        # Archive any existing session first
        archived_session = self._archive_current_if_exists()

        # Create new session
        session_id = self._generate_session_id()
        session = Session(
            session_id=session_id,
            started_at=datetime.now(),
            mode=mode,
            status="active"
        )

        self._current_session = session
        self._save_current()

        # Phase 4: Bind Claude bridge to new session
        bridge_manager = _get_claude_bridge_manager()
        if bridge_manager:
            bridge_manager.bind_to_workshop_session(session_id)
            log.info(f"Bound Claude bridge to session: {session_id}")

        log.info(f"Started new session: {session_id} (mode={mode})")
        if archived_session:
            log.info(f"Archived previous session: {archived_session}")

        # Emit event if dashboard available
        if self._dashboard:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._dashboard.emit(
                        "session_started",
                        {
                            "session": session.to_dict(),
                            "archived_session": archived_session
                        }
                    ))
            except Exception as e:
                log.debug(f"Could not emit session_started: {e}")

        return session

    def _archive_current_if_exists(self) -> Optional[str]:
        """Archive the current session if one exists. Returns archived session_id."""
        if not self.current_file.exists():
            return None

        try:
            with open(self.current_file) as f:
                data = json.load(f)

            old_session_id = data.get("session_id")
            if not old_session_id:
                return None

            # Create archive directory for this session
            archive_path = self.archive_dir / old_session_id
            archive_path.mkdir(exist_ok=True)

            # Update session status to archived
            data["status"] = "archived"
            data["ended_at"] = datetime.now().isoformat()

            # Save metadata to archive
            with open(archive_path / "metadata.json", "w") as f:
                json.dump(data, f, indent=2)

            # Move any existing task file to archive
            old_tasks_file = Path.home() / ".workshop" / "tasks" / "current.json"
            if old_tasks_file.exists():
                try:
                    with open(old_tasks_file) as f:
                        tasks_data = json.load(f)

                    # Only archive if it's from this session
                    if tasks_data.get("session_id") == old_session_id:
                        shutil.copy(old_tasks_file, archive_path / "tasks.json")
                        # Clear the current tasks file
                        old_tasks_file.unlink()
                        log.debug(f"Archived tasks for session {old_session_id}")
                except Exception as e:
                    log.warning(f"Could not archive tasks: {e}")

            log.info(f"Archived session {old_session_id} to {archive_path}")
            return old_session_id

        except Exception as e:
            log.error(f"Failed to archive current session: {e}")
            return None

    def _save_current(self):
        """Save current session to disk"""
        if not self._current_session:
            return

        try:
            with open(self.current_file, "w") as f:
                json.dump(self._current_session.to_dict(), f, indent=2)
        except Exception as e:
            log.error(f"Failed to save current session: {e}")

    def get_current_session(self) -> Optional[Session]:
        """Get the current active session"""
        if self._current_session:
            return self._current_session

        # Try to load from disk
        if self.current_file.exists():
            try:
                with open(self.current_file) as f:
                    data = json.load(f)
                self._current_session = Session.from_dict(data)
                return self._current_session
            except Exception as e:
                log.error(f"Failed to load current session: {e}")

        return None

    def get_current_session_id(self) -> Optional[str]:
        """Get just the current session ID"""
        session = self.get_current_session()
        return session.session_id if session else None

    def is_current_session(self, session_id: str) -> bool:
        """Check if the given session_id matches the current session"""
        current = self.get_current_session_id()
        return current == session_id if current else False

    def end_session(self, archive: bool = True) -> Dict[str, Any]:
        """
        End the current session.

        Args:
            archive: If True, archive the session. If False, just clear it.

        Returns:
            Summary of the ended session
        """
        session = self.get_current_session()
        if not session:
            return {"error": "No active session"}

        session.ended_at = datetime.now()
        session.status = "archived" if archive else "abandoned"

        summary = {
            "session_id": session.session_id,
            "duration_seconds": (session.ended_at - session.started_at).total_seconds(),
            "task_count": session.task_count,
            "message_count": session.message_count,
            "status": session.status
        }

        if archive:
            self._save_current()  # Save final state
            self._archive_current_if_exists()

        # Clear current
        self._current_session = None
        if self.current_file.exists():
            self.current_file.unlink()

        log.info(f"Ended session {session.session_id}: {summary}")

        # Emit event
        if self._dashboard:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._dashboard.emit(
                        "session_ended",
                        {"summary": summary}
                    ))
            except Exception:
                pass

        return summary

    def update_session_stats(self, task_count: int = None, message_count: int = None):
        """Update session statistics"""
        session = self.get_current_session()
        if not session:
            return

        if task_count is not None:
            session.task_count = task_count
        if message_count is not None:
            session.message_count = message_count

        self._save_current()

    def set_original_request(self, request: str):
        """Set the original request that started this session (first user input)"""
        session = self.get_current_session()
        if session and not session.original_request:
            session.original_request = request[:200]  # Truncate for storage
            self._save_current()

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List archived sessions, most recent first"""
        sessions = []

        try:
            # Get all archived session directories
            for session_dir in sorted(self.archive_dir.iterdir(), reverse=True):
                if not session_dir.is_dir():
                    continue

                metadata_file = session_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file) as f:
                        data = json.load(f)

                    # Add summary info
                    tasks_file = session_dir / "tasks.json"
                    if tasks_file.exists():
                        with open(tasks_file) as f:
                            tasks_data = json.load(f)
                        data["task_count"] = len(tasks_data.get("tasks", []))
                        data["original_request"] = tasks_data.get("original_request", "")

                    sessions.append(data)

                    if len(sessions) >= limit:
                        break

                except Exception as e:
                    log.warning(f"Could not read session {session_dir.name}: {e}")

        except Exception as e:
            log.error(f"Failed to list sessions: {e}")

        return sessions

    def resume_session(self, session_id: str) -> Optional[Session]:
        """
        Resume a previously archived session.

        This will:
        1. Archive the current session (if any)
        2. Restore the specified session as current
        3. Restore its tasks
        """
        # Find the archived session
        archive_path = self.archive_dir / session_id
        if not archive_path.exists():
            log.error(f"Session not found: {session_id}")
            return None

        metadata_file = archive_path / "metadata.json"
        if not metadata_file.exists():
            log.error(f"Session metadata not found: {session_id}")
            return None

        # Archive current session first
        self._archive_current_if_exists()

        try:
            # Load the archived session
            with open(metadata_file) as f:
                data = json.load(f)

            # Update status and timestamps
            data["status"] = "active"
            # Keep the original started_at, but note it was resumed

            session = Session.from_dict(data)
            session.status = "active"

            # Save as current
            self._current_session = session
            self._save_current()

            # Restore tasks if they exist
            tasks_file = archive_path / "tasks.json"
            if tasks_file.exists():
                dest_tasks = Path.home() / ".workshop" / "tasks" / "current.json"
                dest_tasks.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(tasks_file, dest_tasks)
                log.info(f"Restored tasks from session {session_id}")

            # Phase 4: Bind Claude bridge to resumed session
            # This will automatically restore Claude session if available
            bridge_manager = _get_claude_bridge_manager()
            if bridge_manager:
                bridge_manager.bind_to_workshop_session(session_id)
                log.info(f"Bound Claude bridge to resumed session: {session_id}")

            log.info(f"Resumed session: {session_id}")

            # Emit event
            if self._dashboard:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._dashboard.emit(
                            "session_resumed",
                            {"session": session.to_dict()}
                        ))
                except Exception:
                    pass

            return session

        except Exception as e:
            log.error(f"Failed to resume session {session_id}: {e}")
            return None

    def detect_stale_state(self) -> Dict[str, Any]:
        """
        Detect any stale state from previous sessions.

        Returns info about orphan tasks/messages that don't belong
        to the current session.
        """
        result = {
            "has_stale_state": False,
            "stale_tasks": False,
            "stale_session_id": None,
            "details": []
        }

        current_session_id = self.get_current_session_id()

        # Check for tasks from different session
        tasks_file = Path.home() / ".workshop" / "tasks" / "current.json"
        if tasks_file.exists():
            try:
                with open(tasks_file) as f:
                    data = json.load(f)

                task_session = data.get("session_id")
                if task_session and task_session != current_session_id:
                    result["has_stale_state"] = True
                    result["stale_tasks"] = True
                    result["stale_session_id"] = task_session
                    result["details"].append(
                        f"Tasks from session {task_session} (request: {data.get('original_request', 'unknown')[:50]}...)"
                    )
            except Exception as e:
                log.warning(f"Could not check tasks for stale state: {e}")

        return result

    def get_claude_session_info(self) -> Dict[str, Any]:
        """
        Get Claude session information for the current Workshop session.

        Returns info about:
        - Claude session ID (for --resume)
        - Turn count
        - Estimated context usage
        """
        bridge_manager = _get_claude_bridge_manager()
        if bridge_manager:
            return bridge_manager.get_session_info()
        return {
            "claude_session_id": None,
            "turn_count": 0,
            "context_tokens_estimated": 0,
        }

    def clear_stale_state(self) -> bool:
        """
        Clear any stale state from previous sessions.

        Archives orphan tasks and clears memory messages that
        don't belong to the current session.
        """
        stale = self.detect_stale_state()
        if not stale["has_stale_state"]:
            log.info("No stale state to clear")
            return True

        try:
            if stale["stale_tasks"] and stale["stale_session_id"]:
                # Archive the stale tasks to their session
                tasks_file = Path.home() / ".workshop" / "tasks" / "current.json"
                if tasks_file.exists():
                    archive_path = self.archive_dir / stale["stale_session_id"]
                    archive_path.mkdir(exist_ok=True)

                    shutil.move(str(tasks_file), str(archive_path / "tasks.json"))
                    log.info(f"Moved stale tasks to archive: {stale['stale_session_id']}")

            log.info("Cleared stale state")

            # Emit event
            if self._dashboard:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._dashboard.emit(
                            "stale_state_cleared",
                            {"cleared": stale}
                        ))
                except Exception:
                    pass

            return True

        except Exception as e:
            log.error(f"Failed to clear stale state: {e}")
            return False


# Singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global SessionManager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
