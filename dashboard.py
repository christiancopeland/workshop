"""
Workshop Dashboard Server
Real-time visualization of Workshop events, tool calls, and workflows.

Provides a WebSocket server that broadcasts events to connected dashboard clients.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import websockets
from websockets.server import WebSocketServerProtocol

from logger import get_logger

log = get_logger("dashboard")


class EventType(str, Enum):
    """Types of events the dashboard can display"""
    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_STARTED = "session_started"  # New session created
    SESSION_ENDED = "session_ended"  # Session archived
    SESSION_RESUMED = "session_resumed"  # Previous session restored
    STALE_STATE_DETECTED = "stale_state_detected"  # Found orphan state
    STALE_STATE_CLEARED = "stale_state_cleared"  # Orphan state cleaned up

    # User interaction
    USER_INPUT = "user_input"
    ASSISTANT_RESPONSE = "assistant_response"

    # Processing stages
    CONTEXT_LOADING = "context_loading"
    CONTEXT_LOADED = "context_loaded"
    INTENT_DETECTING = "intent_detecting"
    INTENT_DETECTED = "intent_detected"

    # Skill/Workflow events
    SKILL_MATCHED = "skill_matched"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_STEP = "workflow_step"
    WORKFLOW_COMPLETED = "workflow_completed"

    # Tool events
    TOOL_CALLING = "tool_calling"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # LLM events
    LLM_CALLING = "llm_calling"
    LLM_STREAMING = "llm_streaming"
    LLM_COMPLETE = "llm_complete"

    # Research events
    RESEARCH_STARTED = "research_started"
    RESEARCH_QUERY = "research_query"
    RESEARCH_FETCHING = "research_fetching"
    RESEARCH_COMPLETE = "research_complete"
    RESEARCH_SAVED = "research_saved"

    # Crawl4AI events
    CRAWL_FETCH_STARTED = "crawl_fetch_started"
    CRAWL_FETCH_COMPLETE = "crawl_fetch_complete"
    CRAWL_FETCH_ERROR = "crawl_fetch_error"
    CRAWL_DEEP_STARTED = "crawl_deep_started"
    CRAWL_DEEP_PAGE = "crawl_deep_page"
    CRAWL_DEEP_COMPLETE = "crawl_deep_complete"
    CRAWL_STEALTH_ENABLED = "crawl_stealth_enabled"
    CRAWL_DOMAIN_BLOCKED = "crawl_domain_blocked"
    CRAWL_PARALLEL_STARTED = "crawl_parallel_started"
    CRAWL_PARALLEL_COMPLETE = "crawl_parallel_complete"

    # Subagent events
    SUBAGENT_SPAWNING = "subagent_spawning"
    SUBAGENT_MODEL_SWAP = "subagent_model_swap"
    SUBAGENT_EXECUTING = "subagent_executing"
    SUBAGENT_COMPLETE = "subagent_complete"
    CONTEXT_SNAPSHOT = "context_snapshot"

    # Task management events
    TASK_LIST_UPDATED = "task_list_updated"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_CLEARED = "task_cleared"

    # System events
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class DashboardEvent:
    """A single event to display on the dashboard"""
    event_type: EventType | str
    timestamp: float = field(default_factory=time.time)
    data: Dict = field(default_factory=dict)
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict:
        # Handle both EventType enum and plain strings
        event_value = self.event_type.value if isinstance(self.event_type, EventType) else self.event_type
        return {
            "type": event_value,
            "timestamp": self.timestamp,
            "time_str": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3],
            "data": self.data,
            "trace_id": self.trace_id
        }


class DashboardServer:
    """
    WebSocket server for real-time dashboard updates.

    Broadcasts events to all connected dashboard clients.
    Also maintains event history for new clients joining.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8766,  # Different from construct server (8765)
        max_history: int = 500
    ):
        self.host = host
        self.port = port
        self.max_history = max_history

        self.clients: Set[WebSocketServerProtocol] = set()
        self._server = None
        self._event_history: List[DashboardEvent] = []

        # Current state tracking
        self._current_trace_id: Optional[str] = None
        self._active_tools: Dict[str, Dict] = {}  # call_id -> tool info
        self._session_start: Optional[float] = None

        # Handler callbacks (set by Workshop)
        self._message_handler = None  # Async function to handle user messages
        self._stop_handler = None  # Async function to trigger emergency stop
        self._agent_running = False  # Track if agent is currently processing

        # Session manager reference (set via set_session_manager())
        self._session_manager = None

        # Stats
        self._stats = {
            "total_events": 0,
            "tool_calls": 0,
            "llm_calls": 0,
            "errors": 0,
            "user_messages": 0
        }

    async def start(self) -> bool:
        """Start the dashboard server"""
        try:
            self._server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port
            )
            self._session_start = time.time()
            log.info(f"Dashboard server started on ws://{self.host}:{self.port}")
            print(f"📊 Dashboard server: ws://{self.host}:{self.port}")
            print(f"   Open dashboard/index.html in browser to view")
            return True
        except Exception as e:
            log.error(f"Failed to start dashboard server: {e}")
            return False

    async def stop(self):
        """Stop the dashboard server"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            log.info("Dashboard server stopped")

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new dashboard client connection"""
        self.clients.add(websocket)
        client_id = id(websocket)
        log.info(f"Dashboard client connected: {client_id}")

        try:
            # Send initial state
            await self._send_initial_state(websocket)

            # Handle incoming messages (commands from dashboard)
            async for message in websocket:
                await self._handle_dashboard_command(message, websocket)

        except websockets.exceptions.ConnectionClosed:
            log.info(f"Dashboard client disconnected: {client_id}")
        finally:
            self.clients.discard(websocket)

    async def _send_initial_state(self, websocket: WebSocketServerProtocol):
        """Send current state and recent history to new client"""
        state = {
            "action": "initial_state",
            "stats": self._stats,
            "session_start": self._session_start,
            "active_tools": list(self._active_tools.values()),
            "current_trace_id": self._current_trace_id,
            "history": [e.to_dict() for e in self._event_history[-100:]]  # Last 100 events
        }
        await websocket.send(json.dumps(state))

    async def _handle_dashboard_command(self, message: str, websocket: WebSocketServerProtocol):
        """Handle commands from the dashboard UI"""
        try:
            data = json.loads(message)
            action = data.get("action")

            if action == "get_history":
                # Send more history
                limit = data.get("limit", 100)
                offset = data.get("offset", 0)
                events = self._event_history[-(limit + offset):-offset] if offset else self._event_history[-limit:]
                await websocket.send(json.dumps({
                    "action": "history",
                    "events": [e.to_dict() for e in events]
                }))

            elif action == "get_stats":
                await websocket.send(json.dumps({
                    "action": "stats",
                    "stats": self._stats
                }))

            elif action == "clear_history":
                self._event_history.clear()
                await self._broadcast({"action": "history_cleared"})

            elif action == "get_research_library":
                # Get all research platforms from ~/.workshop/research/
                library = await self._get_research_library()
                await websocket.send(json.dumps({
                    "action": "research_library",
                    "platforms": library
                }))

            elif action == "get_research_detail":
                # Get full details of a specific research platform
                filename = data.get("filename")
                if filename:
                    detail = await self._get_research_detail(filename)
                    await websocket.send(json.dumps({
                        "action": "research_detail",
                        "platform": detail
                    }))

            elif action == "get_subagent_history":
                # Get history of subagent executions from snapshots
                history = await self._get_subagent_history()
                await websocket.send(json.dumps({
                    "action": "subagent_history",
                    "snapshots": history
                }))

            elif action == "get_snapshot_detail":
                # Get full context snapshot
                snapshot_id = data.get("snapshot_id")
                if snapshot_id:
                    detail = await self._get_snapshot_detail(snapshot_id)
                    await websocket.send(json.dumps({
                        "action": "snapshot_detail",
                        "snapshot": detail
                    }))

            elif action == "get_current_tasks":
                # Get current task list from task manager
                tasks_data = await self._get_current_tasks()
                await websocket.send(json.dumps({
                    "action": "task_list",
                    **tasks_data
                }))

            elif action == "send_message":
                # Send a message to the agent
                message = data.get("message", "").strip()
                if message and self._message_handler:
                    # Call the registered message handler
                    asyncio.create_task(self._handle_user_message(message))
                    await websocket.send(json.dumps({
                        "action": "message_received",
                        "message": message
                    }))

            elif action == "emergency_stop":
                # Emergency stop - cancel current agent execution
                if self._stop_handler:
                    await self._stop_handler()
                await self._broadcast({
                    "action": "emergency_stop_triggered",
                    "timestamp": time.time()
                })

            elif action == "get_system_status":
                # Get system resource status (VRAM, loaded models, etc.)
                status = await self._get_system_status()
                await websocket.send(json.dumps({
                    "action": "system_status",
                    **status
                }))

            # === Memory Inspector Actions ===
            elif action == "get_memory_stats":
                stats = await self._get_memory_stats()
                await websocket.send(json.dumps({
                    "action": "memory_stats",
                    "stats": stats
                }))

            elif action == "get_memory_facts":
                facts = await self._get_memory_facts()
                await websocket.send(json.dumps({
                    "action": "memory_facts",
                    "facts": facts
                }))

            elif action == "get_memory_projects":
                projects_data = await self._get_memory_projects()
                await websocket.send(json.dumps({
                    "action": "memory_projects",
                    **projects_data
                }))

            elif action == "get_memory_profile":
                profile = await self._get_memory_profile()
                await websocket.send(json.dumps({
                    "action": "memory_profile",
                    "profile": profile
                }))

            elif action == "get_memory_messages":
                limit = data.get("limit", 20)
                messages = await self._get_memory_messages(limit)
                await websocket.send(json.dumps({
                    "action": "memory_messages",
                    "messages": messages
                }))

            elif action == "search_memory":
                query = data.get("query", "")
                limit = data.get("limit", 10)
                results = await self._search_memory(query, limit)
                await websocket.send(json.dumps({
                    "action": "memory_search_results",
                    "results": results
                }))

            elif action == "delete_memory_fact":
                key = data.get("key", "")
                if key:
                    success = await self._delete_memory_fact(key)
                    await websocket.send(json.dumps({
                        "action": "memory_fact_deleted",
                        "key": key,
                        "success": success
                    }))

            # === Memory Write Actions (New for Knowledge Base) ===
            elif action == "update_user_profile":
                profile = data.get("profile", "")
                success = await self._update_user_profile(profile)
                await websocket.send(json.dumps({
                    "action": "profile_updated",
                    "success": success
                }))

            elif action == "add_project_note":
                content = data.get("content", "")
                category = data.get("category", "general")
                result = await self._add_project_note(content, category)
                await websocket.send(json.dumps({
                    "action": "note_added",
                    **result
                }))

            elif action == "get_project_notes":
                project_name = data.get("project_name")
                notes = await self._get_project_notes(project_name)
                await websocket.send(json.dumps({
                    "action": "project_notes",
                    "notes": notes
                }))

            elif action == "get_active_project":
                project = await self._get_active_project()
                await websocket.send(json.dumps({
                    "action": "active_project",
                    "project": project
                }))

            elif action == "set_active_project":
                name = data.get("name")
                path = data.get("path")
                description = data.get("description")
                result = await self._set_active_project(name, path, description)
                await websocket.send(json.dumps({
                    "action": "project_set",
                    **result
                }))

            elif action == "update_fact":
                key = data.get("key")
                value = data.get("value")
                result = await self._update_fact(key, value)
                await websocket.send(json.dumps({
                    "action": "fact_updated",
                    **result
                }))

            elif action == "update_project_note":
                note_id = data.get("note_id")
                content = data.get("content")
                result = await self._update_project_note(note_id, content)
                await websocket.send(json.dumps({
                    "action": "note_updated",
                    **result
                }))

            elif action == "delete_project_note":
                note_id = data.get("note_id")
                result = await self._delete_project_note(note_id)
                await websocket.send(json.dumps({
                    "action": "note_deleted",
                    **result
                }))

            # === Session Management Actions ===
            elif action == "get_session_info":
                session_info = await self._get_session_info()
                await websocket.send(json.dumps({
                    "action": "session_info",
                    **session_info
                }))

            elif action == "start_new_session":
                result = await self._start_new_session(data.get("mode", "dashboard"))
                await websocket.send(json.dumps({
                    "action": "session_started",
                    **result
                }))

            elif action == "end_session":
                result = await self._end_session(data.get("archive", True))
                await websocket.send(json.dumps({
                    "action": "session_ended",
                    **result
                }))

            elif action == "list_sessions":
                sessions = await self._list_sessions(data.get("limit", 20))
                await websocket.send(json.dumps({
                    "action": "session_list",
                    "sessions": sessions
                }))

            elif action == "resume_session":
                session_id = data.get("session_id")
                if session_id:
                    result = await self._resume_session(session_id)
                    await websocket.send(json.dumps({
                        "action": "session_resumed",
                        **result
                    }))

            elif action == "detect_stale_state":
                stale = await self._detect_stale_state()
                await websocket.send(json.dumps({
                    "action": "stale_state_info",
                    **stale
                }))

            elif action == "clear_stale_state":
                result = await self._clear_stale_state()
                await websocket.send(json.dumps({
                    "action": "stale_state_cleared",
                    "success": result
                }))

        except json.JSONDecodeError:
            log.warning(f"Invalid JSON from dashboard: {message[:100]}")

    async def _broadcast(self, message: Dict):
        """Broadcast a message to all connected clients"""
        if not self.clients:
            return

        payload = json.dumps(message)
        await asyncio.gather(
            *[client.send(payload) for client in self.clients],
            return_exceptions=True
        )

    async def _get_research_library(self) -> List[Dict]:
        """Get list of all research platforms from ~/.workshop/research/"""
        research_dir = Path.home() / ".workshop" / "research"
        platforms = []

        if not research_dir.exists():
            return platforms

        # Check for active research
        active_path = None
        active_file = research_dir / "_active.json"
        if active_file.exists():
            try:
                with open(active_file) as f:
                    active_data = json.load(f)
                    active_path = active_data.get("active_platform")
            except Exception:
                pass

        for f in sorted(research_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.name.startswith("_"):  # Skip _active.json
                continue
            try:
                with open(f) as fp:
                    data = json.load(fp)
                platforms.append({
                    "filename": f.name,
                    "topic": data.get("topic", f.stem),
                    "original_query": data.get("original_query", ""),
                    "source_count": len(data.get("sources", [])),
                    "created_at": data.get("created_at", ""),
                    "size_kb": f.stat().st_size // 1024,
                    "is_active": str(f) == active_path
                })
            except Exception as e:
                log.warning(f"Failed to read research file {f}: {e}")

        return platforms

    async def _get_research_detail(self, filename: str) -> Optional[Dict]:
        """Get full details of a specific research platform"""
        research_dir = Path.home() / ".workshop" / "research"
        filepath = research_dir / filename

        if not filepath.exists() or not filepath.is_file():
            return None

        try:
            with open(filepath) as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load research detail: {e}")
            return None

    async def _get_subagent_history(self) -> List[Dict]:
        """Get history of subagent executions from snapshots"""
        snapshots_dir = Path.home() / ".workshop" / "agents" / "snapshots"
        history = []

        if not snapshots_dir.exists():
            return history

        for f in sorted(snapshots_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:  # Limit to 50 most recent
            try:
                with open(f) as fp:
                    data = json.load(fp)

                # Get output preview if available
                output = data.get("output", "")
                output_preview = output[:200] if output else None

                # Extract trace summary if available
                trace = data.get("trace", {})
                trace_summary = None
                if trace:
                    trace_summary = {
                        "llm_call_count": trace.get("llm_call_count", len(trace.get("llm_calls", []))),
                        "tool_call_count": trace.get("tool_call_count", len(trace.get("tool_calls", []))),
                        "total_tokens": trace.get("total_tokens", 0),
                        "total_duration_ms": trace.get("total_duration_ms", 0)
                    }

                history.append({
                    "snapshot_id": data.get("snapshot_id", f.stem),
                    "timestamp": data.get("timestamp", ""),
                    "subagent_name": data.get("subagent_name", ""),
                    "subagent_model": data.get("subagent_model", ""),
                    "task": data.get("task_description", ""),
                    "research_topic": data.get("research_topic", ""),
                    "primary_model": data.get("primary_model", ""),
                    "status": data.get("status", "completed"),
                    "duration_ms": data.get("duration_ms"),
                    "output_length": len(output) if output else 0,
                    "output_preview": output_preview,
                    "has_trace": bool(trace),
                    "trace_summary": trace_summary
                })
            except Exception as e:
                log.warning(f"Failed to read snapshot {f}: {e}")

        return history

    async def _get_snapshot_detail(self, snapshot_id: str) -> Optional[Dict]:
        """Get full context snapshot details"""
        snapshots_dir = Path.home() / ".workshop" / "agents" / "snapshots"
        filepath = snapshots_dir / f"{snapshot_id}.json"

        if not filepath.exists():
            return None

        try:
            with open(filepath) as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load snapshot detail: {e}")
            return None

    async def _get_current_tasks(self) -> Dict:
        """Get current tasks from task manager"""
        tasks_file = Path.home() / ".workshop" / "tasks" / "current.json"

        if not tasks_file.exists():
            return {
                "has_tasks": False,
                "tasks": [],
                "original_request": "",
                "stats": {
                    "total": 0,
                    "completed": 0,
                    "in_progress": 0,
                    "pending": 0
                }
            }

        try:
            with open(tasks_file) as f:
                data = json.load(f)

            tasks = data.get("tasks", [])
            completed = sum(1 for t in tasks if t.get("status") == "completed")
            in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")
            pending = sum(1 for t in tasks if t.get("status") == "pending")

            return {
                "has_tasks": len(tasks) > 0,
                "tasks": tasks,
                "original_request": data.get("original_request", ""),
                "session_id": data.get("session_id", ""),
                "stats": {
                    "total": len(tasks),
                    "completed": completed,
                    "in_progress": in_progress,
                    "pending": pending
                }
            }
        except Exception as e:
            log.error(f"Failed to load tasks: {e}")
            return {
                "has_tasks": False,
                "tasks": [],
                "original_request": "",
                "stats": {
                    "total": 0,
                    "completed": 0,
                    "in_progress": 0,
                    "pending": 0
                }
            }

    def set_message_handler(self, handler):
        """Set the handler for processing user messages from dashboard"""
        self._message_handler = handler

    def set_stop_handler(self, handler):
        """Set the handler for emergency stop"""
        self._stop_handler = handler

    def set_session_manager(self, session_manager):
        """Set the session manager for session control from dashboard"""
        self._session_manager = session_manager

    def set_agent_running(self, running: bool):
        """Update agent running state"""
        self._agent_running = running

    async def _handle_user_message(self, message: str):
        """Handle a user message from the dashboard"""
        if self._message_handler:
            try:
                self.set_agent_running(True)
                await self._broadcast({
                    "action": "agent_processing",
                    "message": message,
                    "timestamp": time.time()
                })
                await self._message_handler(message)
            except Exception as e:
                log.error(f"Error processing dashboard message: {e}")
                await self.emit_error(str(e), "dashboard_message")
            finally:
                self.set_agent_running(False)
                await self._broadcast({
                    "action": "agent_idle",
                    "timestamp": time.time()
                })

    async def _get_system_status(self) -> Dict:
        """Get system status including VRAM, loaded models, etc."""
        import aiohttp
        import subprocess

        status = {
            "agent_running": self._agent_running,
            "connected_clients": len(self.clients),
            "loaded_models": [],
            "vram": None
        }

        # Helper to format bytes
        def format_bytes(num_bytes: int) -> str:
            if num_bytes == 0:
                return "0 B"
            k = 1024
            sizes = ['B', 'KB', 'MB', 'GB', 'TB']
            i = 0
            while num_bytes >= k and i < len(sizes) - 1:
                num_bytes /= k
                i += 1
            return f"{num_bytes:.1f} {sizes[i]}"

        # Get total GPU VRAM using nvidia-smi
        total_vram_bytes = 0
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # nvidia-smi returns memory in MiB
                total_mib = int(result.stdout.strip().split('\n')[0])
                total_vram_bytes = total_mib * 1024 * 1024
        except Exception as e:
            log.debug(f"Could not get total VRAM from nvidia-smi: {e}")

        # Try to get Ollama model status
        used_vram_bytes = 0
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:11434/api/ps",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("models", [])
                        status["loaded_models"] = [
                            {
                                "name": m.get("name", "unknown"),
                                "size": m.get("size_vram", m.get("size", 0)),
                                "sizeFormatted": format_bytes(m.get("size_vram", m.get("size", 0)))
                            }
                            for m in models
                        ]
                        # Calculate total VRAM usage from loaded models
                        used_vram_bytes = sum(m.get("size_vram", 0) for m in models)
        except Exception as e:
            log.debug(f"Could not get Ollama status: {e}")

        # Build VRAM status with proper format for frontend
        if total_vram_bytes > 0 or used_vram_bytes > 0:
            # If we couldn't get total VRAM, estimate it as used + some headroom
            if total_vram_bytes == 0:
                total_vram_bytes = max(used_vram_bytes, 8 * 1024 * 1024 * 1024)  # At least 8GB

            percent = (used_vram_bytes / total_vram_bytes * 100) if total_vram_bytes > 0 else 0
            status["vram"] = {
                "used": used_vram_bytes,
                "total": total_vram_bytes,
                "percent": round(percent, 1)
            }

        return status

    # === Memory Inspector Methods ===

    def _get_memory_system(self):
        """Get the memory system instance from the global context"""
        try:
            from memory import MemorySystem
            from config import Config
            config = Config()
            # Try to get existing memory system or create one
            memory = MemorySystem(config.CHROMA_PATH, config.SQLITE_PATH)
            return memory
        except Exception as e:
            log.error(f"Failed to get memory system: {e}")
            return None

    async def _get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        memory = self._get_memory_system()
        if not memory:
            return {
                "semantic_count": 0,
                "facts_count": 0,
                "projects_count": 0,
                "profile_exists": False,
                "message_count": 0
            }
        try:
            raw_stats = memory.get_memory_stats()
            # Map to frontend expected field names
            return {
                "semantic_count": raw_stats.get("long_term_memories", 0),
                "facts_count": raw_stats.get("total_facts", 0),
                "projects_count": len(memory.list_projects()),
                "profile_exists": bool(memory.get_user_profile()),
                "message_count": raw_stats.get("total_messages", 0)
            }
        except Exception as e:
            log.error(f"Failed to get memory stats: {e}")
            return {
                "semantic_count": 0,
                "facts_count": 0,
                "projects_count": 0,
                "profile_exists": False,
                "message_count": 0
            }

    async def _get_memory_facts(self) -> List[Dict]:
        """Get all stored facts"""
        memory = self._get_memory_system()
        if not memory:
            return []
        try:
            cursor = memory.conn.execute("""
                SELECT key, value, category, created_at, updated_at
                FROM facts
                ORDER BY updated_at DESC
                LIMIT 100
            """)
            facts = []
            for row in cursor:
                try:
                    value = json.loads(row["value"])
                except:
                    value = row["value"]
                facts.append({
                    "key": row["key"],
                    "value": value,
                    "category": row["category"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                })
            return facts
        except Exception as e:
            log.error(f"Failed to get facts: {e}")
            return []

    async def _get_memory_projects(self) -> Dict:
        """Get all projects and active project"""
        memory = self._get_memory_system()
        if not memory:
            return {"projects": [], "active_project": None}
        try:
            projects = memory.list_projects()
            active_project = memory.get_active_project()
            return {
                "projects": projects,
                "active_project": active_project
            }
        except Exception as e:
            log.error(f"Failed to get projects: {e}")
            return {"projects": [], "active_project": None}

    async def _get_memory_profile(self) -> str:
        """Get user profile"""
        memory = self._get_memory_system()
        if not memory:
            return ""
        try:
            return memory.get_user_profile() or ""
        except Exception as e:
            log.error(f"Failed to get profile: {e}")
            return ""

    async def _get_memory_messages(self, limit: int = 20) -> List[Dict]:
        """Get recent conversation messages from database"""
        memory = self._get_memory_system()
        if not memory:
            return []
        try:
            # Query messages from database (not just in-memory)
            cursor = memory.conn.execute("""
                SELECT role, content, timestamp
                FROM messages
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            messages = []
            for row in cursor:
                messages.append({
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"]
                })
            # Reverse to get chronological order
            return list(reversed(messages))
        except Exception as e:
            log.error(f"Failed to get messages: {e}")
            return []

    async def _search_memory(self, query: str, limit: int = 10) -> List[Dict]:
        """Search semantic memory"""
        memory = self._get_memory_system()
        if not memory or not query.strip():
            return []
        try:
            # Use include_metadata=True to get distances and metadata
            results = memory.search_memories(query, k=limit, include_metadata=True)
            return results
        except Exception as e:
            log.error(f"Failed to search memory: {e}")
            return []

    async def _delete_memory_fact(self, key: str) -> bool:
        """Delete a fact by key"""
        memory = self._get_memory_system()
        if not memory:
            return False
        try:
            memory.conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            memory.conn.commit()
            log.info(f"Deleted fact: {key}")
            return True
        except Exception as e:
            log.error(f"Failed to delete fact: {e}")
            return False

    # === Memory Write Methods (New for Knowledge Base) ===

    async def _update_user_profile(self, profile: str) -> bool:
        """Update the user profile"""
        memory = self._get_memory_system()
        if not memory:
            return False
        try:
            memory.set_user_profile(profile)
            log.info(f"User profile updated: {len(profile)} chars")
            return True
        except Exception as e:
            log.error(f"Failed to update profile: {e}")
            return False

    async def _add_project_note(self, content: str, category: str = "general") -> Dict:
        """Add a note to the active project"""
        memory = self._get_memory_system()
        if not memory:
            return {"success": False, "error": "Memory system not available"}
        try:
            result = memory.add_project_note(content, category)
            log.info(f"Project note added: {content[:50]}...")
            return {"success": True, "message": result}
        except Exception as e:
            log.error(f"Failed to add project note: {e}")
            return {"success": False, "error": str(e)}

    async def _get_project_notes(self, project_name: str = None) -> List[Dict]:
        """Get notes for a project"""
        memory = self._get_memory_system()
        if not memory:
            return []
        try:
            notes = memory.get_project_notes(project_name)
            return notes
        except Exception as e:
            log.error(f"Failed to get project notes: {e}")
            return []

    async def _get_active_project(self) -> Optional[Dict]:
        """Get the currently active project"""
        memory = self._get_memory_system()
        if not memory:
            return None
        try:
            return memory.get_active_project()
        except Exception as e:
            log.error(f"Failed to get active project: {e}")
            return None

    async def _set_active_project(self, name: str, path: str = None, description: str = None) -> Dict:
        """Set the active project"""
        memory = self._get_memory_system()
        if not memory:
            return {"success": False, "error": "Memory system not available"}
        try:
            result = memory.set_active_project(name, path, description)
            log.info(f"Active project set: {name}")
            return {"success": True, "message": result, "project_name": name}
        except Exception as e:
            log.error(f"Failed to set active project: {e}")
            return {"success": False, "error": str(e)}

    async def _update_fact(self, key: str, value: str) -> Dict:
        """Update a fact's value"""
        memory = self._get_memory_system()
        if not memory:
            return {"success": False, "error": "Memory system not available"}
        try:
            memory.set_fact(key, value)
            log.info(f"Fact updated: {key}")
            return {"success": True, "key": key}
        except Exception as e:
            log.error(f"Failed to update fact: {e}")
            return {"success": False, "error": str(e)}

    async def _update_project_note(self, note_id: int, content: str) -> Dict:
        """Update a project note"""
        memory = self._get_memory_system()
        if not memory:
            return {"success": False, "error": "Memory system not available"}
        try:
            memory.conn.execute("""
                UPDATE project_notes
                SET content = ?, created_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (content, note_id))
            memory.conn.commit()
            log.info(f"Project note updated: {note_id}")
            return {"success": True, "note_id": note_id}
        except Exception as e:
            log.error(f"Failed to update project note: {e}")
            return {"success": False, "error": str(e)}

    async def _delete_project_note(self, note_id: int) -> Dict:
        """Delete a project note"""
        memory = self._get_memory_system()
        if not memory:
            return {"success": False, "error": "Memory system not available"}
        try:
            memory.conn.execute("DELETE FROM project_notes WHERE id = ?", (note_id,))
            memory.conn.commit()
            log.info(f"Project note deleted: {note_id}")
            return {"success": True, "note_id": note_id}
        except Exception as e:
            log.error(f"Failed to delete project note: {e}")
            return {"success": False, "error": str(e)}

    # === Session Management Methods ===

    async def _get_session_info(self) -> Dict:
        """Get current session information"""
        if not self._session_manager:
            return {"error": "Session manager not available", "has_session": False}

        session = self._session_manager.get_current_session()
        if not session:
            return {"has_session": False}

        return {
            "has_session": True,
            "session": session.to_dict(),
            "stale_state": self._session_manager.detect_stale_state()
        }

    async def _start_new_session(self, mode: str = "dashboard") -> Dict:
        """Start a new session"""
        if not self._session_manager:
            return {"error": "Session manager not available", "success": False}

        try:
            session = self._session_manager.start_session(mode=mode)
            log.info(f"Started new session from dashboard: {session.session_id}")

            # Broadcast to all clients
            await self._broadcast({
                "action": "session_started",
                "session": session.to_dict()
            })

            return {
                "success": True,
                "session": session.to_dict()
            }
        except Exception as e:
            log.error(f"Failed to start session: {e}")
            return {"error": str(e), "success": False}

    async def _end_session(self, archive: bool = True) -> Dict:
        """End the current session"""
        if not self._session_manager:
            return {"error": "Session manager not available", "success": False}

        try:
            summary = self._session_manager.end_session(archive=archive)

            # Broadcast to all clients
            await self._broadcast({
                "action": "session_ended",
                "summary": summary
            })

            return {
                "success": True,
                "summary": summary
            }
        except Exception as e:
            log.error(f"Failed to end session: {e}")
            return {"error": str(e), "success": False}

    async def _list_sessions(self, limit: int = 20) -> List[Dict]:
        """List archived sessions"""
        if not self._session_manager:
            return []

        try:
            return self._session_manager.list_sessions(limit=limit)
        except Exception as e:
            log.error(f"Failed to list sessions: {e}")
            return []

    async def _resume_session(self, session_id: str) -> Dict:
        """Resume a previous session"""
        if not self._session_manager:
            return {"error": "Session manager not available", "success": False}

        try:
            session = self._session_manager.resume_session(session_id)
            if session:
                # Broadcast to all clients
                await self._broadcast({
                    "action": "session_resumed",
                    "session": session.to_dict()
                })

                return {
                    "success": True,
                    "session": session.to_dict()
                }
            else:
                return {"error": f"Session not found: {session_id}", "success": False}
        except Exception as e:
            log.error(f"Failed to resume session: {e}")
            return {"error": str(e), "success": False}

    async def _detect_stale_state(self) -> Dict:
        """Detect stale state from previous sessions"""
        if not self._session_manager:
            return {"has_stale_state": False}

        try:
            return self._session_manager.detect_stale_state()
        except Exception as e:
            log.error(f"Failed to detect stale state: {e}")
            return {"has_stale_state": False, "error": str(e)}

    async def _clear_stale_state(self) -> bool:
        """Clear stale state from previous sessions"""
        if not self._session_manager:
            return False

        try:
            result = self._session_manager.clear_stale_state()

            # Broadcast to all clients
            await self._broadcast({
                "action": "stale_state_cleared",
                "success": result
            })

            return result
        except Exception as e:
            log.error(f"Failed to clear stale state: {e}")
            return False

    def _add_event(self, event: DashboardEvent):
        """Add event to history and update stats"""
        self._event_history.append(event)
        if len(self._event_history) > self.max_history:
            self._event_history = self._event_history[-self.max_history:]

        self._stats["total_events"] += 1

        # Update type-specific stats
        if event.event_type == EventType.USER_INPUT:
            self._stats["user_messages"] += 1
        elif event.event_type == EventType.TOOL_CALLING:
            self._stats["tool_calls"] += 1
        elif event.event_type == EventType.LLM_CALLING:
            self._stats["llm_calls"] += 1
        elif event.event_type == EventType.ERROR:
            self._stats["errors"] += 1

    # === Public API for emitting events ===

    async def emit(self, event_type: EventType | str, data: Dict = None, trace_id: str = None):
        """Emit an event to all connected dashboards"""
        # Convert string to EventType if needed, or use string directly
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                # String doesn't match any enum value - use as-is
                pass

        event = DashboardEvent(
            event_type=event_type,
            data=data or {},
            trace_id=trace_id or self._current_trace_id
        )

        self._add_event(event)

        await self._broadcast({
            "action": "event",
            **event.to_dict()
        })

        event_value = event_type.value if isinstance(event_type, EventType) else event_type
        log.debug(f"Dashboard event: {event_value}")

    async def emit_user_input(self, text: str, trace_id: str = None):
        """Emit a user input event"""
        self._current_trace_id = trace_id
        await self.emit(EventType.USER_INPUT, {
            "text": text,
            "length": len(text)
        }, trace_id)

    async def emit_context_loading(self, sources: List[str], trace_id: str = None):
        """Emit context loading start"""
        await self.emit(EventType.CONTEXT_LOADING, {
            "sources": sources
        }, trace_id)

    async def emit_context_loaded(self, sources: List[Dict], total_length: int, trace_id: str = None):
        """Emit context loaded"""
        await self.emit(EventType.CONTEXT_LOADED, {
            "sources": sources,
            "total_length": total_length,
            "source_count": len(sources)
        }, trace_id)

    async def emit_intent_detected(self, tool_name: str, skill_name: str, pattern: str = None, trace_id: str = None):
        """Emit intent detection result"""
        await self.emit(EventType.INTENT_DETECTED, {
            "tool_name": tool_name,
            "skill_name": skill_name,
            "pattern": pattern
        }, trace_id)

    async def emit_skill_matched(self, skill_name: str, reason: str, trace_id: str = None):
        """Emit skill match event"""
        await self.emit(EventType.SKILL_MATCHED, {
            "skill_name": skill_name,
            "reason": reason
        }, trace_id)

    async def emit_workflow_started(self, workflow_name: str, skill_name: str, trigger: str = None, trace_id: str = None):
        """Emit workflow start"""
        await self.emit(EventType.WORKFLOW_STARTED, {
            "workflow_name": workflow_name,
            "skill_name": skill_name,
            "trigger": trigger
        }, trace_id)

    async def emit_workflow_step(self, workflow_name: str, step: int, description: str, trace_id: str = None):
        """Emit workflow step progress"""
        await self.emit(EventType.WORKFLOW_STEP, {
            "workflow_name": workflow_name,
            "step": step,
            "description": description
        }, trace_id)

    async def emit_tool_calling(self, tool_name: str, skill_name: str, args: Dict, call_id: str = None, trace_id: str = None):
        """Emit tool call start"""
        call_id = call_id or f"call_{int(time.time() * 1000)}"

        self._active_tools[call_id] = {
            "call_id": call_id,
            "tool_name": tool_name,
            "skill_name": skill_name,
            "args": args,
            "start_time": time.time()
        }

        await self.emit(EventType.TOOL_CALLING, {
            "call_id": call_id,
            "tool_name": tool_name,
            "skill_name": skill_name,
            "args": args
        }, trace_id)

        return call_id

    async def emit_tool_result(self, call_id: str, result: str, duration_ms: int = None, trace_id: str = None):
        """Emit tool result"""
        tool_info = self._active_tools.pop(call_id, {})

        if duration_ms is None and tool_info.get("start_time"):
            duration_ms = int((time.time() - tool_info["start_time"]) * 1000)

        await self.emit(EventType.TOOL_RESULT, {
            "call_id": call_id,
            "tool_name": tool_info.get("tool_name", "unknown"),
            "result_preview": result[:500] if result else "",
            "result_length": len(result) if result else 0,
            "duration_ms": duration_ms
        }, trace_id)

    async def emit_tool_error(self, call_id: str, error: str, trace_id: str = None):
        """Emit tool error"""
        tool_info = self._active_tools.pop(call_id, {})

        await self.emit(EventType.TOOL_ERROR, {
            "call_id": call_id,
            "tool_name": tool_info.get("tool_name", "unknown"),
            "error": error
        }, trace_id)

    async def emit_llm_calling(self, model: str, message_count: int, messages: list = None, system_prompt: str = None, trace_id: str = None):
        """Emit LLM call start with full message context"""
        # Format messages for display - include role and content preview
        formatted_messages = []
        if messages:
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate very long messages for display but keep substantial context
                if isinstance(content, str):
                    preview = content[:2000] + "..." if len(content) > 2000 else content
                else:
                    preview = str(content)[:2000]
                formatted_messages.append({
                    "role": role,
                    "content": preview,
                    "full_length": len(str(content)) if content else 0
                })

        await self.emit(EventType.LLM_CALLING, {
            "model": model,
            "message_count": message_count,
            "messages": formatted_messages,
            "system_prompt": system_prompt[:2000] + "..." if system_prompt and len(system_prompt) > 2000 else system_prompt,
            "system_prompt_length": len(system_prompt) if system_prompt else 0
        }, trace_id)

    async def emit_llm_streaming(self, chunk: str, trace_id: str = None):
        """Emit LLM streaming chunk (for real-time display)"""
        await self.emit(EventType.LLM_STREAMING, {
            "chunk": chunk
        }, trace_id)

    async def emit_llm_complete(self, response_length: int, duration_ms: int, tool_calls: int = 0, trace_id: str = None):
        """Emit LLM call complete"""
        await self.emit(EventType.LLM_COMPLETE, {
            "response_length": response_length,
            "duration_ms": duration_ms,
            "tool_calls_found": tool_calls
        }, trace_id)

    async def emit_research_started(self, query: str, trace_id: str = None):
        """Emit research start"""
        await self.emit(EventType.RESEARCH_STARTED, {
            "query": query
        }, trace_id)

    async def emit_research_query(self, query: str, angle: str = None, trace_id: str = None):
        """Emit individual research query"""
        await self.emit(EventType.RESEARCH_QUERY, {
            "query": query,
            "angle": angle
        }, trace_id)

    async def emit_research_fetching(self, url: str, title: str = None, trace_id: str = None):
        """Emit URL fetch start"""
        await self.emit(EventType.RESEARCH_FETCHING, {
            "url": url,
            "title": title
        }, trace_id)

    async def emit_research_complete(self, source_count: int, queries: int, trace_id: str = None):
        """Emit research complete"""
        await self.emit(EventType.RESEARCH_COMPLETE, {
            "source_count": source_count,
            "query_count": queries
        }, trace_id)

    async def emit_research_saved(self, topic: str, filename: str, source_count: int, trace_id: str = None):
        """Emit research platform saved"""
        await self.emit(EventType.RESEARCH_SAVED, {
            "topic": topic,
            "filename": filename,
            "source_count": source_count
        }, trace_id)

    # === Crawl4AI Events ===

    async def emit_crawl_fetch_started(self, url: str, stealth_mode: bool = False, trace_id: str = None):
        """Emit crawl fetch started"""
        self._stats["crawl_fetches"] = self._stats.get("crawl_fetches", 0) + 1
        await self.emit(EventType.CRAWL_FETCH_STARTED, {
            "url": url,
            "stealth_mode": stealth_mode
        }, trace_id)

    async def emit_crawl_fetch_complete(self, url: str, content_length: int, duration_ms: int, title: str = None, trace_id: str = None):
        """Emit crawl fetch complete"""
        await self.emit(EventType.CRAWL_FETCH_COMPLETE, {
            "url": url,
            "content_length": content_length,
            "duration_ms": duration_ms,
            "title": title
        }, trace_id)

    async def emit_crawl_fetch_error(self, url: str, error: str, trace_id: str = None):
        """Emit crawl fetch error"""
        self._stats["crawl_errors"] = self._stats.get("crawl_errors", 0) + 1
        await self.emit(EventType.CRAWL_FETCH_ERROR, {
            "url": url,
            "error": error
        }, trace_id)

    async def emit_crawl_deep_started(self, url: str, keywords: list, max_depth: int, max_pages: int, strategy: str = "bfs", trace_id: str = None):
        """Emit deep crawl started"""
        self._stats["deep_crawls"] = self._stats.get("deep_crawls", 0) + 1
        await self.emit(EventType.CRAWL_DEEP_STARTED, {
            "url": url,
            "keywords": keywords,
            "max_depth": max_depth,
            "max_pages": max_pages,
            "strategy": strategy
        }, trace_id)

    async def emit_crawl_deep_page(self, url: str, depth: int, score: float, content_length: int, page_num: int, trace_id: str = None):
        """Emit deep crawl page fetched"""
        await self.emit(EventType.CRAWL_DEEP_PAGE, {
            "url": url,
            "depth": depth,
            "score": score,
            "content_length": content_length,
            "page_num": page_num
        }, trace_id)

    async def emit_crawl_deep_complete(self, seed_url: str, pages_crawled: int, duration_ms: int, trace_id: str = None):
        """Emit deep crawl complete"""
        await self.emit(EventType.CRAWL_DEEP_COMPLETE, {
            "seed_url": seed_url,
            "pages_crawled": pages_crawled,
            "duration_ms": duration_ms
        }, trace_id)

    async def emit_crawl_stealth_enabled(self, url: str, undetected_mode: bool = False, trace_id: str = None):
        """Emit stealth mode enabled for fetch"""
        await self.emit(EventType.CRAWL_STEALTH_ENABLED, {
            "url": url,
            "undetected_mode": undetected_mode
        }, trace_id)

    async def emit_crawl_domain_blocked(self, url: str, domain: str, reason: str, trace_id: str = None):
        """Emit domain blocked event"""
        self._stats["domains_blocked"] = self._stats.get("domains_blocked", 0) + 1
        await self.emit(EventType.CRAWL_DOMAIN_BLOCKED, {
            "url": url,
            "domain": domain,
            "reason": reason
        }, trace_id)

    async def emit_crawl_parallel_started(self, urls: list, max_concurrent: int, trace_id: str = None):
        """Emit parallel fetch started"""
        await self.emit(EventType.CRAWL_PARALLEL_STARTED, {
            "url_count": len(urls),
            "urls": urls[:5],  # Preview first 5
            "max_concurrent": max_concurrent
        }, trace_id)

    async def emit_crawl_parallel_complete(self, total_urls: int, successful: int, failed: int, blocked: int, duration_ms: int, trace_id: str = None):
        """Emit parallel fetch complete"""
        await self.emit(EventType.CRAWL_PARALLEL_COMPLETE, {
            "total_urls": total_urls,
            "successful": successful,
            "failed": failed,
            "blocked": blocked,
            "duration_ms": duration_ms
        }, trace_id)

    async def emit_subagent_spawning(self, agent_name: str, model: str, task: str, trace_id: str = None):
        """Emit subagent spawning event"""
        self._stats["subagent_calls"] = self._stats.get("subagent_calls", 0) + 1
        await self.emit(EventType.SUBAGENT_SPAWNING, {
            "agent_name": agent_name,
            "model": model,
            "task": task
        }, trace_id)

    async def emit_subagent_model_swap(self, from_model: str, to_model: str, action: str, trace_id: str = None):
        """Emit model swap event (loading/unloading)"""
        self._stats["model_swaps"] = self._stats.get("model_swaps", 0) + 1
        await self.emit(EventType.SUBAGENT_MODEL_SWAP, {
            "from_model": from_model,
            "to_model": to_model,
            "action": action  # "unloading", "loading"
        }, trace_id)

    async def emit_subagent_executing(self, agent_name: str, model: str, trace_id: str = None):
        """Emit subagent executing event"""
        await self.emit(EventType.SUBAGENT_EXECUTING, {
            "agent_name": agent_name,
            "model": model
        }, trace_id)

    async def emit_subagent_complete(self, agent_name: str, model: str, duration_ms: int, output_length: int, success: bool, trace_id: str = None):
        """Emit subagent completion event"""
        await self.emit(EventType.SUBAGENT_COMPLETE, {
            "agent_name": agent_name,
            "model": model,
            "duration_ms": duration_ms,
            "output_length": output_length,
            "success": success
        }, trace_id)

    async def emit_context_snapshot(self, snapshot_id: str, research_topic: str = None, trace_id: str = None):
        """Emit context snapshot created event"""
        await self.emit(EventType.CONTEXT_SNAPSHOT, {
            "snapshot_id": snapshot_id,
            "research_topic": research_topic
        }, trace_id)

    async def emit_response(self, text: str, trace_id: str = None):
        """Emit assistant response"""
        await self.emit(EventType.ASSISTANT_RESPONSE, {
            "text": text[:1000],  # Preview only
            "length": len(text)
        }, trace_id)
        self._current_trace_id = None

    async def emit_error(self, error: str, stage: str = None, trace_id: str = None):
        """Emit error event"""
        await self.emit(EventType.ERROR, {
            "error": error,
            "stage": stage
        }, trace_id)

    async def emit_warning(self, message: str, trace_id: str = None):
        """Emit warning event"""
        await self.emit(EventType.WARNING, {
            "message": message
        }, trace_id)

    async def emit_info(self, message: str, trace_id: str = None):
        """Emit info event"""
        await self.emit(EventType.INFO, {
            "message": message
        }, trace_id)

    # === Task Management Events ===

    async def emit_task_list_updated(self, tasks: List[Dict], original_request: str = "", trace_id: str = None):
        """Emit task list updated event (called when tasks are written)"""
        completed = sum(1 for t in tasks if t.get("status") == "completed")
        in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")
        pending = sum(1 for t in tasks if t.get("status") == "pending")

        await self.emit(EventType.TASK_LIST_UPDATED, {
            "tasks": tasks,
            "original_request": original_request,
            "stats": {
                "total": len(tasks),
                "completed": completed,
                "in_progress": in_progress,
                "pending": pending
            }
        }, trace_id)

    async def emit_task_started(self, task_content: str, task_active_form: str, task_index: int, trace_id: str = None):
        """Emit task started event (when a task transitions to in_progress)"""
        await self.emit(EventType.TASK_STARTED, {
            "content": task_content,
            "active_form": task_active_form,
            "index": task_index
        }, trace_id)

    async def emit_task_completed(self, task_content: str, task_index: int, trace_id: str = None):
        """Emit task completed event"""
        await self.emit(EventType.TASK_COMPLETED, {
            "content": task_content,
            "index": task_index
        }, trace_id)

    async def emit_task_cleared(self, task_count: int, trace_id: str = None):
        """Emit task cleared event"""
        await self.emit(EventType.TASK_CLEARED, {
            "cleared_count": task_count
        }, trace_id)

    def get_stats(self) -> Dict:
        """Get current dashboard stats"""
        return {
            **self._stats,
            "connected_clients": len(self.clients),
            "history_size": len(self._event_history),
            "active_tools": len(self._active_tools),
            "uptime_seconds": int(time.time() - self._session_start) if self._session_start else 0
        }


# Singleton instance
_dashboard: Optional[DashboardServer] = None


def get_dashboard() -> DashboardServer:
    """Get the global dashboard server instance"""
    global _dashboard
    if _dashboard is None:
        _dashboard = DashboardServer()
    return _dashboard


async def start_dashboard(host: str = "localhost", port: int = 8766) -> DashboardServer:
    """Start the dashboard server and return it"""
    global _dashboard
    _dashboard = DashboardServer(host=host, port=port)
    await _dashboard.start()
    return _dashboard
