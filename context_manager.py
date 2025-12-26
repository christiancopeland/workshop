"""
Workshop Context Manager - Phase 3
Intelligent context assembly, file monitoring, and workflow detection
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import threading

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from logger import get_logger

log = get_logger("context_manager")

# this is a comment for watchdog
@dataclass
class FileEdit:
    """Record of a file edit event"""
    path: Path
    timestamp: datetime
    edit_type: str  # 'create', 'modify', 'delete'
    project: str = ""

    @property
    def age_seconds(self) -> float:
        """How long ago was this edit"""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class FileNode:
    """Node in the context graph representing a file"""
    path: Path
    project: str
    language: str  # 'cpp', 'python', 'javascript', etc.
    imports: Set[str] = field(default_factory=set)
    imported_by: Set[str] = field(default_factory=set)
    symbols_defined: Set[str] = field(default_factory=set)
    last_modified: Optional[datetime] = None
    edit_count: int = 0


class FileWatcher(FileSystemEventHandler):
    """
    Monitor project directories for file changes using watchdog.
    Debounces rapid changes and notifies ContextGraph.
    """

    def __init__(self, context_graph: 'ContextGraph', debounce_seconds: float = 1.0):
        self.context_graph = context_graph
        self.debounce_seconds = debounce_seconds
        self.observer = Observer()
        self.watched_paths: Set[Path] = set()

        # Debouncing: track pending events
        self._pending_events: Dict[str, Tuple[FileSystemEvent, float]] = {}
        self._debounce_lock = threading.Lock()
        self._debounce_thread = None
        self._running = False

        log.info("FileWatcher initialized")

    def watch_directory(self, path: Path):
        """Start watching a directory"""
        if not path.exists() or not path.is_dir():
            log.warning(f"Cannot watch {path}: not a valid directory")
            return

        if path in self.watched_paths:
            log.debug(f"Already watching {path}")
            return

        try:
            self.observer.schedule(self, str(path), recursive=True)
            self.watched_paths.add(path)
            log.info(f"Now watching: {path}")
        except Exception as e:
            log.error(f"Failed to watch {path}: {e}")

    def start(self):
        """Start the file watcher"""
        if not self._running:
            self._running = True
            self.observer.start()
            self._start_debounce_processor()
            log.info("FileWatcher started")

    def stop(self):
        """Stop the file watcher"""
        if self._running:
            self._running = False
            self.observer.stop()
            self.observer.join()
            log.info("FileWatcher stopped")

    def _start_debounce_processor(self):
        """Start background thread to process debounced events"""
        def process_loop():
            while self._running:
                time.sleep(0.5)  # Check every 500ms
                self._process_pending_events()

        self._debounce_thread = threading.Thread(target=process_loop, daemon=True)
        self._debounce_thread.start()

    def _process_pending_events(self):
        """Process events that have passed the debounce window"""
        now = time.time()
        to_process = []

        with self._debounce_lock:
            for key, (event, timestamp) in list(self._pending_events.items()):
                if now - timestamp >= self.debounce_seconds:
                    to_process.append(event)
                    del self._pending_events[key]

        # Process outside the lock
        for event in to_process:
            self._handle_event(event)

    def _add_pending_event(self, event: FileSystemEvent):
        """Add event to pending queue (debounced)"""
        key = event.src_path
        with self._debounce_lock:
            self._pending_events[key] = (event, time.time())

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory:
            self._add_pending_event(event)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self._add_pending_event(event)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory:
            self._add_pending_event(event)

    def _handle_event(self, event: FileSystemEvent):
        """Handle a debounced file system event"""
        path = Path(event.src_path)

        # Determine edit type
        if event.event_type == 'created':
            edit_type = 'create'
        elif event.event_type == 'deleted':
            edit_type = 'delete'
        else:
            edit_type = 'modify'

        # Notify context graph
        file_edit = FileEdit(
            path=path,
            timestamp=datetime.now(),
            edit_type=edit_type
        )

        self.context_graph.on_file_edited(file_edit)
        log.debug(f"File {edit_type}: {path.name}")


class ContextGraph:
    """
    Build and maintain a graph of file relationships.
    Tracks: imports, dependencies, co-edited files.
    """

    def __init__(self, ignore_dirs: Set[str] = None):
        self.nodes: Dict[str, FileNode] = {}  # path -> FileNode
        self.recent_edits: List[FileEdit] = []
        self.co_occurrence_matrix: Dict[Tuple[str, str], int] = defaultdict(int)

        self.ignore_dirs = ignore_dirs or {
            '__pycache__', 'node_modules', '.git', 'venv', '.venv',
            'build', 'dist', '.DS_Store', '.pytest_cache', 'target'
        }

        # Also ignore log files, databases, and temp files
        self.ignore_patterns = {'.log', '.db', '.db-journal', '.tmp', '.swp', '.pyc'}

        log.info("ContextGraph initialized")

    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored"""
        # Check if any parent directory is in ignore list
        for part in path.parts:
            if part in self.ignore_dirs or part.startswith('.'):
                return True

        # Check file extension patterns
        if path.suffix in self.ignore_patterns:
            return True

        # Ignore files in 'logs' or 'data' directories
        if 'logs' in path.parts or 'data' in path.parts:
            return True

        return False

    def on_file_edited(self, file_edit: FileEdit):
        """Handle a file edit event"""
        if self.should_ignore(file_edit.path):
            return

        # Add to recent edits
        self.recent_edits.append(file_edit)

        # Keep only last 100 edits
        if len(self.recent_edits) > 100:
            self.recent_edits = self.recent_edits[-100:]

        # Update or create node
        path_str = str(file_edit.path)

        if file_edit.edit_type == 'delete':
            if path_str in self.nodes:
                del self.nodes[path_str]
        else:
            if path_str not in self.nodes:
                self.nodes[path_str] = FileNode(
                    path=file_edit.path,
                    project=file_edit.project,
                    language=self._detect_language(file_edit.path),
                )

            node = self.nodes[path_str]
            node.last_modified = file_edit.timestamp
            node.edit_count += 1

            # Parse imports if it's a code file
            if file_edit.edit_type in ('create', 'modify'):
                self._parse_imports(node)

        # Update co-occurrence matrix
        self._update_co_occurrence()

    def _detect_language(self, path: Path) -> str:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'python',
            '.ino': 'cpp',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'cpp',
            '.hpp': 'cpp',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.md': 'markdown',
            '.txt': 'text',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
        }
        return ext_map.get(path.suffix.lower(), 'unknown')

    def _parse_imports(self, node: FileNode):
        """Parse imports/includes from a file"""
        if not node.path.exists():
            return

        try:
            content = node.path.read_text(errors='ignore')

            # Reset imports
            node.imports.clear()

            if node.language == 'python':
                # Parse Python imports
                import re
                for match in re.finditer(r'^\s*(?:from\s+(\S+)\s+)?import\s+(.+)', content, re.MULTILINE):
                    if match.group(1):
                        node.imports.add(match.group(1))
                    else:
                        # Handle "import foo, bar"
                        for imp in match.group(2).split(','):
                            node.imports.add(imp.strip().split()[0])

            elif node.language in ('cpp', 'c'):
                # Parse C/C++ includes
                import re
                for match in re.finditer(r'^\s*#include\s+[<"]([^>"]+)[>"]', content, re.MULTILINE):
                    node.imports.add(match.group(1))

            elif node.language in ('javascript', 'typescript'):
                # Parse JS/TS imports
                import re
                for match in re.finditer(r'^\s*import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content, re.MULTILINE):
                    node.imports.add(match.group(1))

            # Update reverse relationships (imported_by)
            for imported_file in node.imports:
                # Try to resolve to actual file path
                resolved = self._resolve_import(imported_file, node.path)
                if resolved and resolved in self.nodes:
                    self.nodes[resolved].imported_by.add(str(node.path))

        except Exception as e:
            log.debug(f"Failed to parse imports for {node.path}: {e}")

    def _resolve_import(self, import_name: str, from_file: Path) -> Optional[str]:
        """Try to resolve an import to an actual file path"""
        # For local imports, try to find the file
        base_dir = from_file.parent

        # Try common patterns
        candidates = [
            base_dir / import_name,
            base_dir / f"{import_name}.py",
            base_dir / f"{import_name}.h",
            base_dir / f"{import_name}.cpp",
            base_dir / f"{import_name}.js",
            base_dir / import_name / "__init__.py",
        ]

        for candidate in candidates:
            if candidate.exists() and str(candidate) in self.nodes:
                return str(candidate)

        return None

    def _update_co_occurrence(self):
        """Update co-occurrence matrix based on recent edits"""
        # Look at files edited in the last 5 minutes
        recent = [e for e in self.recent_edits if e.age_seconds < 300]

        if len(recent) < 2:
            return

        # Get unique file paths
        files = list(set(str(e.path) for e in recent))

        # Update co-occurrence counts
        for i, file1 in enumerate(files):
            for file2 in files[i+1:]:
                key = tuple(sorted([file1, file2]))
                self.co_occurrence_matrix[key] += 1

    def get_related_files(self, file_path: Path, max_results: int = 10) -> List[Tuple[Path, float]]:
        """
        Get files related to the given file.
        Returns list of (path, relevance_score) tuples.
        """
        path_str = str(file_path)

        if path_str not in self.nodes:
            return []

        node = self.nodes[path_str]
        related: Dict[str, float] = {}

        # 1. Files this imports (score: 0.8)
        for imported in node.imports:
            resolved = self._resolve_import(imported, file_path)
            if resolved:
                related[resolved] = related.get(resolved, 0) + 0.8

        # 2. Files that import this (score: 0.7)
        for importer in node.imported_by:
            related[importer] = related.get(importer, 0) + 0.7

        # 3. Co-edited files (score based on count)
        for (f1, f2), count in self.co_occurrence_matrix.items():
            if f1 == path_str:
                related[f2] = related.get(f2, 0) + min(count * 0.1, 0.6)
            elif f2 == path_str:
                related[f1] = related.get(f1, 0) + min(count * 0.1, 0.6)

        # Convert to list and sort by score
        results = [(Path(p), score) for p, score in related.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:max_results]

    def get_recent_edits(self, limit: int = 20, max_age_seconds: int = 3600) -> List[FileEdit]:
        """Get recent file edits"""
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        recent = [e for e in self.recent_edits if e.timestamp >= cutoff]
        return recent[-limit:]

    def get_active_files(self, max_age_seconds: int = 300) -> List[Path]:
        """Get files edited in the last N seconds (default 5 minutes)"""
        recent = self.get_recent_edits(max_age_seconds=max_age_seconds)
        # Return unique paths, most recent first
        seen = set()
        active = []
        for edit in reversed(recent):
            if str(edit.path) not in seen:
                active.append(edit.path)
                seen.add(str(edit.path))
        return active

    def scan_project(self, project_path: Path, extensions: Set[str]):
        """
        Initial scan of a project directory.
        Builds the context graph for existing files.
        """
        log.info(f"Scanning project: {project_path}")

        count = 0
        for file_path in project_path.rglob("*"):
            if not file_path.is_file():
                continue

            if self.should_ignore(file_path):
                continue

            if file_path.suffix.lower() not in extensions:
                continue

            # Create node
            path_str = str(file_path)
            if path_str not in self.nodes:
                self.nodes[path_str] = FileNode(
                    path=file_path,
                    project=project_path.name,
                    language=self._detect_language(file_path),
                    last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
                )
                count += 1

                # Parse imports
                self._parse_imports(self.nodes[path_str])

        log.info(f"Scanned {count} files in {project_path.name}")


class WorkflowDetector:
    """
    Detect development workflow patterns based on file activity.
    Recognizes: debugging, feature development, configuration editing, research.
    """

    # Workflow definitions with indicators and weights
    WORKFLOWS = {
        'debugging': {
            'description': 'Debugging hardware/software issues',
            'indicators': {
                'repeated_edits_same_file': 0.3,      # Same file edited 3+ times
                'serial_monitor_activity': 0.2,        # Serial port access (Arduino)
                'config_file_tweaks': 0.2,             # Calibration/config changes
                'small_frequent_changes': 0.2,         # Many small edits
                'error_search_keywords': 0.1,          # Searches for "error", "fix", etc.
            },
            'context_priority': ['error_logs', 'recent_changes', 'datasheets'],
        },
        'feature_development': {
            'description': 'Adding new features or functionality',
            'indicators': {
                'new_files_created': 0.4,              # New .cpp/.h/.py files
                'multiple_files_edited': 0.3,          # 3+ different files
                'test_files_modified': 0.2,            # Tests updated
                'header_and_impl_pairs': 0.1,          # .h and .cpp together
            },
            'context_priority': ['related_files', 'api_docs', 'similar_code'],
        },
        'configuration': {
            'description': 'Adjusting settings and calibration',
            'indicators': {
                'config_files_only': 0.5,              # Only .h/.json/.yaml modified
                'no_code_changes': 0.3,                # No .cpp/.py changes
                'repeated_value_changes': 0.2,         # Same lines edited repeatedly
            },
            'context_priority': ['config_schema', 'valid_ranges', 'examples'],
        },
        'research': {
            'description': 'Exploring codebase or learning',
            'indicators': {
                'many_files_opened': 0.3,              # 5+ files opened (no edits)
                'web_searches': 0.3,                   # Documentation lookups
                'read_only_activity': 0.2,             # Reads > Writes
                'diverse_file_types': 0.2,             # .md, .txt, code mixed
            },
            'context_priority': ['documentation', 'examples', 'related_concepts'],
        },
    }

    def __init__(self):
        self.current_workflow: Optional[str] = None
        self.confidence: float = 0.0
        self.workflow_start_time: Optional[datetime] = None
        log.info("WorkflowDetector initialized")

    def detect_workflow(self, recent_edits: List[FileEdit], lookback_minutes: int = 15) -> Tuple[str, float]:
        """
        Analyze recent activity and return detected workflow.

        Args:
            recent_edits: List of recent file edits
            lookback_minutes: How far back to analyze

        Returns:
            (workflow_name, confidence) tuple
            confidence is 0.0-1.0 (>0.7 is considered confident)
        """
        if not recent_edits:
            return 'unknown', 0.0

        # Filter to recent activity
        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        recent = [e for e in recent_edits if e.timestamp >= cutoff]

        if not recent:
            return 'unknown', 0.0

        # Calculate scores for each workflow
        scores = {name: 0.0 for name in self.WORKFLOWS}

        for workflow_name, workflow_config in self.WORKFLOWS.items():
            indicators = workflow_config['indicators']
            total_weight = sum(indicators.values())

            # Check each indicator
            if self._check_repeated_edits(recent):
                scores[workflow_name] += indicators.get('repeated_edits_same_file', 0)

            if self._check_config_files_only(recent):
                scores[workflow_name] += indicators.get('config_files_only', 0)
                scores[workflow_name] += indicators.get('no_code_changes', 0)

            if self._check_new_files_created(recent):
                scores[workflow_name] += indicators.get('new_files_created', 0)

            if self._check_multiple_files_edited(recent):
                scores[workflow_name] += indicators.get('multiple_files_edited', 0)

            if self._check_config_file_tweaks(recent):
                scores[workflow_name] += indicators.get('config_file_tweaks', 0)

            if self._check_small_frequent_changes(recent):
                scores[workflow_name] += indicators.get('small_frequent_changes', 0)

            # Normalize score
            if total_weight > 0:
                scores[workflow_name] = scores[workflow_name] / total_weight

        # Find best match
        if not scores:
            return 'unknown', 0.0

        best_workflow = max(scores, key=scores.get)
        confidence = scores[best_workflow]

        # Update tracking if confident
        if confidence >= 0.5:
            if self.current_workflow != best_workflow:
                self.current_workflow = best_workflow
                self.workflow_start_time = datetime.now()
            self.confidence = confidence
        else:
            best_workflow = 'unknown'

        log.debug(f"Workflow detected: {best_workflow} (confidence: {confidence:.2f})")
        return best_workflow, confidence

    def _check_repeated_edits(self, edits: List[FileEdit]) -> bool:
        """Check if same file edited multiple times"""
        file_counts = defaultdict(int)
        for edit in edits:
            if edit.edit_type == 'modify':
                file_counts[str(edit.path)] += 1
        return any(count >= 3 for count in file_counts.values())

    def _check_config_files_only(self, edits: List[FileEdit]) -> bool:
        """Check if only config files were modified"""
        config_extensions = {'.h', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
        for edit in edits:
            if edit.path.suffix not in config_extensions:
                return False
        return True

    def _check_new_files_created(self, edits: List[FileEdit]) -> bool:
        """Check if new files were created"""
        return any(e.edit_type == 'create' for e in edits)

    def _check_multiple_files_edited(self, edits: List[FileEdit]) -> bool:
        """Check if 3+ different files were edited"""
        unique_files = set(str(e.path) for e in edits)
        return len(unique_files) >= 3

    def _check_config_file_tweaks(self, edits: List[FileEdit]) -> bool:
        """Check for config file modifications"""
        config_extensions = {'.h', '.json', '.yaml', '.yml', '.toml'}
        config_edits = [e for e in edits if e.path.suffix in config_extensions]
        return len(config_edits) > 0

    def _check_small_frequent_changes(self, edits: List[FileEdit]) -> bool:
        """Check for many small edits in short time"""
        if len(edits) < 5:
            return False

        # Check time span
        if edits:
            time_span = (edits[-1].timestamp - edits[0].timestamp).total_seconds()
            return time_span < 300  # 5 minutes

        return False

    def get_context_priorities(self, workflow: str = None) -> List[str]:
        """Get context priorities for a workflow"""
        if workflow is None:
            workflow = self.current_workflow or 'unknown'

        if workflow in self.WORKFLOWS:
            return self.WORKFLOWS[workflow]['context_priority']

        return []


class ContextManager:
    """
    Main orchestrator for Phase 3 context intelligence.
    Manages FileWatcher, ContextGraph, and context assembly.
    """

    def __init__(
        self,
        monitored_projects: List[Path],
        indexable_extensions: Set[str],
        memory_system: 'MemorySystem' = None,
    ):
        self.monitored_projects = monitored_projects
        self.indexable_extensions = indexable_extensions
        self.memory = memory_system

        # Initialize components
        self.context_graph = ContextGraph()
        self.file_watcher = FileWatcher(self.context_graph)
        self.workflow_detector = WorkflowDetector()

        # Current session state
        self.current_retrievable: Dict = {}
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize monitoring
        self._initialize_monitoring()

        log.info(f"ContextManager initialized with {len(monitored_projects)} projects")

    def _initialize_monitoring(self):
        """Initialize file watching and scan existing projects"""
        for project_path in self.monitored_projects:
            if not project_path.exists():
                log.warning(f"Project path does not exist: {project_path}")
                continue

            # Scan existing files
            self.context_graph.scan_project(project_path, self.indexable_extensions)

            # Start watching
            self.file_watcher.watch_directory(project_path)

        # Start the watcher
        self.file_watcher.start()

    def stop(self):
        """Stop file watching"""
        self.file_watcher.stop()

    def assemble_context(self, user_query: str) -> Dict:
        """
        Assemble relevant context for a user query.

        Returns:
            {
                'critical': {  # For system prompt
                    'active_files': [...],
                    'recent_changes': [...],
                    'detected_workflow': '...',
                },
                'retrievable': {  # Available via tools
                    'related_files': {...},
                    'file_graph': {...},
                }
            }
        """
        context = {
            'critical': {},
            'retrievable': {}
        }

        # Get active files (edited in last 5 minutes)
        active_files = self.context_graph.get_active_files(max_age_seconds=300)

        if active_files:
            context['critical']['active_files'] = [
                {
                    'path': str(f),
                    'name': f.name,
                    'age': self._get_file_age(f)
                }
                for f in active_files[:5]  # Top 5
            ]

        # Get recent changes
        recent_edits = self.context_graph.get_recent_edits(limit=10)
        if recent_edits:
            context['critical']['recent_changes'] = [
                {
                    'file': e.path.name,
                    'type': e.edit_type,
                    'age': f"{int(e.age_seconds)}s ago"
                }
                for e in recent_edits[:5]
            ]

        # Detect workflow
        workflow, confidence = self.workflow_detector.detect_workflow(recent_edits, lookback_minutes=15)
        if workflow != 'unknown' and confidence >= 0.5:
            context['critical']['detected_workflow'] = {
                'type': workflow,
                'confidence': f"{int(confidence * 100)}%",
                'description': self.workflow_detector.WORKFLOWS[workflow]['description']
            }

            # Save to memory if we have it
            if self.memory and confidence >= 0.7:
                # High confidence - start tracking this workflow session
                self.memory.start_workflow_session(self.session_id, workflow)

        # Build related files map for retrievable context
        related_files = {}
        for active_file in active_files[:3]:  # Top 3 active files
            related = self.context_graph.get_related_files(active_file, max_results=5)
            if related:
                related_files[str(active_file)] = [
                    {'path': str(p), 'score': score}
                    for p, score in related
                ]

        context['retrievable']['related_files'] = related_files

        # Log file edits to memory
        if self.memory:
            for edit in recent_edits[-5:]:  # Last 5 edits
                self.memory.log_file_edit(
                    str(edit.path),
                    edit.edit_type,
                    edit.project
                )

        return context

    def _get_file_age(self, file_path: Path) -> str:
        """Get human-readable age of file's last modification"""
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_seconds = (datetime.now() - mtime).total_seconds()

            if age_seconds < 60:
                return f"{int(age_seconds)}s ago"
            elif age_seconds < 3600:
                return f"{int(age_seconds / 60)}m ago"
            elif age_seconds < 86400:
                return f"{int(age_seconds / 3600)}h ago"
            else:
                return f"{int(age_seconds / 86400)}d ago"
        except:
            return "unknown"

    def get_context(self) -> Dict:
        """
        Compatibility method for Phase 2 agent integration.
        Returns assembled context dict.
        """
        return self.assemble_context("")

    def format_context_for_llm(self, context: Dict = None) -> str:
        """Format context dict as string for LLM system prompt"""
        if context is None:
            context = self.get_context()

        if not context.get('critical'):
            return ""

        lines = []
        critical = context['critical']

        if critical.get('active_files'):
            lines.append("Active Files:")
            for f in critical['active_files']:
                lines.append(f"  • {f['name']} ({f['age']})")

        if critical.get('recent_changes'):
            lines.append("\nRecent Changes:")
            for c in critical['recent_changes']:
                lines.append(f"  • {c['type']}: {c['file']} ({c['age']})")

        if critical.get('detected_workflow'):
            wf = critical['detected_workflow']
            lines.append(f"\nDetected Workflow: {wf['type']} ({wf['confidence']})")
            lines.append(f"  {wf['description']}")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get statistics about the context system"""
        return {
            'monitored_projects': len(self.monitored_projects),
            'watched_paths': len(self.file_watcher.watched_paths),
            'indexed_files': len(self.context_graph.nodes),
            'recent_edits': len(self.context_graph.recent_edits),
            'active_files': len(self.context_graph.get_active_files()),
        }


# === Singleton instance ===
_context_manager_instance: Optional[ContextManager] = None


def get_context_manager_v3(
    monitored_projects: List[Path] = None,
    indexable_extensions: Set[str] = None,
    memory_system: 'MemorySystem' = None,
) -> ContextManager:
    """Get or create the global Phase 3 context manager"""
    global _context_manager_instance

    if _context_manager_instance is None:
        if monitored_projects is None:
            raise ValueError("monitored_projects required for first initialization")

        _context_manager_instance = ContextManager(
            monitored_projects=monitored_projects,
            indexable_extensions=indexable_extensions or {'.py', '.ino', '.cpp', '.h'},
            memory_system=memory_system,
        )

    return _context_manager_instance
