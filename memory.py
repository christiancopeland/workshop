"""
Workshop Memory System
ChromaDB for semantic search + SQLite for structured data
"""

import json
import sqlite3
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import concurrent.futures

from logger import get_logger

log = get_logger("memory")


class MemorySystem:
    """
    Tiered memory system:
    - Immediate: current conversation (in-memory list)
    - Session: SQLite for structured facts and session data
    - Long-term: ChromaDB for semantic search over memories
    """
    
    def __init__(self, chroma_path: Path, sqlite_path: Path):
        self.chroma_path = chroma_path
        self.sqlite_path = sqlite_path
        
        # Immediate memory (current conversation)
        self._messages: List[Dict[str, str]] = []
        self._session_start = datetime.now()
        self.message_count = 0
        self._current_session_id: Optional[str] = None  # Set via start_session()
        
        # Index cache - track what's already indexed
        self._indexed_files: Dict[str, float] = {}  # path -> mtime
        
        # Initialize stores
        self._init_sqlite()
        self._init_chroma()
        
        # Load index cache
        self._load_index_cache()
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY,
                content TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE,
                value TEXT,
                category TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                message_count INTEGER,
                summary TEXT
            );
            
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS indexed_files (
                path TEXT PRIMARY KEY,
                mtime REAL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                path TEXT,
                description TEXT,
                last_active TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS project_notes (
                id INTEGER PRIMARY KEY,
                project_id INTEGER,
                content TEXT,
                category TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            
            CREATE TABLE IF NOT EXISTS project_sessions (
                id INTEGER PRIMARY KEY,
                project_id INTEGER,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                summary TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            
            CREATE TABLE IF NOT EXISTS active_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Phase 3: Context Intelligence Tables
            CREATE TABLE IF NOT EXISTS session_context (
                session_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                active_files TEXT,  -- JSON array of file paths
                detected_workflow TEXT,
                project_focus TEXT,
                PRIMARY KEY (session_id, timestamp)
            );

            CREATE TABLE IF NOT EXISTS file_edits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                edit_type TEXT NOT NULL,  -- 'create', 'modify', 'delete'
                project TEXT,
                file_size INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_file_edits_path ON file_edits(file_path);
            CREATE INDEX IF NOT EXISTS idx_file_edits_timestamp ON file_edits(timestamp);

            CREATE TABLE IF NOT EXISTS workflow_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                workflow_type TEXT,  -- 'debugging', 'feature_dev', 'config_edit', 'research'
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                primary_files TEXT,  -- JSON array of main files worked on
                outcome TEXT,  -- 'completed', 'abandoned', 'ongoing'
                notes TEXT
            );

            INSERT OR IGNORE INTO user_profile (id, content) VALUES (1, '');
        """)
        self.conn.commit()
        log.debug("SQLite initialized")
    
    def _init_chroma(self):
        """Initialize ChromaDB"""
        self._chroma_client = None
        self._collection = None
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            self._collection = self._chroma_client.get_or_create_collection(
                name="workshop_memories",
                metadata={"hnsw:space": "cosine"}
            )
            
            log.info(f"ChromaDB initialized with {self._collection.count()} documents")
            
        except ImportError:
            log.warning("ChromaDB not installed - long-term memory disabled")
        except Exception as e:
            log.warning(f"ChromaDB initialization failed: {e}")
    
    def _load_index_cache(self):
        """Load the index cache from SQLite"""
        try:
            cursor = self.conn.execute("SELECT path, mtime FROM indexed_files")
            for row in cursor:
                self._indexed_files[row["path"]] = row["mtime"]
            log.debug(f"Loaded index cache: {len(self._indexed_files)} files")
        except Exception as e:
            log.warning(f"Failed to load index cache: {e}")
    
    def _save_index_cache(self, path: str, mtime: float):
        """Save a file to the index cache"""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO indexed_files (path, mtime) VALUES (?, ?)",
                (path, mtime)
            )
            self.conn.commit()
            self._indexed_files[path] = mtime
        except Exception as e:
            log.warning(f"Failed to save index cache: {e}")
    
    # === Immediate Memory (Conversation) ===
    
    def add_message(self, role: str, content: str):
        """Add a message to the current conversation"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self._messages.append(msg)
        self.message_count += 1

        # Log to SQLite with session_id
        try:
            self.conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (self._current_session_id, role, content)
            )
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to log message: {e}")
        
        # Add to long-term memory periodically
        if role == "user" and self.message_count % 3 == 0:
            self.add_memory(content, category="conversation")
    
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """Get the most recent messages"""
        return self._messages[-n:]
    
    def clear_conversation(self):
        """Clear the current conversation"""
        self._messages = []

    def start_session(self, session_id: str):
        """
        Start a new session.

        This clears the in-memory message buffer and sets the session ID
        for all future messages. Called by SessionManager on startup.
        """
        self._current_session_id = session_id
        self._messages = []  # Clear in-memory messages for fresh start
        self._session_start = datetime.now()
        self.message_count = 0
        log.info(f"Memory session started: {session_id}")

    def end_session(self) -> Dict[str, Any]:
        """
        End the current session.

        Returns summary of the session for archival.
        """
        summary = {
            "session_id": self._current_session_id,
            "started_at": self._session_start.isoformat(),
            "ended_at": datetime.now().isoformat(),
            "message_count": self.message_count
        }

        # Clear session state
        self._current_session_id = None
        self._messages = []
        self.message_count = 0

        log.info(f"Memory session ended: {summary}")
        return summary

    def get_session_id(self) -> Optional[str]:
        """Get the current session ID"""
        return self._current_session_id

    # === Session Memory (SQLite) ===
    
    def get_user_profile(self) -> Optional[str]:
        """Get the user profile"""
        try:
            row = self.conn.execute(
                "SELECT content FROM user_profile WHERE id = 1"
            ).fetchone()
            return row["content"] if row else None
        except:
            return None
    
    def set_user_profile(self, content: str):
        """Update the user profile"""
        try:
            self.conn.execute(
                "UPDATE user_profile SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                (content,)
            )
            self.conn.commit()
            log.debug(f"User profile updated: {len(content)} chars")
        except Exception as e:
            log.warning(f"Failed to update profile: {e}")
    
    def set_fact(self, key: str, value: Any, category: str = "general"):
        """Store a fact (key-value pair)"""
        value_str = json.dumps(value) if not isinstance(value, str) else value
        try:
            self.conn.execute("""
                INSERT INTO facts (key, value, category, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, value_str, category))
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to set fact: {e}")
    
    def get_fact(self, key: str) -> Optional[Any]:
        """Retrieve a fact by key"""
        try:
            row = self.conn.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
            
            if row:
                try:
                    return json.loads(row["value"])
                except:
                    return row["value"]
        except:
            pass
        return None
    
    def save_session(self):
        """Save the current session"""
        if not self._messages:
            return
        
        try:
            self.conn.execute("""
                INSERT INTO sessions (started_at, ended_at, message_count)
                VALUES (?, ?, ?)
            """, (
                self._session_start.isoformat(),
                datetime.now().isoformat(),
                len(self._messages)
            ))
            self.conn.commit()
            log.info(f"Session saved: {len(self._messages)} messages")
        except Exception as e:
            log.warning(f"Failed to save session: {e}")
    
    # === Long-term Memory (ChromaDB) ===
    
    def add_memory(self, content: str, category: str = "general", metadata: dict = None):
        """Add content to long-term semantic memory"""
        if self._collection is None or not content.strip():
            return
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        doc_id = f"{category}_{content_hash}"
        
        meta = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "length": len(content)
        }
        if metadata:
            meta.update(metadata)
        
        try:
            self._collection.upsert(
                documents=[content],
                ids=[doc_id],
                metadatas=[meta]
            )
        except Exception as e:
            log.debug(f"Failed to add memory: {e}")
    
    def add_memories_batch(self, items: list):
        """Add multiple memories in a single batch"""
        if self._collection is None or not items:
            return
        
        documents = []
        ids = []
        metadatas = []
        
        for item in items:
            content = item.get("content", "")
            if not content.strip():
                continue
                
            category = item.get("category", "general")
            metadata = item.get("metadata", {})
            
            content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
            doc_id = f"{category}_{content_hash}"
            
            meta = {
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "length": len(content)
            }
            meta.update(metadata)
            
            documents.append(content)
            ids.append(doc_id)
            metadatas.append(meta)
        
        if not documents:
            return
        
        try:
            self._collection.upsert(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
        except Exception as e:
            log.debug(f"Failed to add batch: {e}")
    
    def search_memories(self, query: str, k: int = 5, category: str = None, include_metadata: bool = False) -> List:
        """Search long-term memory semantically

        Args:
            query: Search query
            k: Number of results to return
            category: Optional category filter
            include_metadata: If True, return dicts with content, metadata, and distance
                            If False, return list of strings (legacy behavior)

        Returns:
            List of strings or dicts depending on include_metadata
        """
        if self._collection is None or not query.strip():
            return []

        try:
            where = {"category": category} if category else None

            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                return []

            documents = results["documents"][0]

            if not include_metadata:
                # Legacy behavior - return list of strings
                return documents

            # Return rich results with metadata and distances
            metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)
            distances = results.get("distances", [[]])[0] or [0.0] * len(documents)

            return [
                {
                    "content": doc,
                    "metadata": meta or {},
                    "distance": dist
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]

        except Exception as e:
            log.debug(f"Search failed: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        stats = {
            "current_conversation": len(self._messages),
            "indexed_files": len(self._indexed_files),
        }
        
        try:
            stats["total_facts"] = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            stats["total_messages"] = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        except:
            pass
        
        if self._collection:
            stats["long_term_memories"] = self._collection.count()
        
        return stats
    
    # === Document Indexing ===
    
    def index_document(self, path: Path, category: str = "documents") -> bool:
        """Index a document for RAG"""
        if self._collection is None:
            return False
        
        try:
            path_str = str(path)
            mtime = path.stat().st_mtime
            
            # Check if already indexed with same mtime
            if path_str in self._indexed_files:
                if self._indexed_files[path_str] >= mtime:
                    return True  # Already indexed, skip
            
            content = path.read_text(errors='ignore')
            
            # Skip empty or very small files
            if len(content.strip()) < 50:
                return False
            
            # Split into chunks
            chunks = self._chunk_text(content, chunk_size=500, overlap=50)
            
            if not chunks:
                return False
            
            # Batch all chunks together
            items = []
            for i, chunk in enumerate(chunks):
                items.append({
                    "content": chunk,
                    "category": category,
                    "metadata": {
                        "source": path_str,
                        "filename": path.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
            
            self.add_memories_batch(items)
            
            # Update cache
            self._save_index_cache(path_str, mtime)
            
            return True
            
        except Exception as e:
            log.debug(f"Failed to index {path}: {e}")
            return False
    
    def index_project(
        self, 
        path: Path, 
        extensions: set = None, 
        max_workers: int = 16
    ) -> int:
        """Index all matching files in a project directory (parallelized)"""
        if self._collection is None:
            log.warning("ChromaDB not available, skipping indexing")
            return 0
        
        if extensions is None:
            extensions = {
                '.py', '.md', '.txt', '.json', '.yaml', '.yml',
                '.ino', '.cpp', '.c', '.h', '.hpp',
                '.js', '.ts', '.jsx', '.tsx',
                '.sh', '.bash', '.zsh',
                '.toml', '.cfg', '.ini',
            }
        
        skip_dirs = {
            'node_modules', '__pycache__', 'venv', '.venv', 'env',
            '.git', '.svn', 'build', 'dist', 'target',
            '.idea', '.vscode', 'coverage', '.pytest_cache'
        }
        
        # Collect files to index
        files_to_index = []
        skipped_cached = 0
        
        for file in path.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in extensions:
                continue
            
            # Skip hidden and ignored directories
            if any(part.startswith('.') for part in file.parts):
                continue
            if any(skip in file.parts for skip in skip_dirs):
                continue
            
            # Check cache
            path_str = str(file)
            try:
                mtime = file.stat().st_mtime
                if path_str in self._indexed_files:
                    if self._indexed_files[path_str] >= mtime:
                        skipped_cached += 1
                        continue
            except:
                continue
            
            files_to_index.append(file)
        
        if not files_to_index:
            if skipped_cached > 0:
                log.info(f"All {skipped_cached} files already indexed")
            return skipped_cached
        
        log.info(f"Indexing {len(files_to_index)} files ({skipped_cached} cached)...")
        
        indexed = 0
        failed = 0
        
        # Process in parallel
        def index_single(file: Path) -> bool:
            try:
                return self.index_document(file, category="project")
            except Exception:
                return False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(index_single, f): f for f in files_to_index}
            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        indexed += 1
                    else:
                        failed += 1
                except:
                    failed += 1
        
        total = indexed + skipped_cached
        msg = f"  → Indexed {indexed} new files from {path.name}"
        if skipped_cached:
            msg += f" ({skipped_cached} cached)"
        if failed:
            msg += f" ({failed} failed)"
        print(msg)
        log.info(msg)
        
        return total
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 50:  # Skip tiny chunks
                chunks.append(chunk)
        
        return chunks
    
    def clear_index_cache(self):
        """Clear the index cache to force re-indexing"""
        try:
            self.conn.execute("DELETE FROM indexed_files")
            self.conn.commit()
            self._indexed_files.clear()
            log.info("Index cache cleared")
        except Exception as e:
            log.warning(f"Failed to clear cache: {e}")
    
    # === Project Context ===
    
    def set_active_project(self, name: str, path: str = None, description: str = None) -> str:
        """Set the active project for this session"""
        try:
            # Create or update project
            self.conn.execute("""
                INSERT INTO projects (name, path, description, last_active)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                    path = COALESCE(?, path),
                    description = COALESCE(?, description),
                    last_active = CURRENT_TIMESTAMP
            """, (name, path, description, path, description))
            
            # Set as active
            self.conn.execute("""
                INSERT OR REPLACE INTO active_state (key, value, updated_at)
                VALUES ('active_project', ?, CURRENT_TIMESTAMP)
            """, (name,))
            
            # Start a new project session
            project = self.get_project(name)
            if project:
                self.conn.execute("""
                    INSERT INTO project_sessions (project_id, started_at)
                    VALUES (?, CURRENT_TIMESTAMP)
                """, (project['id'],))
            
            self.conn.commit()
            log.info(f"Active project set: {name}")
            return f"Now working on: {name}"
        except Exception as e:
            log.error(f"Failed to set active project: {e}")
            return f"Error setting project: {e}"
    
    def get_active_project(self) -> Optional[Dict]:
        """Get the currently active project"""
        try:
            row = self.conn.execute("""
                SELECT value FROM active_state WHERE key = 'active_project'
            """).fetchone()
            
            if row:
                return self.get_project(row['value'])
            return None
        except Exception as e:
            log.error(f"Failed to get active project: {e}")
            return None
    
    def get_project(self, name: str) -> Optional[Dict]:
        """Get project details by name"""
        try:
            row = self.conn.execute("""
                SELECT * FROM projects WHERE name = ?
            """, (name,)).fetchone()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            log.error(f"Failed to get project: {e}")
            return None
    
    def list_projects(self) -> List[Dict]:
        """List all known projects"""
        try:
            rows = self.conn.execute("""
                SELECT name, path, description, last_active 
                FROM projects 
                ORDER BY last_active DESC
            """).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.error(f"Failed to list projects: {e}")
            return []
    
    def add_project_note(self, content: str, category: str = "general") -> str:
        """Add a note to the active project"""
        project = self.get_active_project()
        if not project:
            # No active project, just add to general memory
            return self.set_fact(f"note_{datetime.now().isoformat()}", content, category)
        
        try:
            self.conn.execute("""
                INSERT INTO project_notes (project_id, content, category)
                VALUES (?, ?, ?)
            """, (project['id'], content, category))
            self.conn.commit()
            
            # Also add to semantic memory
            self.add_memory(
                content, 
                category=f"project:{project['name']}",
                metadata={"project": project['name'], "type": "note"}
            )
            
            log.info(f"Note added to project {project['name']}: {content[:50]}...")
            return f"Note added to {project['name']}"
        except Exception as e:
            log.error(f"Failed to add project note: {e}")
            return f"Error adding note: {e}"
    
    def get_project_notes(self, project_name: str = None, limit: int = 20) -> List[Dict]:
        """Get notes for a project (or active project if none specified)"""
        if not project_name:
            project = self.get_active_project()
            if not project:
                return []
            project_name = project['name']
        
        project = self.get_project(project_name)
        if not project:
            return []
        
        try:
            rows = self.conn.execute("""
                SELECT content, category, created_at 
                FROM project_notes 
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (project['id'], limit)).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.error(f"Failed to get project notes: {e}")
            return []
    
    def get_project_context(self) -> str:
        """Get formatted context for the active project"""
        project = self.get_active_project()
        if not project:
            return "No active project."
        
        context_parts = [f"Active Project: {project['name']}"]
        
        if project.get('path'):
            context_parts.append(f"Path: {project['path']}")
        
        if project.get('description'):
            context_parts.append(f"Description: {project['description']}")
        
        # Recent notes
        notes = self.get_project_notes(limit=5)
        if notes:
            context_parts.append("\nRecent Notes:")
            for note in notes:
                context_parts.append(f"  - {note['content'][:100]}")
        
        # Last session summary
        try:
            row = self.conn.execute("""
                SELECT summary, ended_at FROM project_sessions 
                WHERE project_id = ? AND summary IS NOT NULL
                ORDER BY ended_at DESC LIMIT 1
            """, (project['id'],)).fetchone()
            
            if row and row['summary']:
                context_parts.append(f"\nLast session: {row['summary']}")
        except:
            pass
        
        return "\n".join(context_parts)
    
    def save_session_summary(self, summary: str) -> str:
        """Save a summary for the current project session"""
        project = self.get_active_project()
        if not project:
            return "No active project to save summary for."
        
        try:
            # Update the most recent session for this project
            self.conn.execute("""
                UPDATE project_sessions 
                SET ended_at = CURRENT_TIMESTAMP, summary = ?
                WHERE project_id = ? AND ended_at IS NULL
            """, (summary, project['id']))
            self.conn.commit()
            
            log.info(f"Session summary saved for {project['name']}")
            return f"Session summary saved for {project['name']}"
        except Exception as e:
            log.error(f"Failed to save session summary: {e}")
            return f"Error saving summary: {e}"

    def get_last_session(self, project_name: str = None) -> Optional[Dict]:
        """Get the last session info for a project"""
        if not project_name:
            project = self.get_active_project()
            if not project:
                return None
            project_name = project['name']

        project = self.get_project(project_name)
        if not project:
            return None

        try:
            row = self.conn.execute("""
                SELECT started_at, ended_at, summary
                FROM project_sessions
                WHERE project_id = ? AND summary IS NOT NULL
                ORDER BY ended_at DESC LIMIT 1
            """, (project['id'],)).fetchone()

            if row:
                return dict(row)
            return None
        except Exception as e:
            log.error(f"Failed to get last session: {e}")
            return None

    # === Phase 3: Context Intelligence Methods ===

    def log_file_edit(self, file_path: str, edit_type: str, project: str = None) -> bool:
        """Log a file edit event (Phase 3)"""
        try:
            from pathlib import Path
            p = Path(file_path)
            file_size = p.stat().st_size if p.exists() else None

            self.conn.execute("""
                INSERT INTO file_edits (file_path, edit_type, project, file_size)
                VALUES (?, ?, ?, ?)
            """, (file_path, edit_type, project, file_size))
            self.conn.commit()
            return True
        except Exception as e:
            log.error(f"Failed to log file edit: {e}")
            return False

    def get_recent_file_edits(self, limit: int = 50, max_age_hours: int = 24) -> List[Dict]:
        """Get recent file edits (Phase 3)"""
        try:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            rows = self.conn.execute("""
                SELECT file_path, timestamp, edit_type, project
                FROM file_edits
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (cutoff.isoformat(), limit)).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.error(f"Failed to get recent file edits: {e}")
            return []

    def save_session_context(self, session_id: str, active_files: List[str],
                            detected_workflow: str = None, project_focus: str = None) -> bool:
        """Save current session context snapshot (Phase 3)"""
        try:
            self.conn.execute("""
                INSERT INTO session_context (session_id, active_files, detected_workflow, project_focus)
                VALUES (?, ?, ?, ?)
            """, (session_id, json.dumps(active_files), detected_workflow, project_focus))
            self.conn.commit()
            return True
        except Exception as e:
            log.error(f"Failed to save session context: {e}")
            return False

    def get_session_context_history(self, session_id: str) -> List[Dict]:
        """Get session context history (Phase 3)"""
        try:
            rows = self.conn.execute("""
                SELECT timestamp, active_files, detected_workflow, project_focus
                FROM session_context
                WHERE session_id = ?
                ORDER BY timestamp DESC
            """, (session_id,)).fetchall()

            results = []
            for row in rows:
                ctx = dict(row)
                # Parse JSON
                if ctx.get('active_files'):
                    try:
                        ctx['active_files'] = json.loads(ctx['active_files'])
                    except:
                        ctx['active_files'] = []
                results.append(ctx)

            return results
        except Exception as e:
            log.error(f"Failed to get session context history: {e}")
            return []

    def start_workflow_session(self, session_id: str, workflow_type: str) -> bool:
        """Start tracking a workflow session (Phase 3)"""
        try:
            self.conn.execute("""
                INSERT INTO workflow_sessions (session_id, workflow_type, outcome)
                VALUES (?, ?, 'ongoing')
            """, (session_id, workflow_type))
            self.conn.commit()
            log.info(f"Started {workflow_type} workflow session: {session_id}")
            return True
        except Exception as e:
            log.error(f"Failed to start workflow session: {e}")
            return False

    def end_workflow_session(self, session_id: str, outcome: str,
                           primary_files: List[str] = None, notes: str = None) -> bool:
        """End a workflow session (Phase 3)"""
        try:
            self.conn.execute("""
                UPDATE workflow_sessions
                SET end_time = CURRENT_TIMESTAMP, outcome = ?, primary_files = ?, notes = ?
                WHERE session_id = ? AND end_time IS NULL
            """, (outcome, json.dumps(primary_files) if primary_files else None, notes, session_id))
            self.conn.commit()
            log.info(f"Ended workflow session: {session_id} ({outcome})")
            return True
        except Exception as e:
            log.error(f"Failed to end workflow session: {e}")
            return False

    def get_workflow_history(self, workflow_type: str = None, limit: int = 10) -> List[Dict]:
        """Get workflow session history (Phase 3)"""
        try:
            if workflow_type:
                rows = self.conn.execute("""
                    SELECT session_id, workflow_type, start_time, end_time, outcome, primary_files, notes
                    FROM workflow_sessions
                    WHERE workflow_type = ?
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (workflow_type, limit)).fetchall()
            else:
                rows = self.conn.execute("""
                    SELECT session_id, workflow_type, start_time, end_time, outcome, primary_files, notes
                    FROM workflow_sessions
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (limit,)).fetchall()

            results = []
            for row in rows:
                wf = dict(row)
                # Parse JSON
                if wf.get('primary_files'):
                    try:
                        wf['primary_files'] = json.loads(wf['primary_files'])
                    except:
                        wf['primary_files'] = []
                results.append(wf)

            return results
        except Exception as e:
            log.error(f"Failed to get workflow history: {e}")
            return []
