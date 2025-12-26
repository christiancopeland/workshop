"""
Workshop Context Tools - Phase 3
LLM-callable tools for retrieving additional context on-demand
"""

from pathlib import Path
from typing import Optional, List
import re

from logger import get_logger

log = get_logger("context_tools")


def register_context_tools(registry: 'ToolRegistry', context_manager: 'ContextManager', memory: 'MemorySystem'):
    """Register Phase 3 context retrieval tools"""

    # === Tool 1: get_file_content ===

    def get_file_content(path: str, lines: str = None) -> str:
        """
        Read file or specific line range.

        Args:
            path: File path (relative or absolute)
            lines: Optional line range like "10-25" or "42"

        Examples:
            get_file_content("battery_guardian.ino")
            get_file_content("config.h", lines="10-25")
        """
        try:
            # Resolve path
            p = Path(path).expanduser()

            # If relative, try to find in monitored projects
            if not p.is_absolute():
                for project_path in context_manager.monitored_projects:
                    candidate = project_path / path
                    if candidate.exists():
                        p = candidate
                        break

            if not p.exists():
                return f"Error: File not found: {path}"

            if not p.is_file():
                return f"Error: Not a file: {path}"

            content = p.read_text(errors='ignore')
            all_lines = content.split('\n')

            # Parse line range if specified
            if lines:
                try:
                    if '-' in lines:
                        start, end = map(int, lines.split('-'))
                        selected_lines = all_lines[start-1:end]  # 1-indexed
                    else:
                        line_num = int(lines)
                        selected_lines = [all_lines[line_num-1]]  # 1-indexed

                    result = f"File: {p} (lines {lines})\n"
                    result += "-" * 40 + "\n"
                    result += '\n'.join(selected_lines)
                    return result

                except (ValueError, IndexError) as e:
                    return f"Error: Invalid line range '{lines}': {e}"

            # Return full file
            result = f"File: {p}\n"
            result += f"Size: {len(content):,} bytes, {len(all_lines)} lines\n"
            result += "-" * 40 + "\n"
            result += content
            return result

        except Exception as e:
            log.error(f"get_file_content error: {e}")
            return f"Error reading file: {e}"

    registry.register(
        name="get_file_content",
        func=get_file_content,
        description="Read file contents or specific line range. Use when you need to see code/config details.",
        signature="get_file_content(path: str, lines: str = None)"
    )

    # === Tool 2: search_project_files ===

    def search_project_files(query: str, project: str = None) -> str:
        """
        Semantic search across project files using ChromaDB.

        Args:
            query: Search query (e.g., "BQ76940 initialization")
            project: Optional project name to limit search

        Returns matches with file paths and relevance.
        """
        try:
            # Use memory's semantic search
            if project:
                results = memory.search_memories(query, k=10, category=f"project:{project}")
            else:
                results = memory.search_memories(query, k=10, category="project")

            if not results:
                return f"No results found for '{query}'"

            output = [f"Search results for '{query}':"]
            for i, result in enumerate(results, 1):
                # Extract file info from metadata if available
                output.append(f"\n{i}. {result[:200]}...")

            return "\n".join(output)

        except Exception as e:
            log.error(f"search_project_files error: {e}")
            return f"Error searching: {e}"

    registry.register(
        name="search_project_files",
        func=search_project_files,
        description="Semantic search across project files. Use to find code/docs related to a concept.",
        signature="search_project_files(query: str, project: str = None)"
    )

    # === Tool 3: find_definition ===

    def find_definition(symbol: str, project: str = None) -> str:
        """
        Find where a variable/function/class is defined.

        Args:
            symbol: Symbol name (e.g., "CELL_VOLTAGE_SCALE", "setup()")
            project: Optional project name to limit search

        Searches for common definition patterns across languages.
        """
        try:
            # Build regex patterns for different languages
            patterns = [
                # C/C++/Arduino
                rf'#define\s+{re.escape(symbol)}\b',  # #define SYMBOL
                rf'\b{re.escape(symbol)}\s*\(',       # function_name(
                rf'(int|float|double|char|void|bool|uint\w*)\s+{re.escape(symbol)}\b',  # type symbol

                # Python
                rf'def\s+{re.escape(symbol)}\s*\(',   # def func(
                rf'class\s+{re.escape(symbol)}\b',    # class Name
                rf'{re.escape(symbol)}\s*=',           # variable =

                # JavaScript/TypeScript
                rf'(const|let|var)\s+{re.escape(symbol)}\b',  # const/let/var name
                rf'function\s+{re.escape(symbol)}\s*\(',      # function name(
            ]

            results = []
            search_paths = context_manager.monitored_projects

            if project:
                # Filter to specific project
                search_paths = [p for p in search_paths if project.lower() in p.name.lower()]

            for project_path in search_paths:
                for file_path in project_path.rglob("*"):
                    if not file_path.is_file():
                        continue

                    if file_path.suffix not in context_manager.indexable_extensions:
                        continue

                    try:
                        content = file_path.read_text(errors='ignore')

                        for pattern in patterns:
                            matches = list(re.finditer(pattern, content, re.MULTILINE))
                            for match in matches:
                                # Get line number
                                line_num = content[:match.start()].count('\n') + 1

                                # Get the line content
                                lines = content.split('\n')
                                line_content = lines[line_num - 1].strip()

                                results.append({
                                    'file': file_path,
                                    'line': line_num,
                                    'content': line_content
                                })

                    except:
                        continue

            if not results:
                return f"Definition not found for '{symbol}'"

            # Format results
            output = [f"Found {len(results)} definition(s) for '{symbol}':\n"]
            for r in results[:10]:  # Limit to 10 results
                output.append(f"  {r['file'].name}:{r['line']}")
                output.append(f"    {r['content']}")

            return "\n".join(output)

        except Exception as e:
            log.error(f"find_definition error: {e}")
            return f"Error finding definition: {e}"

    registry.register(
        name="find_definition",
        func=find_definition,
        description="Find where a symbol (variable/function/class) is defined. Use when you need to locate code.",
        signature="find_definition(symbol: str, project: str = None)"
    )

    # === Tool 4: find_references ===

    def find_references(symbol: str, project: str = None) -> str:
        """
        Find all uses of a variable/function/class.

        Args:
            symbol: Symbol name
            project: Optional project to limit search

        Returns all locations where the symbol is used.
        """
        try:
            results = []
            search_paths = context_manager.monitored_projects

            if project:
                search_paths = [p for p in search_paths if project.lower() in p.name.lower()]

            # Simple word boundary search
            pattern = rf'\b{re.escape(symbol)}\b'

            for project_path in search_paths:
                for file_path in project_path.rglob("*"):
                    if not file_path.is_file():
                        continue

                    if file_path.suffix not in context_manager.indexable_extensions:
                        continue

                    try:
                        content = file_path.read_text(errors='ignore')
                        matches = list(re.finditer(pattern, content))

                        if matches:
                            lines = content.split('\n')
                            for match in matches[:5]:  # Max 5 per file
                                line_num = content[:match.start()].count('\n') + 1
                                line_content = lines[line_num - 1].strip()

                                results.append({
                                    'file': file_path,
                                    'line': line_num,
                                    'content': line_content
                                })

                    except:
                        continue

            if not results:
                return f"No references found for '{symbol}'"

            # Format results
            output = [f"Found {len(results)} reference(s) to '{symbol}':\n"]
            for r in results[:20]:  # Limit to 20
                output.append(f"  {r['file'].name}:{r['line']} - {r['content'][:80]}")

            if len(results) > 20:
                output.append(f"\n... and {len(results) - 20} more")

            return "\n".join(output)

        except Exception as e:
            log.error(f"find_references error: {e}")
            return f"Error finding references: {e}"

    registry.register(
        name="find_references",
        func=find_references,
        description="Find all uses of a symbol. Use to understand where code is called/used.",
        signature="find_references(symbol: str, project: str = None)"
    )

    # === Tool 5: get_related_files ===

    def get_related_files(file_path: str) -> str:
        """
        Get files related to the given file.

        Returns:
            - Files this imports
            - Files that import this
            - Files co-edited in recent sessions

        Args:
            file_path: Path to file (can be relative)
        """
        try:
            # Resolve path
            p = Path(file_path).expanduser()

            if not p.is_absolute():
                for project_path in context_manager.monitored_projects:
                    candidate = project_path / file_path
                    if candidate.exists():
                        p = candidate
                        break

            if not p.exists():
                return f"Error: File not found: {file_path}"

            # Get related files from context graph
            related = context_manager.context_graph.get_related_files(p, max_results=10)

            if not related:
                return f"No related files found for {p.name}"

            # Format results
            output = [f"Files related to {p.name}:\n"]

            # Get node info if available
            path_str = str(p)
            node = context_manager.context_graph.nodes.get(path_str)

            if node:
                if node.imports:
                    output.append("Imports:")
                    for imp in list(node.imports)[:5]:
                        output.append(f"  • {imp}")

                if node.imported_by:
                    output.append("\nImported by:")
                    for imp in list(node.imported_by)[:5]:
                        output.append(f"  • {Path(imp).name}")

            output.append("\nRelated files (by usage/editing):")
            for rel_path, score in related:
                output.append(f"  • {rel_path.name} (relevance: {score:.2f})")

            return "\n".join(output)

        except Exception as e:
            log.error(f"get_related_files error: {e}")
            return f"Error getting related files: {e}"

    registry.register(
        name="get_related_files",
        func=get_related_files,
        description="Get files related to a given file (imports, dependencies, co-edited files).",
        signature="get_related_files(file_path: str)"
    )

    # === Tool 6: get_recent_edits ===

    def get_recent_edits(path: str = None, limit: int = 10) -> str:
        """
        Get recent edit history.

        Args:
            path: Optional file path to filter edits
            limit: Max number of edits to return

        If path is None, returns all recent edits across projects.
        """
        try:
            if path:
                # Get edits for specific file from memory
                all_edits = memory.get_recent_file_edits(limit=50)
                edits = [e for e in all_edits if path in e['file_path']][:limit]

                if not edits:
                    return f"No recent edits found for {path}"

                output = [f"Recent edits to {Path(path).name}:\n"]
                for e in edits:
                    output.append(f"  • {e['edit_type']} - {e['timestamp']}")

                return "\n".join(output)

            else:
                # Get all recent edits from context graph
                recent_edits = context_manager.context_graph.get_recent_edits(limit=limit)

                if not recent_edits:
                    return "No recent file edits"

                output = ["Recent file edits:\n"]
                for edit in recent_edits:
                    output.append(f"  • {edit.edit_type}: {edit.path.name} ({int(edit.age_seconds)}s ago)")

                return "\n".join(output)

        except Exception as e:
            log.error(f"get_recent_edits error: {e}")
            return f"Error getting recent edits: {e}"

    registry.register(
        name="get_recent_edits",
        func=get_recent_edits,
        description="Get recent file edit history. Use to understand what's been changed recently.",
        signature="get_recent_edits(path: str = None, limit: int = 10)"
    )

    # === Tool 7: search_web_docs ===

    async def search_web_docs(query: str) -> str:
        """
        Fetch external documentation or datasheets.

        Args:
            query: Search query (e.g., "BQ76940 ADC calibration datasheet")

        This is a wrapper around web_search focused on technical documentation.
        """
        try:
            # Try the existing web_search tool if available
            if 'web_search' in registry._tools:
                web_search = registry._tools['web_search']['func']
                result = await web_search(query, max_results=5)
                return result
            else:
                return "Error: Web search not available. Install ddgs: pip install ddgs"

        except Exception as e:
            log.error(f"search_web_docs error: {e}")
            return f"Error searching web: {e}"

    registry.register(
        name="search_web_docs",
        func=search_web_docs,
        description="Search for external documentation, datasheets, or technical references.",
        signature="search_web_docs(query: str)"
    )

    # === Bonus Tool: get_context_stats ===

    def get_context_stats() -> str:
        """Get statistics about the context system"""
        try:
            stats = context_manager.get_stats()

            output = ["Context System Statistics:\n"]
            output.append(f"  Monitored projects: {stats['monitored_projects']}")
            output.append(f"  Watched directories: {stats['watched_paths']}")
            output.append(f"  Indexed files: {stats['indexed_files']}")
            output.append(f"  Recent edits tracked: {stats['recent_edits']}")
            output.append(f"  Active files (last 5min): {stats['active_files']}")

            # Get active project from memory
            active_project = memory.get_active_project()
            if active_project:
                output.append(f"\n  Active project: {active_project['name']}")

            return "\n".join(output)

        except Exception as e:
            log.error(f"get_context_stats error: {e}")
            return f"Error getting stats: {e}"

    registry.register(
        name="get_context_stats",
        func=get_context_stats,
        description="Get statistics about the context tracking system",
        signature="get_context_stats()"
    )

    log.info(f"Registered 8 Phase 3 context tools")
