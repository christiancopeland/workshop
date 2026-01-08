"""
Workshop Construct Tools
Tools for creating and managing visual constructs
"""

import os
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger("construct_tools")


def register_construct_tools(registry, construct_manager):
    """Register construct-related tools"""
    
    async def show_file(path: str, language: Optional[str] = None) -> str:
        """Show a file in a code panel construct"""
        from tools import resolve_path_case_insensitive, find_file
        
        # Resolve the path
        resolved = resolve_path_case_insensitive(path)
        
        # If not found and looks like just a filename, search for it
        if (not resolved or not resolved.exists()) and '/' not in path and '~' not in path:
            log.debug(f"File not found directly, searching for: {path}")
            found = find_file(path)
            if found:
                log.info(f"Found file at: {found}")
                resolved = found
        
        if not resolved or not resolved.exists():
            return f"File not found: {path}"
        
        try:
            with open(resolved, 'r') as f:
                content = f.read()
            
            # Detect language from extension
            if not language:
                ext = resolved.suffix.lstrip('.')
                language = ext if ext else 'text'
            
            title = resolved.name
            
            # Create the construct
            construct_id = await construct_manager.create_code_panel(
                content=content,
                title=title,
                language=language
            )
            
            log.info(f"Created code panel for: {title}")
            return f"Showing {title} in code panel"
            
        except Exception as e:
            log.error(f"Error showing file: {e}")
            return f"Error reading file: {e}"
    
    registry.register(
        name="show_file",
        func=show_file,
        description="Display a file in a visual code panel",
        signature="show_file(path: str, language: str = None)"
    )
    
    async def show_directory(path: str) -> str:
        """Show a directory tree in a visual construct"""
        from tools import resolve_path_case_insensitive
        
        resolved = resolve_path_case_insensitive(path)
        if not resolved:
            return f"Directory not found: {path}"
        
        if not resolved.is_dir():
            return f"Not a directory: {path}"
        
        def build_tree(dir_path: Path, depth: int = 0, max_depth: int = 3) -> dict:
            """Build a nested dict representing the directory tree"""
            tree = {}
            
            if depth >= max_depth:
                return tree
            
            try:
                items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                
                for item in items:
                    # Skip hidden files and common ignored directories
                    if item.name.startswith('.') or item.name in ('node_modules', '__pycache__', 'venv', 'build'):
                        continue
                    
                    if item.is_dir():
                        tree[item.name] = {
                            'type': 'directory',
                            'path': str(item),
                            'children': build_tree(item, depth + 1, max_depth)
                        }
                    else:
                        tree[item.name] = {
                            'type': 'file',
                            'path': str(item),
                            'size': item.stat().st_size
                        }
            except PermissionError:
                pass
            
            return tree
        
        tree = build_tree(resolved)
        title = resolved.name or str(resolved)
        
        construct_id = await construct_manager.create_file_tree(
            tree=tree,
            title=title,
            root_path=str(resolved)
        )
        
        log.info(f"Created file tree for: {title}")
        return f"Showing {title} directory"
    
    registry.register(
        name="show_directory",
        func=show_directory,
        description="Display a directory tree in a visual panel",
        signature="show_directory(path: str)"
    )
    
    async def create_note(content: str, title: str = "Note") -> str:
        """Create a floating note card"""
        construct_id = await construct_manager.create_note(
            content=content,
            title=title
        )
        log.info(f"Created note: {title}")
        return f"Created note: {title}"
    
    registry.register(
        name="create_note",
        func=create_note,
        description="Create a floating note card",
        signature="create_note(content: str, title: str = 'Note')"
    )
    
    async def show_chart(data: list, title: str = "Chart", chart_type: str = "line") -> str:
        """Show data as a visual chart"""
        construct_id = await construct_manager.create_chart(
            data=data,
            title=title,
            chart_type=chart_type
        )
        log.info(f"Created chart: {title}")
        return f"Created {chart_type} chart: {title}"
    
    registry.register(
        name="show_chart",
        func=show_chart,
        description="Display data as a line or bar chart",
        signature="show_chart(data: list, title: str = 'Chart', chart_type: str = 'line')"
    )
    
    async def show_terminal(content: str = "", title: str = "Terminal") -> str:
        """Create a terminal output display"""
        construct_id = await construct_manager.create_terminal(
            content=content,
            title=title
        )
        log.info(f"Created terminal: {title}")
        return f"Created terminal: {title}"
    
    registry.register(
        name="show_terminal",
        func=show_terminal,
        description="Create a terminal-style output display",
        signature="show_terminal(content: str = '', title: str = 'Terminal')"
    )
    
    async def show_markdown(content: str, title: str = "Document") -> str:
        """Display markdown content in a panel"""
        construct_id = await construct_manager.create_markdown(
            content=content,
            title=title
        )
        log.info(f"Created markdown panel: {title}")
        return f"Showing document: {title}"
    
    registry.register(
        name="show_markdown",
        func=show_markdown,
        description="Display markdown content in a visual panel",
        signature="show_markdown(content: str, title: str = 'Document')"
    )
    
    async def close_construct(name: str) -> str:
        """Close a visual construct by name or ID"""
        # Try exact ID match first
        if name in construct_manager.constructs:
            await construct_manager.close_construct(name)
            return f"Closed construct: {name}"
        
        # Try title match
        construct = construct_manager.find_by_title(name)
        if construct:
            await construct_manager.close_construct(construct.id)
            return f"Closed: {construct.title}"
        
        return f"No construct found matching: {name}"
    
    registry.register(
        name="close_construct",
        func=close_construct,
        description="Close a visual construct by name",
        signature="close_construct(name: str)"
    )
    
    async def close_all_constructs() -> str:
        """Close all visual constructs"""
        count = len(construct_manager.constructs)
        await construct_manager.close_all()
        return f"Closed {count} construct(s)"
    
    registry.register(
        name="close_all_constructs",
        func=close_all_constructs,
        description="Close all visual constructs",
        signature="close_all_constructs()"
    )
    
    async def list_constructs() -> str:
        """List all active constructs"""
        constructs = construct_manager.list_constructs()
        if not constructs:
            return "No active constructs"
        
        lines = ["Active constructs:"]
        for c in constructs:
            lines.append(f"  • {c['title']} ({c['type']})")
        
        return "\n".join(lines)
    
    registry.register(
        name="list_constructs",
        func=list_constructs,
        description="List all active visual constructs",
        signature="list_constructs()"
    )
    
    async def move_construct(name: str, x: int, y: int) -> str:
        """Move a construct to a new position"""
        construct = construct_manager.find_by_title(name)
        if not construct:
            if name in construct_manager.constructs:
                construct = construct_manager.constructs[name]
            else:
                return f"No construct found matching: {name}"
        
        await construct_manager.move_construct(construct.id, x, y)
        return f"Moved {construct.title} to ({x}, {y})"
    
    registry.register(
        name="move_construct",
        func=move_construct,
        description="Move a construct to a new screen position",
        signature="move_construct(name: str, x: int, y: int)"
    )
    
    log.info("Registered 9 construct tools")
