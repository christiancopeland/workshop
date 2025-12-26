"""
Workshop Constructs (Phase 3)
Visual elements that can be manipulated through gesture and voice

This is a skeleton for Phase 3 - spatial UI constructs.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import uuid
import math


class ConstructType(Enum):
    """Types of visual constructs"""
    TEXT = "text"              # Text display
    CODE = "code"              # Syntax-highlighted code
    IMAGE = "image"            # Image display
    CHART = "chart"            # Data visualization
    FILE_TREE = "file_tree"    # Directory structure
    WAVEFORM = "waveform"      # Audio/signal waveform
    TERMINAL = "terminal"      # Terminal output
    PANEL = "panel"            # Generic container
    BUTTON = "button"          # Interactive button
    SCHEMATIC = "schematic"    # Circuit diagram


@dataclass
class Vector3:
    """3D vector for position, rotation, scale"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)
    
    def distance_to(self, other: "Vector3") -> float:
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )


@dataclass
class Construct:
    """A visual construct in the spatial UI"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ConstructType = ConstructType.PANEL
    
    # Transform
    position: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    
    # Appearance
    opacity: float = 1.0
    color: str = "#00ff88"  # Holographic green
    glow: float = 0.5
    
    # Content
    title: str = ""
    content: Any = None
    
    # Behavior
    visible: bool = True
    interactive: bool = True
    pinned: bool = False  # If pinned, doesn't auto-hide
    
    # Metadata
    created_at: float = 0.0
    last_interaction: float = 0.0
    
    def to_dict(self) -> dict:
        """Serialize construct to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position.to_tuple(),
            "rotation": self.rotation.to_tuple(),
            "scale": self.scale.to_tuple(),
            "opacity": self.opacity,
            "color": self.color,
            "glow": self.glow,
            "title": self.title,
            "content": self.content,
            "visible": self.visible,
            "interactive": self.interactive,
            "pinned": self.pinned,
        }


class ConstructManager:
    """
    Manages visual constructs in the spatial UI.
    
    This would integrate with a renderer (Three.js, Godot, etc.)
    to display the constructs on screen.
    """
    
    def __init__(self):
        self._constructs: Dict[str, Construct] = {}
        self._focused: Optional[str] = None
        self._render_callback: Optional[Callable] = None
    
    def create(
        self,
        type: ConstructType,
        title: str = "",
        content: Any = None,
        position: tuple = (0, 0, 0),
        **kwargs
    ) -> Construct:
        """Create a new construct"""
        import time
        
        construct = Construct(
            type=type,
            title=title,
            content=content,
            position=Vector3(*position),
            created_at=time.time(),
            last_interaction=time.time(),
            **kwargs
        )
        
        self._constructs[construct.id] = construct
        self._request_render()
        
        return construct
    
    def get(self, id: str) -> Optional[Construct]:
        """Get a construct by ID"""
        return self._constructs.get(id)
    
    def remove(self, id: str) -> bool:
        """Remove a construct"""
        if id in self._constructs:
            del self._constructs[id]
            if self._focused == id:
                self._focused = None
            self._request_render()
            return True
        return False
    
    def clear(self):
        """Remove all constructs"""
        self._constructs.clear()
        self._focused = None
        self._request_render()
    
    def list_all(self) -> List[Construct]:
        """Get all constructs"""
        return list(self._constructs.values())
    
    def list_visible(self) -> List[Construct]:
        """Get all visible constructs"""
        return [c for c in self._constructs.values() if c.visible]
    
    # === Transform Operations ===
    
    def move(self, id: str, position: tuple):
        """Move a construct to a new position"""
        if construct := self.get(id):
            construct.position = Vector3(*position)
            self._update_interaction(construct)
            self._request_render()
    
    def rotate(self, id: str, rotation: tuple):
        """Rotate a construct"""
        if construct := self.get(id):
            construct.rotation = Vector3(*rotation)
            self._update_interaction(construct)
            self._request_render()
    
    def resize(self, id: str, scale: tuple):
        """Resize a construct"""
        if construct := self.get(id):
            construct.scale = Vector3(*scale)
            self._update_interaction(construct)
            self._request_render()
    
    # === Appearance Operations ===
    
    def set_opacity(self, id: str, opacity: float):
        """Set construct opacity"""
        if construct := self.get(id):
            construct.opacity = max(0, min(1, opacity))
            self._request_render()
    
    def fade_in(self, id: str, duration: float = 0.3):
        """Fade construct in"""
        # TODO: Implement animation
        if construct := self.get(id):
            construct.opacity = 1.0
            construct.visible = True
            self._request_render()
    
    def fade_out(self, id: str, duration: float = 0.3, remove: bool = False):
        """Fade construct out"""
        # TODO: Implement animation
        if construct := self.get(id):
            construct.opacity = 0.0
            construct.visible = False
            if remove:
                self.remove(id)
            else:
                self._request_render()
    
    # === Focus Operations ===
    
    def focus(self, id: str):
        """Focus on a construct"""
        if id in self._constructs:
            self._focused = id
            self._update_interaction(self._constructs[id])
            self._request_render()
    
    def unfocus(self):
        """Clear focus"""
        self._focused = None
        self._request_render()
    
    @property
    def focused(self) -> Optional[Construct]:
        """Get the currently focused construct"""
        if self._focused:
            return self._constructs.get(self._focused)
        return None
    
    # === Content Operations ===
    
    def update_content(self, id: str, content: Any):
        """Update construct content"""
        if construct := self.get(id):
            construct.content = content
            self._update_interaction(construct)
            self._request_render()
    
    def append_content(self, id: str, content: str):
        """Append to construct content (for text/terminal)"""
        if construct := self.get(id):
            if isinstance(construct.content, str):
                construct.content += content
            elif construct.content is None:
                construct.content = content
            self._update_interaction(construct)
            self._request_render()
    
    # === Spatial Queries ===
    
    def find_at_position(self, position: tuple, radius: float = 0.1) -> List[Construct]:
        """Find constructs near a position"""
        pos = Vector3(*position)
        return [
            c for c in self._constructs.values()
            if c.visible and c.position.distance_to(pos) <= radius
        ]
    
    def find_in_front(self) -> List[Construct]:
        """Find constructs in the forward viewing area"""
        # TODO: Implement based on camera/user position
        return [c for c in self._constructs.values() if c.visible]
    
    # === Layout Helpers ===
    
    def arrange_grid(
        self, 
        ids: List[str], 
        center: tuple = (0, 0, 0),
        spacing: float = 0.3,
        columns: int = 3
    ):
        """Arrange constructs in a grid"""
        cx, cy, cz = center
        
        for i, id in enumerate(ids):
            row = i // columns
            col = i % columns
            
            x = cx + (col - columns/2) * spacing
            y = cy - row * spacing
            
            self.move(id, (x, y, cz))
    
    def arrange_circle(
        self,
        ids: List[str],
        center: tuple = (0, 0, 0),
        radius: float = 0.5
    ):
        """Arrange constructs in a circle"""
        cx, cy, cz = center
        n = len(ids)
        
        for i, id in enumerate(ids):
            angle = (2 * math.pi * i) / n
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            
            self.move(id, (x, y, cz))
    
    # === Internal ===
    
    def _update_interaction(self, construct: Construct):
        """Update last interaction time"""
        import time
        construct.last_interaction = time.time()
    
    def _request_render(self):
        """Request a render update"""
        if self._render_callback:
            self._render_callback(self)
    
    def on_render(self, callback: Callable):
        """Set the render callback"""
        self._render_callback = callback
    
    def get_render_data(self) -> List[dict]:
        """Get all visible constructs as render-ready data"""
        return [
            c.to_dict() 
            for c in self._constructs.values() 
            if c.visible
        ]


# === Construct Factory Functions ===

def create_text_panel(
    manager: ConstructManager,
    text: str,
    title: str = "",
    position: tuple = (0, 0, 0)
) -> Construct:
    """Create a text display panel"""
    return manager.create(
        type=ConstructType.TEXT,
        title=title,
        content=text,
        position=position,
        color="#00ff88"
    )


def create_code_view(
    manager: ConstructManager,
    code: str,
    language: str = "python",
    title: str = "Code",
    position: tuple = (0, 0, 0)
) -> Construct:
    """Create a syntax-highlighted code view"""
    return manager.create(
        type=ConstructType.CODE,
        title=title,
        content={"code": code, "language": language},
        position=position,
        color="#00aaff"
    )


def create_file_tree(
    manager: ConstructManager,
    path: str,
    title: str = "Files",
    position: tuple = (0, 0, 0)
) -> Construct:
    """Create a file tree display"""
    from pathlib import Path
    
    def get_tree(p: Path, depth: int = 0, max_depth: int = 3) -> list:
        if depth > max_depth:
            return []
        
        items = []
        try:
            for item in sorted(p.iterdir()):
                if item.name.startswith('.'):
                    continue
                
                entry = {
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "path": str(item)
                }
                
                if item.is_dir() and depth < max_depth:
                    entry["children"] = get_tree(item, depth + 1, max_depth)
                
                items.append(entry)
        except PermissionError:
            pass
        
        return items
    
    tree = get_tree(Path(path).expanduser())
    
    return manager.create(
        type=ConstructType.FILE_TREE,
        title=title,
        content={"root": path, "tree": tree},
        position=position,
        color="#ffaa00"
    )


def create_terminal(
    manager: ConstructManager,
    title: str = "Terminal",
    position: tuple = (0, 0, 0)
) -> Construct:
    """Create a terminal output display"""
    return manager.create(
        type=ConstructType.TERMINAL,
        title=title,
        content="",
        position=position,
        color="#00ff00",
        glow=0.3
    )


def create_chart(
    manager: ConstructManager,
    data: dict,
    chart_type: str = "line",
    title: str = "Chart",
    position: tuple = (0, 0, 0)
) -> Construct:
    """Create a data visualization chart"""
    return manager.create(
        type=ConstructType.CHART,
        title=title,
        content={"data": data, "chart_type": chart_type},
        position=position,
        color="#ff00aa"
    )


# === Example Integration ===

async def example():
    """Example of construct system"""
    
    manager = ConstructManager()
    
    # Create some constructs
    panel = create_text_panel(
        manager,
        text="Welcome to Workshop\n\nSay 'show files' to see your project.",
        title="Info",
        position=(-0.4, 0.3, 0)
    )
    
    code = create_code_view(
        manager,
        code="def hello():\n    print('Hello, Workshop!')",
        language="python",
        title="example.py",
        position=(0.4, 0.3, 0)
    )
    
    terminal = create_terminal(
        manager,
        title="Output",
        position=(0, -0.2, 0)
    )
    
    # Update terminal content
    manager.append_content(terminal.id, "$ python example.py\n")
    manager.append_content(terminal.id, "Hello, Workshop!\n")
    manager.append_content(terminal.id, "$ _")
    
    # Focus on code view
    manager.focus(code.id)
    
    # Get render data (would be sent to Three.js/Godot)
    render_data = manager.get_render_data()
    print(f"Ready to render {len(render_data)} constructs:")
    for c in render_data:
        print(f"  - {c['title']} ({c['type']}) at {c['position']}")
    
    return manager


if __name__ == "__main__":
    import asyncio
    asyncio.run(example())
