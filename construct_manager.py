"""
Workshop Construct Manager
Manages visual constructs and communicates with the display UI via WebSocket
"""

import asyncio
import json
import uuid
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol

from logger import get_logger

log = get_logger("constructs")


class ConstructType(str, Enum):
    CODE_PANEL = "code_panel"
    FILE_TREE = "file_tree"
    DATA_CHART = "data_chart"
    NOTE_CARD = "note_card"
    TERMINAL = "terminal"
    IMAGE_VIEW = "image_view"
    MARKDOWN = "markdown"


@dataclass
class Position:
    x: int = 100
    y: int = 100
    width: int = 400
    height: int = 300


@dataclass
class Construct:
    id: str
    type: ConstructType
    title: str
    content: Any
    position: Position = field(default_factory=Position)
    visible: bool = True
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "content": self.content,
            "position": asdict(self.position),
            "visible": self.visible,
            "metadata": self.metadata
        }


class ConstructManager:
    """Manages visual constructs and WebSocket communication with UI"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.constructs: Dict[str, Construct] = {}
        self.clients: set[WebSocketServerProtocol] = set()
        self._server = None
        self._server_task = None
        self._position_counter = 0
        self._first_client_connected = False
        
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self._server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port
            )
            log.info(f"Construct server started on ws://{self.host}:{self.port}")
            print(f"🖼️  Construct server: ws://{self.host}:{self.port}")
            return True
        except Exception as e:
            log.error(f"Failed to start construct server: {e}")
            print(f"⚠️  Construct server failed: {e}")
            return False
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            log.info("Construct server stopped")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new client connection"""
        self.clients.add(websocket)
        client_id = id(websocket)
        log.info(f"UI client connected: {client_id}")
        
        # Only print once on first connection
        if not self._first_client_connected:
            self._first_client_connected = True
            print(f"🖥️  Construct UI connected!")
        
        try:
            # Send current state to new client
            await self._send_full_state(websocket)
            
            # Handle incoming messages from UI
            async for message in websocket:
                await self._handle_ui_message(message, websocket)
                
        except websockets.exceptions.ConnectionClosed:
            log.info(f"UI client disconnected: {client_id}")
        finally:
            self.clients.discard(websocket)
    
    async def _send_full_state(self, websocket: WebSocketServerProtocol):
        """Send full construct state to a client"""
        state = {
            "action": "full_state",
            "constructs": [c.to_dict() for c in self.constructs.values()]
        }
        await websocket.send(json.dumps(state))
        log.debug(f"Sent full state: {len(self.constructs)} constructs")
    
    async def _handle_ui_message(self, message: str, websocket: WebSocketServerProtocol):
        """Handle messages from the UI (e.g., user closed a panel)"""
        try:
            data = json.loads(message)
            action = data.get("action")
            
            if action == "close":
                construct_id = data.get("id")
                if construct_id in self.constructs:
                    del self.constructs[construct_id]
                    log.info(f"UI closed construct: {construct_id}")
                    
            elif action == "move":
                construct_id = data.get("id")
                if construct_id in self.constructs:
                    pos = data.get("position", {})
                    self.constructs[construct_id].position.x = pos.get("x", 0)
                    self.constructs[construct_id].position.y = pos.get("y", 0)
                    log.debug(f"UI moved construct: {construct_id}")
                    
            elif action == "resize":
                construct_id = data.get("id")
                if construct_id in self.constructs:
                    size = data.get("size", {})
                    self.constructs[construct_id].position.width = size.get("width", 400)
                    self.constructs[construct_id].position.height = size.get("height", 300)
                    log.debug(f"UI resized construct: {construct_id}")
                    
        except json.JSONDecodeError:
            log.warning(f"Invalid JSON from UI: {message[:100]}")
    
    async def _broadcast(self, message: Dict):
        """Broadcast a message to all connected clients"""
        if not self.clients:
            log.debug("No UI clients connected, message not sent")
            return
            
        payload = json.dumps(message)
        await asyncio.gather(
            *[client.send(payload) for client in self.clients],
            return_exceptions=True
        )
        log.debug(f"Broadcast to {len(self.clients)} clients: {message.get('action')}")
    
    async def broadcast_project_changed(self, project_name: str):
        """Notify UI that the active project has changed"""
        await self._broadcast({
            "action": "project_changed",
            "project": project_name
        })
        log.info(f"Broadcast project change: {project_name}")
    
    def _get_next_position(self) -> Position:
        """Get a cascading position for new constructs"""
        offset = (self._position_counter % 5) * 30
        self._position_counter += 1
        return Position(x=100 + offset, y=100 + offset)
    
    # === Public API for creating constructs ===
    
    async def create_code_panel(
        self,
        content: str,
        title: str = "Code",
        language: str = "python",
        position: Optional[Position] = None
    ) -> str:
        """Create a code panel construct"""
        construct_id = f"code_{uuid.uuid4().hex[:8]}"
        
        construct = Construct(
            id=construct_id,
            type=ConstructType.CODE_PANEL,
            title=title,
            content=content,
            position=position or self._get_next_position(),
            metadata={"language": language}
        )
        
        self.constructs[construct_id] = construct
        
        await self._broadcast({
            "action": "create",
            **construct.to_dict()
        })
        
        log.info(f"Created code panel: {title} ({construct_id})")
        return construct_id
    
    async def create_file_tree(
        self,
        tree: Dict,
        title: str = "Files",
        root_path: str = "",
        position: Optional[Position] = None
    ) -> str:
        """Create a file tree construct"""
        construct_id = f"tree_{uuid.uuid4().hex[:8]}"
        
        construct = Construct(
            id=construct_id,
            type=ConstructType.FILE_TREE,
            title=title,
            content=tree,
            position=position or self._get_next_position(),
            metadata={"root_path": root_path}
        )
        
        self.constructs[construct_id] = construct
        
        await self._broadcast({
            "action": "create",
            **construct.to_dict()
        })
        
        log.info(f"Created file tree: {title} ({construct_id})")
        return construct_id
    
    async def create_chart(
        self,
        data: List[Dict],
        title: str = "Chart",
        chart_type: str = "line",
        position: Optional[Position] = None
    ) -> str:
        """Create a data chart construct"""
        construct_id = f"chart_{uuid.uuid4().hex[:8]}"
        
        construct = Construct(
            id=construct_id,
            type=ConstructType.DATA_CHART,
            title=title,
            content=data,
            position=position or self._get_next_position(),
            metadata={"chart_type": chart_type}
        )
        
        self.constructs[construct_id] = construct
        
        await self._broadcast({
            "action": "create",
            **construct.to_dict()
        })
        
        log.info(f"Created chart: {title} ({construct_id})")
        return construct_id
    
    async def create_note(
        self,
        content: str,
        title: str = "Note",
        position: Optional[Position] = None
    ) -> str:
        """Create a note card construct"""
        construct_id = f"note_{uuid.uuid4().hex[:8]}"
        
        construct = Construct(
            id=construct_id,
            type=ConstructType.NOTE_CARD,
            title=title,
            content=content,
            position=position or self._get_next_position(),
        )
        
        self.constructs[construct_id] = construct
        
        await self._broadcast({
            "action": "create",
            **construct.to_dict()
        })
        
        log.info(f"Created note: {title} ({construct_id})")
        return construct_id
    
    async def create_terminal(
        self,
        content: str = "",
        title: str = "Terminal",
        position: Optional[Position] = None
    ) -> str:
        """Create a terminal output construct"""
        construct_id = f"term_{uuid.uuid4().hex[:8]}"
        
        construct = Construct(
            id=construct_id,
            type=ConstructType.TERMINAL,
            title=title,
            content=content,
            position=position or self._get_next_position(),
        )
        
        self.constructs[construct_id] = construct
        
        await self._broadcast({
            "action": "create",
            **construct.to_dict()
        })
        
        log.info(f"Created terminal: {title} ({construct_id})")
        return construct_id
    
    async def create_markdown(
        self,
        content: str,
        title: str = "Document",
        position: Optional[Position] = None
    ) -> str:
        """Create a markdown document construct"""
        construct_id = f"md_{uuid.uuid4().hex[:8]}"
        
        construct = Construct(
            id=construct_id,
            type=ConstructType.MARKDOWN,
            title=title,
            content=content,
            position=position or self._get_next_position(),
        )
        
        self.constructs[construct_id] = construct
        
        await self._broadcast({
            "action": "create",
            **construct.to_dict()
        })
        
        log.info(f"Created markdown: {title} ({construct_id})")
        return construct_id
    
    # === Construct manipulation ===
    
    async def update_content(self, construct_id: str, content: Any):
        """Update the content of a construct"""
        if construct_id not in self.constructs:
            log.warning(f"Cannot update unknown construct: {construct_id}")
            return False
        
        self.constructs[construct_id].content = content
        
        await self._broadcast({
            "action": "update",
            "id": construct_id,
            "content": content
        })
        
        log.debug(f"Updated construct content: {construct_id}")
        return True
    
    async def append_content(self, construct_id: str, content: str):
        """Append content to a construct (useful for terminals)"""
        if construct_id not in self.constructs:
            return False
        
        existing = self.constructs[construct_id].content
        if isinstance(existing, str):
            self.constructs[construct_id].content = existing + content
        
        await self._broadcast({
            "action": "append",
            "id": construct_id,
            "content": content
        })
        
        return True
    
    async def update_title(self, construct_id: str, title: str):
        """Update the title of a construct"""
        if construct_id not in self.constructs:
            return False
        
        self.constructs[construct_id].title = title
        
        await self._broadcast({
            "action": "update_title",
            "id": construct_id,
            "title": title
        })
        
        log.debug(f"Updated construct title: {construct_id} -> {title}")
        return True
    
    async def move_construct(self, construct_id: str, x: int, y: int):
        """Move a construct to a new position"""
        if construct_id not in self.constructs:
            return False
        
        self.constructs[construct_id].position.x = x
        self.constructs[construct_id].position.y = y
        
        await self._broadcast({
            "action": "move",
            "id": construct_id,
            "position": {"x": x, "y": y}
        })
        
        log.debug(f"Moved construct: {construct_id} to ({x}, {y})")
        return True
    
    async def close_construct(self, construct_id: str):
        """Close/remove a construct"""
        if construct_id not in self.constructs:
            # Try to find by partial match or title
            for cid, construct in list(self.constructs.items()):
                if construct_id.lower() in construct.title.lower():
                    construct_id = cid
                    break
            else:
                log.warning(f"Cannot close unknown construct: {construct_id}")
                return False
        
        del self.constructs[construct_id]
        
        await self._broadcast({
            "action": "close",
            "id": construct_id
        })
        
        log.info(f"Closed construct: {construct_id}")
        return True
    
    async def close_all(self):
        """Close all constructs"""
        self.constructs.clear()
        
        await self._broadcast({
            "action": "close_all"
        })
        
        log.info("Closed all constructs")
    
    def get_construct(self, construct_id: str) -> Optional[Construct]:
        """Get a construct by ID"""
        return self.constructs.get(construct_id)
    
    def list_constructs(self) -> List[Dict]:
        """List all active constructs"""
        return [
            {"id": c.id, "type": c.type.value, "title": c.title}
            for c in self.constructs.values()
        ]
    
    def find_by_title(self, title: str) -> Optional[Construct]:
        """Find a construct by title (partial match)"""
        title_lower = title.lower()
        for construct in self.constructs.values():
            if title_lower in construct.title.lower():
                return construct
        return None


# Singleton instance
_manager: Optional[ConstructManager] = None

def get_construct_manager() -> ConstructManager:
    """Get the global construct manager instance"""
    global _manager
    if _manager is None:
        _manager = ConstructManager()
    return _manager
