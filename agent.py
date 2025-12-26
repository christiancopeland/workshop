"""
Workshop Agent
Core LLM integration with function calling via Ollama
"""

import json
import re
import aiohttp
from typing import Any, Optional, List, Dict
from datetime import datetime

from config import Config
from logger import get_logger

log = get_logger("agent")


class Agent:
    """LLM agent with tool use capabilities"""
    
    # Patterns that indicate user wants a tool action
    # NOTE: These are matched against the ORIGINAL input (not lowercased)
    # using re.IGNORECASE so we preserve the original case in captured groups
    TOOL_INTENT_PATTERNS = {
        "read_file": [
            r"read\s+(?:the\s+)?(?:file\s+)?([^\s]+\.[\w]+)",
            r"show\s+(?:me\s+)?(?:the\s+)?(?:file\s+)?([^\s]+\.[\w]+)",
            r"open\s+(?:the\s+)?(?:file\s+)?([^\s]+\.[\w]+)",
            r"cat\s+([^\s]+\.[\w]+)",
            r"what(?:'s| is)\s+in\s+([^\s]+\.[\w]+)",
            r"display\s+([^\s]+\.[\w]+)",
            r"view\s+([^\s]+\.[\w]+)",
        ],
        "list_directory": [
            r"(?:list|ls)\s+(?:the\s+)?(?:files\s+)?(?:in\s+)?([~/][\w\./\-]+)",
            r"show\s+(?:me\s+)?(?:the\s+)?(?:files|directory|folder|contents)\s+(?:in\s+|of\s+)?([~/][\w\./\-]+)",
            r"what\s+files\s+(?:are\s+)?(?:in\s+)?([~/][\w\./\-]+)",
            r"what(?:'s| is)\s+in\s+(?:the\s+)?(?:folder|directory|dir)\s+([~/][\w\./\-]+)",
            r"what(?:'s| is)\s+in\s+([~/][\w\./\-]+)",
            r"files\s+in\s+([~/][\w\./\-]+)",
            r"contents\s+of\s+([~/][\w\./\-]+)",
        ],
        "search_files": [
            r"search\s+(?:for\s+)?['\"]?(.+?)['\"]?\s+in\s+([~/][\w\./\-]+)",
            r"find\s+['\"]?(.+?)['\"]?\s+in\s+([~/][\w\./\-]+)",
            r"grep\s+['\"]?(.+?)['\"]?\s+([~/][\w\./\-]+)",
            r"look\s+for\s+['\"]?(.+?)['\"]?\s+in\s+([~/][\w\./\-]+)",
        ],
        "write_file": [
            r"write\s+(?:to\s+)?(?:the\s+)?(?:file\s+)?([~/][\w\./\-]+)",
            r"create\s+(?:a\s+)?(?:file\s+)?([~/][\w\./\-]+)",
            r"save\s+(?:to\s+)?([~/][\w\./\-]+)",
        ],
        "remember": [
            # Direct content patterns - match when content is explicitly stated
            # "remember the project path is ~/foo"
            # "remember my API key is xyz"
            r"remember\s+(?:that\s+)?(?:the\s+)?(\w+\s+(?:is|are|was|were|=)\s+.+)",
            # "remember: some info"
            r"remember\s*:\s*(.+)",
            # "store that X", "don't forget X" - explicit content
            r"store\s+(?:that\s+)?(?:the\s+)?(\w+\s+(?:is|are|=)\s+.+)",
            r"don't\s+forget\s+(?:that\s+)?(?:the\s+)?(\w+\s+(?:is|are|=)\s+.+)",
            # Everything else like "remember that path I gave you" falls through to LLM
        ],
        "recall": [
            r"what\s+do\s+you\s+(?:remember|recall|know)\s+about\s+(.+)",
            r"recall\s+(?:info(?:rmation)?\s+)?(?:about\s+)?(.+)",
            r"what\s+did\s+(?:i|we)\s+(?:say|tell\s+you)\s+about\s+(.+)",
        ],
        "take_note": [
            r"take\s+(?:a\s+)?note\s+(?:titled?\s+)?['\"](.+?)['\"]?\s+(?:with\s+(?:content\s+)?|saying\s+|:\s*)['\"]?(.+)['\"]?",
            r"make\s+(?:a\s+)?note\s+(?:titled?\s+)?['\"](.+?)['\"]?\s+(?:with\s+(?:content\s+)?|saying\s+|:\s*)['\"]?(.+)['\"]?",
            r"note\s*:\s*(.+)",
            r"jot\s+down\s+(.+)",
        ],
        "run_shell": [
            r"run\s+(?:the\s+)?(?:command\s+)?['\"]?(.+)['\"]?$",
            r"execute\s+['\"]?(.+)['\"]?$",
            r"shell\s*:\s*(.+)",
        ],
        "web_search": [
            r"search\s+(?:the\s+)?(?:web|internet|online)\s+(?:for\s+)?(.+)",
            r"google\s+(.+)",
            r"look\s+up\s+(.+)\s+online",
        ],
        "calculate": [
            r"calculate\s+(.+)",
            r"compute\s+(.+)",
            r"what(?:'s| is)\s+(\d+[\s\d\+\-\*\/\^\.\(\)]+\d)",
            r"math\s*:\s*(.+)",
        ],
        # Construct tools - visual displays
        "show_file": [
            r"show\s+(?:me\s+)?(?:the\s+)?(?:code\s+(?:for|in)\s+)?([~/]?[\w\./\-]+\.[\w]+)",
            r"display\s+(?:the\s+)?(?:code\s+(?:for|in)\s+)?([~/]?[\w\./\-]+\.[\w]+)",
            r"pull\s+up\s+([~/]?[\w\./\-]+\.[\w]+)",
            r"show\s+(?:me\s+)?([~/]?[\w\./\-]+\.[\w]+)\s+(?:code|file)",
        ],
        "show_directory": [
            r"show\s+(?:me\s+)?(?:the\s+)?(?:files|contents)\s+(?:in|of)\s+([~/]?[\w\./\-]+)",
            r"display\s+(?:the\s+)?directory\s+([~/]?[\w\./\-]+)",
        ],
        "create_note": [
            r"create\s+(?:a\s+)?note\s+(?:about|on|for)\s+(.+)",
            r"make\s+(?:a\s+)?note\s+(?:about|on|for)\s+(.+)",
        ],
        "close_construct": [
            r"close\s+(?:the\s+)?construct\s+([a-f0-9\-]+)",
            r"dismiss\s+(?:the\s+)?construct\s+([a-f0-9\-]+)",
        ],
        "close_all_constructs": [
            r"close\s+(?:all\s+)?(?:the\s+)?constructs?",
            r"dismiss\s+(?:all\s+)?(?:the\s+)?constructs?",
            r"clear\s+(?:all\s+)?(?:the\s+)?constructs?",
        ],
        "list_constructs": [
            r"list\s+(?:all\s+)?(?:the\s+)?constructs?",
            r"show\s+(?:all\s+)?(?:the\s+)?constructs?",
            r"what\s+constructs\s+(?:are\s+)?(?:open|active)",
        ],
        # Arduino/ESP32 tools
        "arduino_compile": [
            r"^compile\s+this$",  # NEW: "compile this" with context
            r"^build\s+this$",    # NEW: "build this" with context
            r"^compile$",  # Bare "compile" - use active project
            r"^build$",    # Bare "build" - use active project
            r"^verify$",   # Bare "verify" - use active project
            r"^(?:let'?s\s+)?compile(?:\s+(?:it\s+)?again)?$",  # "compile again", "let's compile"
            r"^(?:let'?s\s+)?build(?:\s+(?:it\s+)?again)?$",
            r"^(?:re)?compile(?:\s+it)?$",  # "recompile", "compile it"
            r"compile\s+(?!again|it)(.+?)(?:\s+for\s+(\w+))?$",  # Negative lookahead for "again"/"it"
            r"build\s+(?!again|it)(.+?)(?:\s+for\s+(\w+))?$",
            r"verify\s+(?!again)(.+?)$",
        ],
        "arduino_upload": [
            r"^upload$",   # Bare "upload" - use active project
            r"^flash$",    # Bare "flash" - use active project
            r"^(?:let'?s\s+)?upload(?:\s+(?:it\s+)?again)?$",  # "upload again", "let's upload"
            r"^(?:re)?upload(?:\s+it)?$",  # "reupload", "upload it"
            r"upload\s+(?!again|it)(.+?)(?:\s+to\s+([\/\w]+))?$",
            r"flash\s+(?!again|it)(.+?)(?:\s+to\s+([\/\w]+))?$",
            r"program\s+(.+?)(?:\s+(?:to|on)\s+([\/\w]+))?$",
        ],
        "arduino_monitor": [
            r"(?:start|open)\s+(?:the\s+)?serial\s+(?:monitor)?",
            r"serial\s+monitor",
            r"show\s+(?:me\s+)?serial\s+(?:output)?",
            r"(?:read|watch)\s+(?:the\s+)?serial",
        ],
        "arduino_boards": [
            r"(?:list|show|what)\s+(?:are\s+)?(?:the\s+)?(?:connected\s+)?boards?",
            r"what(?:'s| is)\s+(?:plugged\s+in|connected)",
        ],
    }
    
    def __init__(
        self,
        model: str,
        tools: "ToolRegistry",
        memory: "MemorySystem",
        ollama_url: str = "http://localhost:11434",
        construct_manager: "ConstructManager" = None,
        context_manager: "ContextAwareness" = None,  # NEW: Added context_manager parameter
    ):
        self.model = model
        self.tools = tools
        self.memory = memory
        self.ollama_url = ollama_url
        self.config = Config()
        self.construct_manager = construct_manager
        self.context_manager = context_manager  # FIXED: Now properly assigned from parameter
        
        log.info(f"Agent initialized with model: {model}")
        log.info(f"Ollama URL: {ollama_url}")
        log.debug(f"Available tools: {list(tools.get_all_tools().keys())}")
        if construct_manager:
            log.info("Construct manager attached")
        if context_manager:
            log.info("Context awareness enabled")
    
    def _should_inject_context(self, user_input: str) -> bool:
        """
        Determine if we should inject development context for this query.
        Returns True if the query is related to development/hardware work.
        """
        # Keywords that indicate development context would be helpful
        dev_keywords = [
            # Arduino/hardware
            'compile', 'upload', 'flash', 'build', 'verify', 'board', 'sketch',
            'arduino', 'esp32', 'nano', 'serial', 'monitor', 'firmware',
            # Files/code
            'file', 'directory', 'folder', 'code', 'script', 'program',
            '.ino', '.cpp', '.h', '.py', '.js',
            # Development actions
            'debug', 'test', 'run', 'execute', 'open', 'show', 'edit',
            'error', 'bug', 'fix', 'issue', 'problem',
            # Context queries
            'working on', 'project', 'current', 'this', 'recent',
            'connected', 'device', 'port',
        ]
        
        input_lower = user_input.lower()
        
        # Check for dev keywords
        for keyword in dev_keywords:
            if keyword in input_lower:
                return True
        
        # Check for file extensions
        if re.search(r'\.\w{1,4}\b', user_input):
            return True
        
        # Don't inject for greetings/small talk
        if len(user_input.split()) <= 3 and any(greeting in input_lower for greeting in [
            'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'goodbye'
        ]):
            return False
        
        return False
    
    async def chat(self, user_input: str) -> str:
        """
        Process user input and generate response.
        Handles tool calls automatically with automatic context injection.
        """
        log.info(f"=" * 50)
        log.info(f"USER INPUT: {user_input}")
        log.info(f"=" * 50)
        
        # === STEP 0: AUTO-INJECT CONTEXT (NEW) ===
        enriched_input = user_input
        context_injected = False
        
        if self.context_manager and self._should_inject_context(user_input):
            try:
                # Get current development context (Phase 3: returns assembled context dict)
                context = self.context_manager.get_context()

                # Format context for LLM
                context_str = self.context_manager.format_context_for_llm(context)
                
                if context_str:
                    # Inject context into the input
                    enriched_input = f"{user_input}\n\n[Current Context:\n{context_str}]"
                    context_injected = True
                    log.info(f"CONTEXT INJECTED ({len(context_str)} chars)")
                    log.debug(f"Context:\n{context_str}")
            
            except Exception as e:
                log.warning(f"Context injection failed: {e}")
                # Continue with original input if context fails
        
        # === STEP 1: Check for direct tool intent (using enriched input) ===
        detected_intent = self._detect_tool_intent(enriched_input)
        if detected_intent:
            log.info(f"DETECTED TOOL INTENT: {detected_intent['tool']}")
            log.info(f"  Args: {detected_intent['args']}")
            
            # Execute tool directly without asking LLM
            result = await self._execute_tool(detected_intent)
            log.info(f"DIRECT TOOL RESULT: {result[:200]}...")
            
            # Format the response nicely using LLM
            return await self._format_tool_result_response(user_input, detected_intent, result)
        
        # === STEP 2: No clear intent - let LLM decide (with enriched input) ===
        log.info("No direct tool intent detected, using LLM...")
        if context_injected:
            log.debug("LLM will receive context-enriched input")
        
        # Build context
        context = self._build_context(enriched_input)  # Use enriched input
        log.debug(f"Context: profile={len(context.get('user_profile') or '')} chars, " +
                f"memories={len(context.get('relevant_memories', []))}, " +
                f"recent={len(context.get('recent_messages', []))}")
        
        # Build messages for LLM
        messages = self._build_messages(enriched_input, context)  # Use enriched input
        log.debug(f"Built {len(messages)} messages for LLM")
        
        # Generate response with tools
        response = await self._generate_with_tools(messages)
        log.info(f"FINAL RESPONSE: {response[:200]}...")
        
        return response
    
    async def _format_tool_result_response(
        self, 
        user_input: str, 
        tool_call: Dict, 
        result: str
    ) -> str:
        """Ask LLM to format tool result into a nice response"""
        
        # For errors, return directly without LLM formatting
        if result.startswith("Error:"):
            return result
        
        # For short results, return directly
        if len(result) < 500:
            return result
        
        prompt = f"""The user asked: "{user_input}"

I executed the '{tool_call['tool']}' tool with args {json.dumps(tool_call['args'])}.

Result:
```
{result[:3000]}
```

Provide a brief, helpful summary of this result. Be concise (2-3 sentences max)."""
        
        messages = [
            {"role": "system", "content": "Summarize tool results concisely. Don't repeat the full content."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self._call_ollama(messages)
        return response if response and not response.startswith("Error") else result
    
    def _detect_tool_intent(self, user_input: str) -> Optional[Dict]:
        """
        Detect if user input clearly indicates a tool should be used.
        Returns tool name and extracted arguments if detected.
        
        IMPORTANT: We match against the ORIGINAL input using re.IGNORECASE
        so that captured groups preserve the original case (e.g., ~/Arduino not ~/arduino)
        """
        original_input = user_input.strip()
        
        # When constructs are enabled, check construct patterns FIRST
        # This ensures "show me file.py" uses show_file (visual) not read_file (text)
        if self.construct_manager:
            construct_tools = [
                "show_file", "show_directory", "create_note", 
                "close_construct", "close_all_constructs", "list_constructs"
            ]
            for tool_name in construct_tools:
                if tool_name in self.TOOL_INTENT_PATTERNS:
                    for pattern in self.TOOL_INTENT_PATTERNS[tool_name]:
                        match = re.search(pattern, original_input, re.IGNORECASE)
                        if match:
                            groups = match.groups()
                            log.debug(f"Construct pattern matched: tool={tool_name}, groups={groups}")
                            args = self._build_tool_args(tool_name, groups)
                            if args is not None:
                                return {"tool": tool_name, "args": args}
        
        # Standard pattern matching
        for tool_name, patterns in self.TOOL_INTENT_PATTERNS.items():
            # Skip construct tools in second pass if we already checked them
            if self.construct_manager and tool_name in [
                "show_file", "show_directory", "create_note",
                "close_construct", "close_all_constructs", "list_constructs"
            ]:
                continue
                
            for pattern in patterns:
                # Match against ORIGINAL input, not lowercased
                match = re.search(pattern, original_input, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    log.debug(f"Pattern matched: tool={tool_name}, groups={groups}")
                    
                    # When constructs enabled, upgrade text tools to visual tools
                    if self.construct_manager:
                        if tool_name == "read_file":
                            tool_name = "show_file"
                            log.debug(f"Upgraded to construct tool: show_file")
                        elif tool_name == "list_directory":
                            tool_name = "show_directory"
                            log.debug(f"Upgraded to construct tool: show_directory")
                    
                    # Build args based on tool - groups preserve original case
                    args = self._build_tool_args(tool_name, groups)
                    
                    if args is not None:
                        return {"tool": tool_name, "args": args}
        
        return None
    
    def _build_tool_args(self, tool_name: str, groups: tuple) -> Optional[Dict]:
        """Build tool arguments from regex capture groups"""
        # Clean up captured values
        def clean(s):
            if s:
                return s.strip().rstrip('.,;:!?')
            return s
        
        if tool_name == "read_file":
            return {"path": clean(groups[0])} if groups else None
        
        elif tool_name == "show_file":
            # Same as read_file but displays visually
            return {"path": clean(groups[0])} if groups else None
        
        elif tool_name == "list_directory":
            return {"path": clean(groups[0])} if groups else None
        
        elif tool_name == "show_directory":
            # Same as list_directory but displays visually
            return {"path": clean(groups[0])} if groups else None
        
        elif tool_name == "search_files" and len(groups) >= 2:
            return {"query": clean(groups[0]), "path": clean(groups[1])}
        
        elif tool_name == "write_file":
            return {"path": clean(groups[0])} if groups else None
        
        elif tool_name == "remember":
            return {"content": clean(groups[0])} if groups else None
        
        elif tool_name == "recall":
            return {"query": clean(groups[0])} if groups else None
        
        elif tool_name == "take_note":
            if len(groups) >= 2:
                return {"title": clean(groups[0]), "content": clean(groups[1])}
            return {"title": "Quick Note", "content": clean(groups[0])} if groups else None
        
        elif tool_name == "run_shell":
            return {"command": clean(groups[0])} if groups else None
        
        elif tool_name == "web_search":
            return {"query": clean(groups[0])} if groups else None
        
        elif tool_name == "calculate":
            return {"expression": clean(groups[0])} if groups else None
        
        # Construct tools
        elif tool_name == "show_file":
            return {"path": clean(groups[0])} if groups else None
        
        elif tool_name == "show_directory":
            return {"path": clean(groups[0])} if groups else None
        
        elif tool_name == "create_note":
            return {"title": clean(groups[0])} if groups else None
        
        elif tool_name == "close_construct":
            return {"construct_id": clean(groups[0])} if groups else None
        
        elif tool_name == "close_all_constructs":
            return {}
        
        elif tool_name == "list_constructs":
            return {}
        
        # Arduino tools
        elif tool_name == "arduino_compile":
            if not groups or not groups[0]:
                # Bare "compile" - use active project
                return {}
            sketch = clean(groups[0]) if groups else None
            board = clean(groups[1]) if len(groups) > 1 and groups[1] else None
            args = {}
            if sketch:
                args["sketch"] = sketch
            if board:
                args["board"] = board
            return args
        
        elif tool_name == "arduino_upload":
            if not groups or not groups[0]:
                # Bare "upload" - use active project
                return {}
            sketch = clean(groups[0]) if groups else None
            port = clean(groups[1]) if len(groups) > 1 and groups[1] else None
            args = {}
            if sketch:
                args["sketch"] = sketch
            if port:
                args["port"] = port
            return args
        
        elif tool_name == "arduino_monitor":
            return {}
        
        elif tool_name == "arduino_boards":
            return {}
        
        return None
    
    def _build_context(self, user_input: str) -> dict:
        """Build context for LLM from memory and conversation history"""
        context = {}
        
        # User profile
        profile = self.memory.get_user_profile()
        if profile:
            context["user_profile"] = profile
        
        # Relevant memories
        memories = self.memory.search_memories(user_input, k=3)
        if memories:
            context["relevant_memories"] = memories
        
        # Recent conversation
        recent = self.memory.get_recent_messages(10)
        if recent:
            context["recent_messages"] = recent
        
        return context
    
    def _build_messages(self, user_input: str, context: dict) -> List[Dict]:
        """Build message list for Ollama"""
        messages = []
        
        # System message with tools
        system_msg = self._build_system_message(context)
        messages.append({"role": "system", "content": system_msg})
        
        # Recent conversation history
        recent = context.get("recent_messages", [])
        if recent:
            # Add last few exchanges
            for msg in recent[-6:]:  # Last 3 exchanges
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Current user message
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _build_system_message(self, context: dict) -> str:
        """Build system message with tools and context"""
        parts = [
            "You are Workshop, a helpful local AI assistant with access to tools.",
            "",
            "Available tools:",
        ]
        
        # List tools
        for name, tool_info in self.tools.get_all_tools().items():
            sig = tool_info.get("signature", name)
            desc = tool_info.get("description", "")
            parts.append(f"- {sig}: {desc}")
        
        parts.append("")
        parts.append("To use a tool, respond with JSON:")
        parts.append('{"tool": "tool_name", "args": {"arg1": "value1"}}')
        parts.append("")
        parts.append("You can call multiple tools in one response.")
        parts.append("After tool results, provide a natural response to the user.")
        
        # Add user profile if available
        profile = context.get("user_profile")
        if profile:
            parts.append("")
            parts.append(f"User context: {profile}")
        
        # Add relevant memories
        memories = context.get("relevant_memories", [])
        if memories:
            parts.append("")
            parts.append("Relevant memories:")
            for m in memories[:3]:
                parts.append(f"- {m[:100]}...")
        
        return "\n".join(parts)
    
    def _format_recent_context(self, messages: list, max_lines: int = 10) -> str:
        """Format recent messages for context"""
        lines = []
        for msg in messages[-max_lines:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    async def _generate_with_tools(self, messages: List[Dict]) -> str:
        """Generate response with tool calling support"""
        max_iterations = 5
        iteration = 0
        tool_results = []
        
        while iteration < max_iterations:
            iteration += 1
            log.debug(f"Generation iteration {iteration}/{max_iterations}")
            
            # Call Ollama
            response = await self._call_ollama(messages)
            
            if not response:
                return "I encountered an error. Please try again."
            
            # Check for tool calls
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                # No tools - clean and return
                clean = self._clean_response(response)
                
                # If we have tool results, incorporate them
                if tool_results:
                    return f"{clean}\n\n(Based on: {', '.join([r['tool'] for r in tool_results])})"
                
                return clean
            
            # Execute tools
            log.debug(f"Found {len(tool_calls)} tool calls")
            results = []
            for call in tool_calls:
                result = await self._execute_tool(call)
                results.append({"tool": call["tool"], "result": result})
                tool_results.append({"tool": call["tool"], "result": result})
            
            # Add tool results to conversation
            results_text = self._format_tool_results(results)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": results_text})
        
        return "I ran out of iterations trying to complete your request."
    
    async def _call_ollama(self, messages: List[Dict]) -> Optional[str]:
        """Make API call to Ollama"""
        url = f"{self.ollama_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        log.error(f"Ollama error: {resp.status} - {error_text}")
                        return f"Error: Ollama returned {resp.status}"
                    
                    data = await resp.json()
                    message = data.get("message", {})
                    content = message.get("content", "")
                    return content
        
        except aiohttp.ClientError as e:
            log.error(f"Connection error: {e}")
            return "Error: Could not connect to Ollama. Is it running?"
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
            return f"Error: {e}"
    
    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        
        # Strategy 1: JSON in code blocks
        for match in re.finditer(r'```(?:json)?\s*(\{[^`]+\})\s*```', response, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if "tool" in data:
                    if data not in tool_calls:
                        tool_calls.append(data)
                        log.debug(f"Extracted (code block): {data}")
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Standalone JSON objects
        for match in re.finditer(r'\{[^{}]*"tool"[^{}]*\}', response):
            try:
                call = json.loads(match.group(0))
                if call not in tool_calls:
                    tool_calls.append(call)
                    log.debug(f"Extracted (standalone): {call}")
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: XML-style tags
        for match in re.finditer(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL):
            try:
                call = json.loads(match.group(1))
                if call not in tool_calls:
                    tool_calls.append(call)
                    log.debug(f"Extracted (code block): {call}")
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Raw JSON
        for match in re.finditer(r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[^}]+\})\s*\}', response):
            try:
                args = json.loads(match.group(2))
                call = {"tool": match.group(1), "args": args}
                if call not in tool_calls:
                    tool_calls.append(call)
                    log.debug(f"Extracted (raw): {call}")
            except:
                pass
        
        return tool_calls
    
    async def _execute_tool(self, call: dict) -> str:
        """Execute a tool call"""
        tool_name = call.get("tool")
        args = call.get("args", {})
        
        if not tool_name:
            return "Error: No tool specified"
        
        try:
            result = await self.tools.execute(tool_name, args)
            return str(result)
        except Exception as e:
            log.error(f"Tool error: {e}", exc_info=True)
            return f"Error: {e}"
    
    def _format_tool_results(self, results: list) -> str:
        """Format tool results for the next turn"""
        lines = ["Tool results:\n"]
        for r in results:
            lines.append(f"**{r['tool']}** result:\n```\n{r['result']}\n```\n")
        lines.append("Use these results to answer the user.")
        return "\n".join(lines)
    
    def _clean_response(self, response: str) -> str:
        """Clean up response for user display"""
        response = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
        response = re.sub(r'\n{3,}', '\n\n', response)
        return response.strip()
    
    async def update_user_profile(self, recent_messages: list):
        """Update user profile based on recent conversation"""
        if not recent_messages:
            return
        
        current_profile = self.memory.get_user_profile() or ""
        
        prompt = f"""Update user profile based on conversation. Be concise (max 200 words).

Current: {current_profile}

Recent:
{self._format_recent_context(recent_messages)}

Updated profile:"""
        
        messages = [
            {"role": "system", "content": "Extract key facts about user. Be concise."},
            {"role": "user", "content": prompt}
        ]
        
        new_profile = await self._call_ollama(messages)
        if new_profile and len(new_profile) > 20:
            self.memory.set_user_profile(new_profile)
            log.info(f"Profile updated: {len(new_profile)} chars")