"""
Workshop Terminal UI
A clean, configurable terminal output system using Rich.

Supports three verbosity levels:
- quiet: Minimal output (errors, final responses only)
- normal: Clean UX with status updates (default)
- verbose: Full debug output with hook traces

Usage:
    from terminal_ui import get_terminal, Verbosity

    term = get_terminal()
    term.init_start()
    term.init_done(skills=10, tools=55, session_id="sess_xxx")

    # Conversation
    term.user_input("Hello")
    term.thinking()
    term.tool_call("web_search", {"query": "..."})
    term.assistant_response("Here's what I found...")

    # Voice mode
    term.voice_wake()
    term.voice_listening()
    term.voice_transcribed("Hello workshop")
    term.voice_speaking()
    term.voice_ready()
"""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live
from rich.style import Style
from rich.markup import escape


class Verbosity(Enum):
    """Output verbosity levels."""
    QUIET = "quiet"      # Errors and responses only
    NORMAL = "normal"    # Clean status updates (default)
    VERBOSE = "verbose"  # Full debug with hook traces


# Color palette optimized for dark terminals (Ghostty, etc.)
COLORS = {
    "brand": "#7C3AED",       # Purple - Workshop brand
    "success": "#10B981",     # Green
    "warning": "#F59E0B",     # Amber
    "error": "#EF4444",       # Red
    "info": "#6366F1",        # Indigo
    "muted": "#6B7280",       # Gray
    "user": "#3B82F6",        # Blue - user input
    "assistant": "#10B981",   # Green - assistant response
    "tool": "#8B5CF6",        # Purple - tool calls
}


@dataclass
class TerminalConfig:
    """Terminal UI configuration."""
    verbosity: Verbosity = Verbosity.NORMAL
    show_spinners: bool = True
    show_tool_calls: bool = True
    show_hook_traces: bool = False
    compact_init: bool = True


class TerminalUI:
    """
    Centralized terminal output manager.

    Provides a clean, consistent interface for all Workshop terminal output.
    Uses Rich for formatting and supports multiple verbosity levels.
    """

    def __init__(self, config: Optional[TerminalConfig] = None):
        self.config = config or TerminalConfig()
        self.console = Console(highlight=False)
        self._live: Optional[Live] = None
        self._spinner_active = False

    def set_verbosity(self, level: Verbosity) -> None:
        """Set output verbosity level."""
        self.config.verbosity = level
        # Adjust related settings based on verbosity
        if level == Verbosity.QUIET:
            self.config.show_spinners = False
            self.config.show_tool_calls = False
            self.config.show_hook_traces = False
            self.config.compact_init = True
        elif level == Verbosity.VERBOSE:
            self.config.show_hook_traces = True
            self.config.compact_init = False

    # =========================================================================
    # Initialization Output
    # =========================================================================

    def init_start(self) -> None:
        """Show initialization started."""
        if self.config.verbosity == Verbosity.QUIET:
            return

        if self.config.compact_init:
            self.console.print(
                f"[{COLORS['brand']}]Workshop[/] initializing...",
                end=" "
            )
        else:
            self.console.print(
                f"\n[bold {COLORS['brand']}]Workshop[/bold {COLORS['brand']}] initializing...\n"
            )

    def init_step(self, step: str) -> None:
        """Show initialization step (verbose mode only)."""
        if self.config.verbosity != Verbosity.VERBOSE:
            return
        self.console.print(f"  [{COLORS['muted']}]\u2192[/] {step}")

    def init_done(
        self,
        skills: int = 0,
        tools: int = 0,
        session_id: str = "",
        log_file: str = ""
    ) -> None:
        """Show initialization complete."""
        if self.config.verbosity == Verbosity.QUIET:
            return

        if self.config.compact_init:
            session_short = session_id[:12] + "..." if len(session_id) > 12 else session_id
            self.console.print(
                f"[{COLORS['success']}]ready[/] "
                f"[{COLORS['muted']}]({skills} skills, {tools} tools)[/]"
            )
        else:
            self.console.print(
                f"\n[{COLORS['success']}]\u2713[/] Ready! "
                f"[{COLORS['muted']}]{skills} skills, {tools} tools[/]"
            )
            if session_id:
                self.console.print(f"  [{COLORS['muted']}]Session: {session_id}[/]")
            if log_file:
                self.console.print(f"  [{COLORS['muted']}]Log: {log_file}[/]")

    def init_voice_done(self) -> None:
        """Show voice stack ready (compact)."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        self.console.print(
            f"[{COLORS['success']}]\u2713[/] Voice stack ready"
        )

    # =========================================================================
    # Conversation Output
    # =========================================================================

    def user_input(self, text: str, source: str = "text") -> None:
        """Show user input."""
        if self.config.verbosity == Verbosity.QUIET:
            return

        icon = "\U0001F3A4" if source == "voice" else "\u276F"  # microphone or chevron
        self.console.print(
            f"\n[bold {COLORS['user']}]{icon} You:[/] {escape(text)}"
        )

    def thinking(self) -> None:
        """Show thinking indicator."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        if not self.config.show_spinners:
            return

        self.console.print(
            f"[{COLORS['muted']}]\u2022 Thinking...[/]",
            end="\r"
        )

    def thinking_done(self) -> None:
        """Clear thinking indicator."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        # Clear the line
        self.console.print(" " * 20, end="\r")

    def assistant_response(self, text: str) -> None:
        """Show assistant response."""
        self.thinking_done()
        self.console.print(
            f"[bold {COLORS['assistant']}]\u276F Workshop:[/] {escape(text)}\n"
        )

    def tool_call(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> None:
        """Show tool being called."""
        if not self.config.show_tool_calls:
            return
        if self.config.verbosity == Verbosity.QUIET:
            return

        # Compact tool display
        args_preview = ""
        if args and self.config.verbosity == Verbosity.VERBOSE:
            # Show first arg value as preview
            first_val = next(iter(args.values()), None) if args else None
            if first_val:
                preview = str(first_val)[:30]
                if len(str(first_val)) > 30:
                    preview += "..."
                args_preview = f" [{COLORS['muted']}]{escape(preview)}[/]"

        self.console.print(
            f"  [{COLORS['tool']}]\u2192 {tool_name}[/]{args_preview}"
        )

    def tool_result(self, tool_name: str, success: bool = True, preview: str = "") -> None:
        """Show tool result (verbose mode only)."""
        if self.config.verbosity != Verbosity.VERBOSE:
            return

        icon = "\u2713" if success else "\u2717"
        color = COLORS['success'] if success else COLORS['error']

        result_text = f"  [{color}]{icon} {tool_name}[/]"
        if preview:
            result_text += f" [{COLORS['muted']}]{escape(preview[:50])}[/]"

        self.console.print(result_text)

    # =========================================================================
    # Voice Mode Output
    # =========================================================================

    def voice_mode_start(self, wake_word: str = "alexa") -> None:
        """Show voice mode started."""
        if self.config.verbosity == Verbosity.QUIET:
            self.console.print(f"Voice mode active. Say '{wake_word}' or type '/text <message>'.")
            return

        self.console.print(
            Panel(
                f"[bold]Voice Mode Active[/bold]\n"
                f"[{COLORS['muted']}]Say '[bold]{wake_word}[/bold]' to activate \u2022 "
                f"Type '[bold]/text[/bold] message' for text input \u2022 "
                f"Say 'goodbye' or type 'exit' to quit[/]",
                border_style=COLORS['brand'],
                padding=(0, 2)
            )
        )

    def voice_wake(self) -> None:
        """Show wake word detected."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        self.console.print(f"\n[bold {COLORS['brand']}]\u2728 Listening...[/]")

    def voice_listening(self, duration: float = 0) -> None:
        """Show listening for speech."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        if duration > 0:
            self.console.print(
                f"  [{COLORS['muted']}]Captured {duration:.1f}s[/]",
                end="\r"
            )

    def voice_transcribing(self) -> None:
        """Show transcription in progress."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        if self.config.verbosity == Verbosity.VERBOSE:
            self.console.print(f"  [{COLORS['muted']}]Transcribing...[/]")

    def voice_transcribed(self, text: str) -> None:
        """Show transcription result."""
        self.user_input(text, source="voice")

    def voice_speaking(self) -> None:
        """Show TTS in progress."""
        if self.config.verbosity != Verbosity.VERBOSE:
            return
        self.console.print(f"  [{COLORS['muted']}]Speaking...[/]")

    def voice_ready(self) -> None:
        """Show ready for next wake word."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        self.console.print(f"[{COLORS['muted']}]\u2022 Ready[/]")

    def voice_no_speech(self) -> None:
        """Show no speech detected."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        self.console.print(f"  [{COLORS['muted']}](no speech detected)[/]")

    # =========================================================================
    # Hook System Output (verbose only)
    # =========================================================================

    def hook_start(self, hook_type: str, handler_count: int, tool_name: str = "") -> None:
        """Show hook execution starting."""
        if not self.config.show_hook_traces:
            return

        icon = "\U0001F517" if hook_type == "session_start" else "\U0001F527"
        extra = f" [{tool_name}]" if tool_name else ""

        self.console.print(
            f"\n[{COLORS['muted']}]{icon} HOOKS \u2500 {hook_type.upper()}{extra} "
            f"({handler_count} handlers)[/]"
        )

    def hook_handler(self, priority: int, name: str, status: str = "running") -> None:
        """Show individual hook handler status."""
        if not self.config.show_hook_traces:
            return

        if status == "running":
            self.console.print(
                f"  [{COLORS['muted']}]\u251C\u2500 [{priority:02d}] {name}...[/]",
                end=" "
            )
        elif status == "success":
            self.console.print(f"[{COLORS['success']}]\u2713[/]")
        elif status == "error":
            self.console.print(f"[{COLORS['error']}]\u2717[/]")

    def hook_summary(self, sections: int, chars: int) -> None:
        """Show hook execution summary."""
        if not self.config.show_hook_traces:
            return
        self.console.print(
            f"  [{COLORS['muted']}]\u2514\u2500 {sections} sections, {chars} chars[/]"
        )

    # =========================================================================
    # Status & Errors
    # =========================================================================

    def status(self, message: str) -> None:
        """Show general status message."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        self.console.print(f"[{COLORS['info']}]\u2022[/] {message}")

    def warning(self, message: str) -> None:
        """Show warning message."""
        self.console.print(f"[{COLORS['warning']}]\u26A0 {message}[/]")

    def error(self, message: str) -> None:
        """Show error message."""
        self.console.print(f"[bold {COLORS['error']}]\u2717 {message}[/]")

    def success(self, message: str) -> None:
        """Show success message."""
        self.console.print(f"[{COLORS['success']}]\u2713 {message}[/]")

    def info(self, message: str) -> None:
        """Show info message (normal+ verbosity)."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        self.console.print(f"[{COLORS['muted']}]{message}[/]")

    def debug(self, message: str) -> None:
        """Show debug message (verbose only)."""
        if self.config.verbosity != Verbosity.VERBOSE:
            return
        self.console.print(f"[dim]{message}[/dim]")

    # =========================================================================
    # Session Management
    # =========================================================================

    def session_info(self, session_id: str, mode: str = "text") -> None:
        """Show session info."""
        if self.config.verbosity != Verbosity.VERBOSE:
            return
        self.console.print(
            f"[{COLORS['muted']}]Session: {session_id} ({mode})[/]"
        )

    def text_mode_prompt(self) -> str:
        """Get text mode prompt string."""
        return f"[{COLORS['user']}]You:[/] "

    def goodbye(self) -> None:
        """Show goodbye message."""
        self.console.print(f"\n[{COLORS['brand']}]\U0001F44B Goodbye![/]\n")

    def cleanup_start(self) -> None:
        """Show cleanup starting."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        self.console.print(f"\n[{COLORS['muted']}]Cleaning up...[/]", end=" ")

    def cleanup_done(self) -> None:
        """Show cleanup complete."""
        if self.config.verbosity == Verbosity.QUIET:
            return
        self.console.print(f"[{COLORS['success']}]done[/]")


# =============================================================================
# Singleton Instance
# =============================================================================

_terminal_instance: Optional[TerminalUI] = None


def get_terminal(config: Optional[TerminalConfig] = None) -> TerminalUI:
    """
    Get the global TerminalUI singleton.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        Shared TerminalUI instance
    """
    global _terminal_instance
    if _terminal_instance is None:
        _terminal_instance = TerminalUI(config)
    return _terminal_instance


def reset_terminal() -> None:
    """Reset the singleton (for testing)."""
    global _terminal_instance
    _terminal_instance = None


def configure_terminal(
    verbosity: str = "normal",
    show_tool_calls: bool = True,
    show_hook_traces: bool = False
) -> TerminalUI:
    """
    Configure and return the terminal instance.

    Convenience function for setting up terminal from CLI args.

    Args:
        verbosity: "quiet", "normal", or "verbose"
        show_tool_calls: Whether to show tool calls
        show_hook_traces: Whether to show hook execution details

    Returns:
        Configured TerminalUI instance
    """
    reset_terminal()

    verb = Verbosity(verbosity) if verbosity in ["quiet", "normal", "verbose"] else Verbosity.NORMAL

    config = TerminalConfig(
        verbosity=verb,
        show_tool_calls=show_tool_calls,
        show_hook_traces=show_hook_traces,
        compact_init=(verb != Verbosity.VERBOSE)
    )

    return get_terminal(config)