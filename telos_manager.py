"""
Telos Manager - Personal Context System
Manages human-editable context files for personalized AI interactions.

Telos (Greek: τέλος) - "ultimate purpose" or "end goal"
"""

from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import re

from logger import get_logger

log = get_logger("telos_manager")


@dataclass
class TelosContext:
    """Structured personal context from Telos files"""
    profile: str = ""
    goals: str = ""
    mission: str = ""
    project_context: Dict[str, str] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def is_empty(self) -> bool:
        """Check if any context is loaded"""
        return not (self.profile or self.goals or self.mission or self.project_context)


class TelosManager:
    """
    Manages personal context files for Workshop.

    Provides human-editable markdown files that define:
    - User identity and preferences (profile.md)
    - Current objectives (goals.md)
    - Long-term vision (mission.md)
    - Project-specific context (projects/*.md)

    Integrates with ContextManager for combined automatic + human context.
    """

    def __init__(self, telos_dir: Path = None):
        """
        Initialize Telos manager.

        Args:
            telos_dir: Directory containing Telos files (default: ~/.workshop/Telos)
        """
        if telos_dir is None:
            telos_dir = Path.home() / ".workshop" / "Telos"

        self.telos_dir = Path(telos_dir)
        self.projects_dir = self.telos_dir / "projects"

        # Create directories if they don't exist
        self._ensure_directories()

        # Create template files if they don't exist
        self._create_templates_if_needed()

        # Cached context
        self._context: Optional[TelosContext] = None
        self._last_load_time: Optional[datetime] = None

        log.info(f"TelosManager initialized at {self.telos_dir}")

    def _ensure_directories(self):
        """Ensure Telos directories exist"""
        self.telos_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(exist_ok=True)
        log.debug(f"Telos directories ready at {self.telos_dir}")

    def _create_templates_if_needed(self):
        """Create template files if they don't exist"""
        templates = {
            "profile.md": self._get_profile_template(),
            "goals.md": self._get_goals_template(),
            "mission.md": self._get_mission_template(),
        }

        for filename, content in templates.items():
            file_path = self.telos_dir / filename
            if not file_path.exists():
                file_path.write_text(content)
                log.info(f"Created template: {filename}")

    def _get_profile_template(self) -> str:
        """Template for profile.md"""
        return """# Personal Profile

<!-- This file defines who you are, your preferences, and your technical context -->

## Identity
<!-- Your name, role, background -->

I'm a developer working on...

## Preferences

### Communication Style
<!-- How you want the AI to communicate with you -->
- Tone: [casual/professional/technical]
- Detail level: [concise/balanced/verbose]
- Preferred terminology: [list any preferred terms]

### Work Style
<!-- How you work and what you value -->
- I prefer to: [work incrementally, test thoroughly, move fast, etc.]
- I prioritize: [correctness, performance, readability, etc.]

## Technical Context

### Primary Tech Stack
<!-- Languages, frameworks, tools you use regularly -->
- Languages: Python, C++, JavaScript
- Hardware: Arduino, ESP32
- Tools: VS Code, Git

### Expertise Level
<!-- Your skill levels in different areas -->
- Python: Advanced
- C++/Arduino: Intermediate
- Electronics: Learning

## Current Projects
<!-- Brief overview of active projects -->
- Workshop: Voice-first AI development assistant
- Arduino projects: Hardware prototyping

<!-- Edit this file to personalize Workshop's understanding of you -->
"""

    def _get_goals_template(self) -> str:
        """Template for goals.md"""
        return """# Current Goals

<!-- Define your current objectives - update this weekly/monthly -->

## This Week
<!-- What are you trying to accomplish this week? -->

- [ ] Complete Telos integration for Workshop
- [ ] Test voice interaction improvements
- [ ] Finish Arduino sensor calibration

## This Month
<!-- Medium-term goals for this month -->

- [ ] Integrate PAI learnings into Workshop
- [ ] Build spatial UI prototype
- [ ] Complete hardware project X

## Ongoing
<!-- Continuous objectives -->

- Learn more about LLM agent architecture
- Improve Workshop's context intelligence
- Document design decisions

<!-- Workshop will use these goals to provide relevant assistance -->
"""

    def _get_mission_template(self) -> str:
        """Template for mission.md"""
        return """# Mission & Vision

<!-- Your long-term vision and guiding principles -->

## Purpose
<!-- Why are you building what you're building? -->

I'm building Workshop to create a voice-first AI development assistant that understands context and helps me work more efficiently on hardware and software projects.

## Long-Term Vision
<!-- Where do you want to be in 6-12 months? -->

- A production-ready AI assistant for maker projects
- Spatial UI for managing multiple contexts
- Open source tool that helps other developers

## Guiding Principles
<!-- Values that guide your decisions -->

1. **Simplicity First** - Start simple, add complexity only when needed
2. **Learn by Building** - Best way to understand is to implement
3. **Document Decisions** - Future me needs to understand why
4. **Iterate Quickly** - Ship, test, improve

## What Success Looks Like
<!-- How will you know when you've achieved your mission? -->

Workshop becomes my daily driver for:
- Arduino development and debugging
- Python coding assistance
- Context-aware project management
- Voice-first hands-free coding

<!-- This file helps Workshop understand your long-term context -->
"""

    def load_context(self, force_reload: bool = False) -> TelosContext:
        """
        Load personal context from Telos files.

        Args:
            force_reload: Force reload even if recently loaded

        Returns:
            TelosContext with all loaded context
        """
        # Use cache if recent (within 5 minutes)
        if not force_reload and self._context and self._last_load_time:
            age_seconds = (datetime.now() - self._last_load_time).total_seconds()
            if age_seconds < 300:  # 5 minutes
                log.debug("Using cached Telos context")
                return self._context

        context = TelosContext()

        # Load core files
        profile_path = self.telos_dir / "profile.md"
        if profile_path.exists():
            context.profile = self._load_and_process_file(profile_path)

        goals_path = self.telos_dir / "goals.md"
        if goals_path.exists():
            context.goals = self._load_and_process_file(goals_path)

        mission_path = self.telos_dir / "mission.md"
        if mission_path.exists():
            context.mission = self._load_and_process_file(mission_path)

        # Load project-specific context
        if self.projects_dir.exists():
            for project_file in self.projects_dir.glob("*.md"):
                project_name = project_file.stem
                content = self._load_and_process_file(project_file)
                if content:
                    context.project_context[project_name] = content

        context.last_updated = datetime.now()

        # Cache it
        self._context = context
        self._last_load_time = datetime.now()

        log.info(f"Loaded Telos context: profile={bool(context.profile)}, "
                f"goals={bool(context.goals)}, mission={bool(context.mission)}, "
                f"projects={len(context.project_context)}")

        return context

    def _load_and_process_file(self, file_path: Path) -> str:
        """
        Load a markdown file and process it.

        Removes:
        - HTML comments
        - Template instructions
        - Excessive blank lines
        """
        try:
            content = file_path.read_text(encoding='utf-8')

            # Remove HTML comments (<!-- ... -->)
            content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

            # Remove excessive blank lines (3+ consecutive)
            content = re.sub(r'\n{3,}', '\n\n', content)

            # Strip leading/trailing whitespace
            content = content.strip()

            return content

        except Exception as e:
            log.error(f"Error loading {file_path}: {e}")
            return ""

    def format_for_llm(
        self,
        context: TelosContext = None,
        active_project: str = None,
        include_all_projects: bool = False
    ) -> str:
        """
        Format Telos context for LLM system prompt.

        Args:
            context: TelosContext to format (loads if None)
            active_project: Name of active project to include
            include_all_projects: Include all project contexts

        Returns:
            Formatted string for system prompt
        """
        if context is None:
            context = self.load_context()

        if context.is_empty():
            return ""

        lines = []

        # Profile section
        if context.profile:
            lines.append("# PERSONAL CONTEXT")
            lines.append("")
            lines.append(context.profile)
            lines.append("")

        # Mission section
        if context.mission:
            lines.append("# YOUR MISSION")
            lines.append("")
            lines.append(context.mission)
            lines.append("")

        # Goals section
        if context.goals:
            lines.append("# CURRENT GOALS")
            lines.append("")
            lines.append(context.goals)
            lines.append("")

        # Project context
        if context.project_context:
            if active_project and active_project in context.project_context:
                # Include only active project
                lines.append(f"# PROJECT: {active_project.upper()}")
                lines.append("")
                lines.append(context.project_context[active_project])
                lines.append("")
            elif include_all_projects:
                # Include all projects
                lines.append("# ACTIVE PROJECTS")
                lines.append("")
                for project_name, project_content in context.project_context.items():
                    lines.append(f"## {project_name}")
                    lines.append(project_content)
                    lines.append("")

        result = "\n".join(lines).strip()

        if result:
            return f"\n{result}\n"

        return ""

    def create_project_context(self, project_name: str, initial_content: str = None) -> Path:
        """
        Create a new project context file.

        Args:
            project_name: Name of the project
            initial_content: Initial content (uses template if None)

        Returns:
            Path to created file
        """
        project_file = self.projects_dir / f"{project_name}.md"

        if project_file.exists():
            log.warning(f"Project context already exists: {project_name}")
            return project_file

        if initial_content is None:
            initial_content = self._get_project_template(project_name)

        project_file.write_text(initial_content)
        log.info(f"Created project context: {project_name}")

        # Invalidate cache
        self._context = None

        return project_file

    def _get_project_template(self, project_name: str) -> str:
        """Template for project-specific context"""
        return f"""# {project_name}

<!-- Project-specific context for {project_name} -->

## Overview
<!-- What is this project about? -->

This project is...

## Current Status
<!-- Where are we now? -->

- Current phase: [planning/development/testing/deployed]
- Last worked on: {datetime.now().strftime('%Y-%m-%d')}

## Technical Details
<!-- Technologies, architecture, key decisions -->

- Tech stack:
- Architecture:
- Key files:

## Important Context
<!-- Things Workshop should know when working on this project -->

- Important conventions:
- Known issues:
- Things to remember:

## Resources
<!-- Links to docs, repos, etc. -->

- Repository:
- Documentation:
- Related links:
"""

    def list_projects(self) -> List[str]:
        """List all project context files"""
        if not self.projects_dir.exists():
            return []

        return [f.stem for f in self.projects_dir.glob("*.md")]

    def get_project_path(self, project_name: str) -> Optional[Path]:
        """Get path to a project context file"""
        project_file = self.projects_dir / f"{project_name}.md"
        return project_file if project_file.exists() else None

    def auto_detect_project(self, current_dir: Path = None) -> Optional[str]:
        """
        Auto-detect active project based on current directory.

        Args:
            current_dir: Current working directory (uses cwd if None)

        Returns:
            Project name if detected, None otherwise
        """
        if current_dir is None:
            current_dir = Path.cwd()

        # Check if current directory name matches a project
        dir_name = current_dir.name.lower()
        projects = self.list_projects()

        for project in projects:
            if project.lower() == dir_name:
                log.debug(f"Auto-detected project: {project}")
                return project

        # Check parent directories (up to 3 levels)
        for parent in list(current_dir.parents)[:3]:
            parent_name = parent.name.lower()
            for project in projects:
                if project.lower() == parent_name:
                    log.debug(f"Auto-detected project from parent: {project}")
                    return project

        return None

    def reload(self):
        """Force reload all context from disk"""
        self._context = None
        self._last_load_time = None
        self.load_context(force_reload=True)
        log.info("Telos context reloaded")

    def get_stats(self) -> Dict:
        """Get statistics about Telos context"""
        context = self.load_context()

        return {
            'telos_dir': str(self.telos_dir),
            'has_profile': bool(context.profile),
            'has_goals': bool(context.goals),
            'has_mission': bool(context.mission),
            'project_count': len(context.project_context),
            'projects': list(context.project_context.keys()),
            'last_updated': context.last_updated.strftime('%Y-%m-%d %H:%M:%S'),
        }


# === Singleton instance ===
_telos_manager_instance: Optional[TelosManager] = None


def get_telos_manager(telos_dir: Path = None) -> TelosManager:
    """Get or create the global Telos manager"""
    global _telos_manager_instance

    if _telos_manager_instance is None:
        _telos_manager_instance = TelosManager(telos_dir=telos_dir)

    return _telos_manager_instance
