"""
Unit tests for Telos Personal Context System
Tests TelosManager and Telos skill tools
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Import Telos components
from telos_manager import TelosManager, TelosContext


class TestTelosManager:
    """Test TelosManager class"""

    def __init__(self):
        self.temp_dir = None
        self.telos_mgr = None

    def setup(self):
        """Create temporary Telos directory for testing"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.telos_mgr = TelosManager(telos_dir=self.temp_dir)
        print(f"✓ Created test Telos dir: {self.temp_dir}")

    def teardown(self):
        """Clean up temporary directory"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"✓ Cleaned up test dir")

    def test_initialization(self):
        """Test TelosManager initializes correctly"""
        assert self.telos_mgr.telos_dir == self.temp_dir
        assert self.telos_mgr.projects_dir == self.temp_dir / "projects"
        assert self.telos_mgr.telos_dir.exists()
        assert self.telos_mgr.projects_dir.exists()
        print("✓ TelosManager initialization")

    def test_template_creation(self):
        """Test that template files are created"""
        profile_path = self.temp_dir / "profile.md"
        goals_path = self.temp_dir / "goals.md"
        mission_path = self.temp_dir / "mission.md"

        assert profile_path.exists(), "profile.md should be created"
        assert goals_path.exists(), "goals.md should be created"
        assert mission_path.exists(), "mission.md should be created"

        # Check that templates have content
        assert len(profile_path.read_text()) > 100
        assert len(goals_path.read_text()) > 100
        assert len(mission_path.read_text()) > 100

        print("✓ Template file creation")

    def test_load_context(self):
        """Test loading context from files"""
        context = self.telos_mgr.load_context()

        assert isinstance(context, TelosContext)
        assert context.profile  # Should have content from template
        assert context.goals
        assert context.mission
        assert isinstance(context.project_context, dict)

        print("✓ Context loading")

    def test_context_caching(self):
        """Test that context is cached"""
        # First load
        context1 = self.telos_mgr.load_context()
        time1 = self.telos_mgr._last_load_time

        # Second load (should use cache)
        context2 = self.telos_mgr.load_context()
        time2 = self.telos_mgr._last_load_time

        assert context1 is context2, "Should return same cached object"
        assert time1 == time2, "Cache timestamp should not change"

        print("✓ Context caching")

    def test_force_reload(self):
        """Test force reload bypasses cache"""
        context1 = self.telos_mgr.load_context()
        time1 = self.telos_mgr._last_load_time

        # Modify profile file
        profile_path = self.temp_dir / "profile.md"
        profile_path.write_text("# Updated Profile\nNew content")

        # Force reload
        context2 = self.telos_mgr.load_context(force_reload=True)
        time2 = self.telos_mgr._last_load_time

        assert context1 is not context2, "Should be new object"
        assert time1 != time2, "Cache timestamp should update"
        assert "New content" in context2.profile

        print("✓ Force reload")

    def test_html_comment_removal(self):
        """Test that HTML comments are removed from loaded content"""
        profile_path = self.temp_dir / "profile.md"
        profile_path.write_text("# Profile\n<!-- This is a comment -->\nActual content\n<!-- Another comment -->")

        context = self.telos_mgr.load_context(force_reload=True)

        assert "<!-- " not in context.profile, "HTML comments should be removed"
        assert "Actual content" in context.profile, "Real content should remain"

        print("✓ HTML comment removal")

    def test_format_for_llm(self):
        """Test formatting context for LLM prompt"""
        formatted = self.telos_mgr.format_for_llm()

        assert isinstance(formatted, str)
        assert len(formatted) > 0, "Should have content"
        assert "PERSONAL CONTEXT" in formatted or "# " in formatted

        print("✓ LLM formatting")

    def test_create_project_context(self):
        """Test creating a new project context file"""
        project_name = "test_project"
        project_path = self.telos_mgr.create_project_context(project_name)

        assert project_path.exists(), "Project file should be created"
        assert project_path.name == f"{project_name}.md"
        assert project_path.parent == self.telos_mgr.projects_dir

        content = project_path.read_text()
        assert project_name in content, "Project name should be in content"

        print("✓ Project context creation")

    def test_list_projects(self):
        """Test listing project contexts"""
        # Create some projects
        self.telos_mgr.create_project_context("project1")
        self.telos_mgr.create_project_context("project2")

        projects = self.telos_mgr.list_projects()

        assert len(projects) >= 2, f"Expected at least 2 projects, got {len(projects)}"
        assert "project1" in projects
        assert "project2" in projects

        print("✓ List projects")

    def test_get_project_path(self):
        """Test getting path to project file"""
        project_name = "myproject"
        created_path = self.telos_mgr.create_project_context(project_name)

        retrieved_path = self.telos_mgr.get_project_path(project_name)

        assert retrieved_path == created_path
        assert retrieved_path.exists()

        # Test non-existent project
        none_path = self.telos_mgr.get_project_path("nonexistent")
        assert none_path is None

        print("✓ Get project path")

    def test_auto_detect_project(self):
        """Test auto-detecting project from directory"""
        # Create a project called "workshop"
        self.telos_mgr.create_project_context("workshop")

        # Simulate being in a directory named "workshop" (use dynamic path)
        workshop_dir = Path(__file__).parent.parent  # Points to workshop dir
        detected = self.telos_mgr.auto_detect_project(current_dir=workshop_dir)

        assert detected == "workshop", "Should detect workshop project"

        print("✓ Auto-detect project")

    def test_get_stats(self):
        """Test getting Telos statistics"""
        # Clear any existing projects first
        self._context = None

        self.telos_mgr.create_project_context("proj1")

        stats = self.telos_mgr.get_stats()

        assert isinstance(stats, dict)
        assert 'telos_dir' in stats
        assert 'has_profile' in stats
        assert 'has_goals' in stats
        assert 'has_mission' in stats
        assert 'project_count' in stats
        assert 'projects' in stats

        assert stats['has_profile'] is True
        assert stats['project_count'] >= 1, f"Expected at least 1 project, got {stats['project_count']}"
        assert 'proj1' in stats['projects']

        print("✓ Get stats")

    def test_reload_method(self):
        """Test reload() method"""
        # Load initial context
        self.telos_mgr.load_context()

        # Modify a file
        profile_path = self.temp_dir / "profile.md"
        profile_path.write_text("# Modified")

        # Reload
        self.telos_mgr.reload()

        # Check that new content is loaded
        context = self.telos_mgr.load_context()
        assert "# Modified" in context.profile

        print("✓ Reload method")

    def test_project_context_in_format(self):
        """Test that active project is included in formatted output"""
        # Create project
        self.telos_mgr.create_project_context("workshop")

        # Format with active project
        formatted = self.telos_mgr.format_for_llm(active_project="workshop")

        assert "workshop" in formatted.lower(), "Project name should appear"
        assert "PROJECT" in formatted, "Should have project section"

        print("✓ Project context in format")

    def run_all(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("Testing TelosManager")
        print("="*60 + "\n")

        try:
            self.setup()

            self.test_initialization()
            self.test_template_creation()
            self.test_load_context()
            self.test_context_caching()
            self.test_force_reload()
            self.test_html_comment_removal()
            self.test_format_for_llm()
            self.test_create_project_context()
            self.test_list_projects()
            self.test_get_project_path()
            self.test_auto_detect_project()
            self.test_get_stats()
            self.test_reload_method()
            self.test_project_context_in_format()

            print("\n✅ All TelosManager tests passed!")
            return True

        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            return False

        finally:
            self.teardown()


class TestTelosSkillTools:
    """Test Telos skill tools"""

    def __init__(self):
        self.temp_dir = None
        self.telos_mgr = None

    def setup(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.telos_mgr = TelosManager(telos_dir=self.temp_dir)
        print(f"✓ Created test environment")

    def teardown(self):
        """Cleanup"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    async def test_reload_telos_tool(self):
        """Test reload_telos tool"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent / ".workshop"))
        from Skills.Telos.tools.reload_telos import reload_telos

        deps = {"telos_manager": self.telos_mgr}
        result = await reload_telos(_deps=deps)

        assert "✓ Telos context reloaded" in result
        assert "Profile" in result
        assert "Goals" in result

        print("✓ reload_telos tool")

    async def test_show_telos_stats_tool(self):
        """Test show_telos_stats tool"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent / ".workshop"))
        from Skills.Telos.tools.show_telos_stats import show_telos_stats

        deps = {"telos_manager": self.telos_mgr}
        result = await show_telos_stats(_deps=deps)

        assert "Telos Personal Context" in result
        assert "Location:" in result

        print("✓ show_telos_stats tool")

    async def test_list_projects_tool(self):
        """Test list_projects tool"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent / ".workshop"))
        from Skills.Telos.tools.list_projects import list_projects

        # Create a project first
        self.telos_mgr.create_project_context("test")

        deps = {"telos_manager": self.telos_mgr}
        result = await list_projects(_deps=deps)

        assert "test" in result
        assert "Project contexts" in result

        print("✓ list_projects tool")

    async def test_create_project_tool(self):
        """Test create_project tool"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent / ".workshop"))
        from Skills.Telos.tools.create_project import create_project

        deps = {"telos_manager": self.telos_mgr}
        result = await create_project("new_project", _deps=deps)

        assert "✓ Created project context" in result
        assert "new_project" in result

        # Verify it was actually created
        projects = self.telos_mgr.list_projects()
        assert "new_project" in projects

        print("✓ create_project tool")

    async def test_edit_profile_tool(self):
        """Test edit_profile tool"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent / ".workshop"))
        from Skills.Telos.tools.edit_profile import edit_profile

        deps = {"telos_manager": self.telos_mgr}
        result = await edit_profile(_deps=deps)

        assert "profile.md" in result
        assert "Opening" in result or "Error" not in result

        print("✓ edit_profile tool")

    async def run_all_async(self):
        """Run all async tests"""
        print("\n" + "="*60)
        print("Testing Telos Skill Tools")
        print("="*60 + "\n")

        try:
            self.setup()

            await self.test_reload_telos_tool()
            await self.test_show_telos_stats_tool()
            await self.test_list_projects_tool()
            await self.test_create_project_tool()
            await self.test_edit_profile_tool()

            print("\n✅ All Telos skill tool tests passed!")
            return True

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.teardown()


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*20 + "TELOS SYSTEM TESTS")
    print("="*70)

    # Test TelosManager
    manager_tests = TestTelosManager()
    manager_passed = manager_tests.run_all()

    # Test Telos skill tools
    tool_tests = TestTelosSkillTools()
    tools_passed = asyncio.run(tool_tests.run_all_async())

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"TelosManager: {'✅ PASS' if manager_passed else '❌ FAIL'}")
    print(f"Telos Tools:  {'✅ PASS' if tools_passed else '❌ FAIL'}")
    print("="*70 + "\n")

    if manager_passed and tools_passed:
        print("🎉 All Telos tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
