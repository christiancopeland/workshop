#!/usr/bin/env python3
"""
Context Awareness Demo

Shows how the context system works without needing full Workshop.
Run this to see what Workshop "sees" about your environment.

Usage:
    python demo_context.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import context
sys.path.insert(0, str(Path(__file__).parent))

from context import ContextAwareness, get_context_manager


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_basic_context():
    """Show basic context detection"""
    print_section("BASIC CONTEXT DETECTION")
    
    # Create context manager with example project path
    project_paths = [
        Path.home() / "projects",
        Path.home() / "Documents",
        Path.home() / "Arduino",
    ]
    
    # Filter to only existing paths
    project_paths = [p for p in project_paths if p.exists()]
    
    if not project_paths:
        print("⚠️  No project paths found. Creating dummy path for demo...")
        project_paths = [Path.home()]
    
    context_mgr = ContextAwareness(
        project_paths=project_paths,
        indexable_extensions={'.py', '.ino', '.cpp', '.h', '.js', '.md', '.txt'}
    )
    
    # Get context
    print("\nGetting current context...")
    context = context_mgr.get_context()
    
    print(f"\n{context}")


def demo_device_detection():
    """Show USB device detection"""
    print_section("USB DEVICE DETECTION")
    
    context_mgr = get_context_manager()
    context = context_mgr.get_context()
    
    if not context.connected_devices:
        print("ℹ️  No USB devices detected.")
        print("\nTip: Connect an Arduino or ESP32 board and run again!")
        return
    
    print(f"\nFound {len(context.connected_devices)} USB device(s):\n")
    
    for i, device in enumerate(context.connected_devices, 1):
        print(f"{i}. {device.description}")
        print(f"   Port: {device.port}")
        
        if device.is_arduino:
            print(f"   ✓ Detected as Arduino/ESP32 board")
        
        if device.vid:
            print(f"   VID/PID: {device.vid}:{device.pid}")
        
        print()


def demo_recent_files():
    """Show recent file detection"""
    print_section("RECENT FILE ACTIVITY")
    
    context_mgr = get_context_manager()
    context = context_mgr.get_context()
    
    if not context.recent_files:
        print("ℹ️  No recent file modifications in the last hour.")
        print("\nTip: Edit a .py or .ino file and run again!")
        return
    
    print(f"\nFiles modified in last hour:\n")
    
    for i, file in enumerate(context.recent_files[:10], 1):
        age_min = int(file.age_seconds / 60)
        age_sec = int(file.age_seconds % 60)
        
        print(f"{i}. {file.path.name}")
        print(f"   Path: {file.path.parent}")
        print(f"   Modified: {age_min}m {age_sec}s ago")
        print(f"   Size: {file.size:,} bytes")
        print()


def demo_llm_context_formatting():
    """Show how context is formatted for LLM injection"""
    print_section("LLM CONTEXT FORMATTING")
    
    context_mgr = get_context_manager()
    context = context_mgr.get_context()
    
    formatted = context_mgr.format_context_for_llm(context)
    
    if not formatted:
        print("ℹ️  No relevant context to inject.")
        print("\nTip: Connect a board and edit some files to see context!")
        return
    
    print("\nThis is what gets injected into LLM prompts:\n")
    print("─" * 60)
    print(formatted)
    print("─" * 60)


def demo_smart_suggestions():
    """Show smart suggestion logic"""
    print_section("SMART SUGGESTIONS")
    
    context_mgr = get_context_manager()
    context = context_mgr.get_context()
    
    print("\nAnalyzing context for suggestions...\n")
    
    # Check if should suggest compile
    should_compile = context_mgr.should_suggest_compile(context)
    if should_compile:
        print("✓ Should suggest: COMPILE")
        print("  Reason: Arduino board connected + recent .ino modifications")
    else:
        print("✗ No compile suggestion")
        if not any(d.is_arduino for d in context.connected_devices):
            print("  Reason: No Arduino board detected")
        elif not any(f.path.suffix == '.ino' for f in context.recent_files):
            print("  Reason: No recent .ino file modifications")
    
    print()
    
    # Check if should suggest upload
    should_upload = context_mgr.should_suggest_upload(context)
    if should_upload:
        print("✓ Should suggest: UPLOAD")
        print("  Reason: Board connected + very recent sketch changes (<1 min)")
    else:
        print("✗ No upload suggestion")
        if should_compile:
            print("  Reason: Sketch changes too old (>1 min)")
    
    print()


def demo_project_detection():
    """Show project detection from file paths"""
    print_section("PROJECT DETECTION")
    
    context_mgr = get_context_manager()
    context = context_mgr.get_context()
    
    if not context.recent_files:
        print("ℹ️  No recent files to analyze.")
        return
    
    print("\nDetecting which projects files belong to:\n")
    
    # Group files by project
    by_project = {}
    for file in context.recent_files[:10]:
        project = context_mgr.get_project_for_file(file.path)
        if project:
            if project not in by_project:
                by_project[project] = []
            by_project[project].append(file.path.name)
    
    if not by_project:
        print("ℹ️  Files not in any configured project path.")
        return
    
    for project, files in by_project.items():
        print(f"📁 {project}")
        for file in files:
            print(f"   • {file}")
        print()


def main():
    """Run all demos"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           WORKSHOP CONTEXT AWARENESS DEMO                    ║
║                                                              ║
║  This shows what Workshop "sees" about your dev environment  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Check dependencies
        try:
            import serial.tools.list_ports
        except ImportError:
            print("\n⚠️  Warning: 'pyserial' not installed.")
            print("   USB device detection will not work.")
            print("   Install with: pip install pyserial\n")
        
        # Run demos
        demo_basic_context()
        demo_device_detection()
        demo_recent_files()
        demo_llm_context_formatting()
        demo_smart_suggestions()
        demo_project_detection()
        
        # Summary
        print_section("WHAT WORKSHOP DOES WITH THIS")
        print("""
When you ask Workshop to do something, it:

1. Captures current context (devices, files, window)
2. Determines if context is relevant to your query
3. Injects context into the LLM prompt automatically
4. Uses context to make smart suggestions

Examples:

Query: "compile this"
→ Sees you have ESP32 connected
→ Sees you modified battery_guardian.ino 2 min ago
→ Auto-injects: "Connected: ESP32, Recent: battery_guardian.ino"
→ LLM understands what to compile without you specifying

Query: "what am i working on?"
→ Sees recent file modifications
→ Groups by project
→ Shows: "Working on battery-guardian (3 files modified in last hour)"

Query: "upload firmware"
→ Sees Arduino Nano on /dev/ttyACM0
→ Auto-uses that port for upload
→ No need to specify port manually

This makes Workshop feel like it "knows" what you're doing!
        """)
        
        print_section("NEXT STEPS")
        print("""
1. Run the tests: python test_context.py
2. Follow CONTEXT_INTEGRATION_GUIDE.txt to add to Workshop
3. Try it out: python main.py --mode text
4. Test commands:
   - "list connected devices"
   - "get dev context"
   - "compile [sketch_name]"
        """)
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()