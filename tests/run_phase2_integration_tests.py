"""
Integration Test Runner
Runs all Phase 2 integration tests in sequence.
"""

import sys
import time


def print_header(title):
    """Print test header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_test(test_name, test_module):
    """Run a single integration test."""
    print_header(f"Running: {test_name}")
    
    print(f"About to run: {test_module}")
    print("\nPress ENTER to start, or 's' to skip...")
    choice = input().strip().lower()
    
    if choice == 's':
        print("⏭️  Skipped\n")
        return False
    
    print()
    
    try:
        # Import and run test
        if test_module == "integration_test_1_vad":
            from integration_test_1_vad import test_vad_detection
            test_vad_detection()
        elif test_module == "integration_test_2_segment":
            from integration_test_2_segment import test_segment_capture
            test_segment_capture()
        elif test_module == "integration_test_3_timeout":
            from integration_test_3_timeout import test_timeout
            test_timeout()
        elif test_module == "integration_test_4_wake":
            from integration_test_4_wake import test_wake_word
            test_wake_word()
        elif test_module == "integration_test_5_interrupt":
            from integration_test_5_interrupt import test_interruption
            test_interruption()
        elif test_module == "integration_test_6_full":
            from integration_test_6_full import test_full_flow
            test_full_flow()
        
        print(f"\n✅ {test_name} complete!\n")
        return True
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  {test_name} stopped by user\n")
        return False
    except Exception as e:
        print(f"\n\n❌ {test_name} failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print_header("WORKSHOP PHASE 2: INTEGRATION TEST SUITE")
    
    print("This suite runs 6 integration tests that require microphone input.")
    print("\nTests:")
    print("  1. Basic VAD Detection (5 min)")
    print("  2. Speech Segment Capture (5 min)")
    print("  3. Timeout Detection (3 min)")
    print("  4. Wake Word Detection (5 min)")
    print("  5. Interruption Detection (5 min)")
    print("  6. Full Conversation Flow (10 min)")
    print("\nTotal estimated time: ~35 minutes")
    print("\nYou can skip any test by pressing 's' when prompted.")
    print("\nReady to begin?")
    input("Press ENTER to start...")
    
    # Define tests
    tests = [
        ("Test 1: VAD Detection", "phase2_integration_test_1_vad"),
        ("Test 2: Segment Capture", "phase2_integration_test_2_segment"),
        ("Test 3: Timeout Detection", "phase2_integration_test_3_timeout"),
        ("Test 4: Wake Word Detection", "phase2_integration_test_4_wake"),
        ("Test 5: Interruption Detection", "phase2_integration_test_5_interrupt"),
        ("Test 6: Full Conversation Flow", "phase2_integration_test_6_full"),
    ]
    
    results = []
    
    # Run each test
    for test_name, test_module in tests:
        result = run_test(test_name, test_module)
        results.append((test_name, result))
        
        # Brief pause between tests
        if result:
            print("\nNext test in 3 seconds...")
            time.sleep(3)
    
    # Summary
    print_header("TEST SUITE COMPLETE - SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests completed")
    print()
    
    for test_name, result in results:
        status = "✅ PASS" if result else "⏭️  SKIP"
        print(f"  {status}  {test_name}")
    
    print("\n" + "="*70)
    
    if passed == total:
        print("🎉 ALL TESTS COMPLETE!")
        print("\nPhase 2 integration validated!")
        print("Ready to proceed to Phase 3! 🚀")
    elif passed > 0:
        print(f"✅ {passed} tests completed")
        print("\nReview any skipped tests and run individually if needed.")
    else:
        print("No tests completed")
        print("\nReview test failures and try running individually.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite stopped by user")
        sys.exit(0)