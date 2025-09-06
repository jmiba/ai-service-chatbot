#!/usr/bin/env python3
"""
Quick Test Runner - runs all tests in fast mode (no API calls)
"""

import sys
import os
sys.path.append('.')

def run_quick_tests():
    """Run all tests in quick mode"""
    print("🏃‍♂️ Quick Test Suite - AI Service Chatbot")
    print("=" * 60)
    print("Running tests in fast mode (no slow API calls)")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Implementation Status (fast - no API calls)
    print("\n1️⃣ Testing Filter Implementation Status...")
    try:
        import tests.test_implementation_status
        print("✅ Filter implementation test passed")
        test_results.append(("Implementation Status", "✅ PASSED"))
    except Exception as e:
        print(f"❌ Filter implementation test failed: {e}")
        test_results.append(("Implementation Status", "❌ FAILED"))
    
    # Test 2: Filter Logic (fast - no API calls)
    print("\n2️⃣ Testing Filter Logic...")
    try:
        import tests.test_filters
        print("✅ Filter logic test passed")
        test_results.append(("Filter Logic", "✅ PASSED"))
    except Exception as e:
        print(f"❌ Filter logic test failed: {e}")
        test_results.append(("Filter Logic", "❌ FAILED"))
    
    # Test 3: Session ID (fast - no API calls)
    print("\n3️⃣ Testing Session Management...")
    try:
        import tests.test_session_id
        print("✅ Session management test passed")
        test_results.append(("Session Management", "✅ PASSED"))
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        test_results.append(("Session Management", "❌ FAILED"))
    
    # Test 4: Streamlit Simulation (fast - no API calls)
    print("\n4️⃣ Testing Streamlit Simulation...")
    try:
        import tests.test_streamlit_sim
        print("✅ Streamlit simulation test passed")
        test_results.append(("Streamlit Simulation", "✅ PASSED"))
    except Exception as e:
        print(f"❌ Streamlit simulation test failed: {e}")
        test_results.append(("Streamlit Simulation", "❌ FAILED"))
    
    # Test 5: Reasoning Latency (quick mode - no API calls)
    print("\n5️⃣ Testing Reasoning Latency (quick mode)...")
    try:
        from tests.test_reasoning_latency import test_reasoning_effort_latency
        test_reasoning_effort_latency(run_api_tests=False)
        test_results.append(("Reasoning Latency", "✅ PASSED (quick mode)"))
    except Exception as e:
        print(f"❌ Reasoning latency test failed: {e}")
        test_results.append(("Reasoning Latency", "❌ FAILED"))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in test_results:
        print(f"{result:20} {test_name}")
        if "✅" in result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All quick tests passed!")
    else:
        print("⚠️  Some tests failed - check details above")
    
    print("\n💡 To run slow API tests:")
    print("   python3 tests/test_reasoning_latency.py --run-api-tests")
    print("   python3 tests/test_temperature_detailed.py --run-api-tests")
    print("   python3 tests/test_advanced_parameters.py --run-api-tests")
    
    return passed == len(test_results)

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)
