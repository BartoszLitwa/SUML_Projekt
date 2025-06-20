#!/usr/bin/env python3
"""
Test runner to execute all model tests in sequence
"""
import sys
import os

# Add project paths
sys.path.append('Model')
sys.path.append('tests')

# Import all test modules
from test_imports import test_imports
from test_model_training import test_model_training
from test_model_accuracy import test_model_accuracy
from test_model_consistency import test_model_consistency
from test_model_validation import test_model_validation


def run_all_tests():
    """Run all tests in sequence"""
    print("🚀 Running comprehensive model test suite...")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Training", test_model_training),
        ("Model Accuracy", test_model_accuracy),
        ("Model Consistency", test_model_consistency),
        ("Model Validation", test_model_validation),
    ]
    
    results = []
    failed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"🧪 Running {test_name}...")
        print("=" * 60)
        
        try:
            success = test_func()
            if success:
                print(f"✅ {test_name} PASSED")
                results.append((test_name, "PASSED"))
            else:
                print(f"❌ {test_name} FAILED")
                results.append((test_name, "FAILED"))
                failed_tests += 1
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
            results.append((test_name, "ERROR"))
            failed_tests += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 60)
    
    for test_name, status in results:
        status_emoji = "✅" if status == "PASSED" else "❌"
        print(f"{status_emoji} {test_name}: {status}")
    
    print(f"\nTotal tests: {len(tests)}")
    print(f"Passed: {len(tests) - failed_tests}")
    print(f"Failed/Errors: {failed_tests}")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Model is ready for deployment")
        return True
    else:
        print(f"\n💥 {failed_tests} TESTS FAILED!")
        print("❌ Model needs attention before deployment")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1) 