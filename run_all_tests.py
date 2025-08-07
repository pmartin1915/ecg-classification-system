#!/usr/bin/env python3
"""
Master Test Orchestrator for ECG Clinical System
Runs comprehensive test suite and generates detailed reports
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class ECGTestOrchestrator:
    """Orchestrates all ECG system tests"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_test_suite(self, test_name, test_file):
        """Run a specific test suite"""
        print(f"\\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run the test file
            if test_file.endswith('.py'):
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, timeout=300)
                
                duration = time.time() - start_time
                success = result.returncode == 0
                
                self.results[test_name] = {
                    'success': success,
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
                
                if success:
                    print(f"✅ {test_name} PASSED ({duration:.2f}s)")
                    self.passed_tests += 1
                else:
                    print(f"❌ {test_name} FAILED ({duration:.2f}s)")
                    print(f"Error: {result.stderr}")
                    self.failed_tests += 1
                
                self.total_tests += 1
                return success
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name} TIMEOUT (>300s)")
            self.results[test_name] = {
                'success': False,
                'duration': 300,
                'error': 'Timeout after 300 seconds'
            }
            self.failed_tests += 1
            self.total_tests += 1
            return False
            
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            self.results[test_name] = {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
            self.failed_tests += 1
            self.total_tests += 1
            return False
    
    def run_quick_verification(self):
        """Run quick system verification checks"""
        print("\\n" + "="*60)
        print("QUICK SYSTEM VERIFICATION")
        print("="*60)
        
        checks = [
            ("Main Application Import", 'python -c "import complete_user_friendly; print(\\'SUCCESS\\')"'),
            ("Models Directory", 'python -c "from pathlib import Path; assert Path(\\'data/models\\').exists()"'),
            ("Cache Directory", 'python -c "from pathlib import Path; assert Path(\\'data/cache\\').exists()"'),
            ("PTB-XL Dataset", 'python -c "from pathlib import Path; assert Path(\\'data/raw/ptbxl/ptbxl_database.csv\\').exists()"')
        ]
        
        verification_passed = 0
        for check_name, command in checks:
            try:
                result = subprocess.run(command.split(), capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"✅ {check_name}")
                    verification_passed += 1
                else:
                    print(f"❌ {check_name}: {result.stderr}")
            except Exception as e:
                print(f"💥 {check_name}: {e}")
        
        print(f"\\nQuick Verification: {verification_passed}/{len(checks)} passed")
        return verification_passed == len(checks)
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time
        
        print("\\n" + "="*80)
        print("ECG CLINICAL SYSTEM - COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        print(f"Test Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        
        print("\\n" + "-"*80)
        print("DETAILED RESULTS")
        print("-"*80)
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result.get('duration', 0)
            print(f"{status} {test_name:<30} ({duration:.2f}s)")
            
            if not result['success'] and 'error' in result:
                print(f"     Error: {result['error']}")
        
        # Generate JSON report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'success_rate': self.passed_tests/self.total_tests*100 if self.total_tests > 0 else 0
            },
            'results': self.results
        }
        
        report_file = Path('test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\\n📊 Detailed report saved to: {report_file}")
        
        # Overall assessment
        print("\\n" + "="*80)
        if self.passed_tests == self.total_tests:
            print("🎉 ALL TESTS PASSED - SYSTEM READY FOR CLINICAL USE")
            print("✅ ECG AI Detection System is fully functional")
        elif self.passed_tests >= self.total_tests * 0.8:  # 80% pass rate
            print("⚠️  MOSTLY FUNCTIONAL - MINOR ISSUES DETECTED")
            print("💡 System is operational but some components need attention")
        else:
            print("🚨 SIGNIFICANT ISSUES DETECTED - REQUIRES ATTENTION")
            print("❌ System may not be ready for clinical deployment")
        
        print("="*80)
        
        return self.passed_tests == self.total_tests
    
    def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        self.start_time = time.time()
        
        print("🚀 Starting ECG Clinical System Comprehensive Tests")
        print("⏱️  This may take 5-15 minutes depending on system performance")
        
        # Quick verification first
        if not self.run_quick_verification():
            print("⚠️  Quick verification failed, but continuing with full tests...")
        
        # Define test suite
        test_suites = [
            ("System Integration Tests", "test_system_integration.py"),
            ("Clinical Accuracy Tests", "test_clinical_accuracy.py"),
            ("Data Pipeline Tests", "test_data_pipeline.py"),
        ]
        
        # Add existing test files if they exist
        existing_tests = [
            ("MI Enhancement Tests", "test_mi_enhancement.py"),
            ("User Interface Tests", "test_user_friendly.py"),
            ("UI Integration Tests", "test_ui_integration.py"),
            ("Enhanced Explainability Tests", "test_enhanced_explainability.py")
        ]
        
        for test_name, test_file in existing_tests:
            if Path(test_file).exists():
                test_suites.append((test_name, test_file))
        
        # Run all test suites
        for test_name, test_file in test_suites:
            if Path(test_file).exists():
                self.run_test_suite(test_name, test_file)
            else:
                print(f"⚠️  Skipping {test_name} - file not found: {test_file}")
        
        # Generate final report
        return self.generate_report()

def main():
    """Main test execution"""
    print("ECG Clinical System - Automated Test Suite")
    print("==========================================")
    
    orchestrator = ECGTestOrchestrator()
    
    # Change to project directory
    os.chdir(project_root)
    
    # Run comprehensive tests
    success = orchestrator.run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()