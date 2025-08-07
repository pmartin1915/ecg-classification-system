"""
Smart ECG System Launcher (Python)
Advanced launcher with intelligent system detection and optimization
Companion to ULTIMATE_ECG_LAUNCHER.bat for power users
"""

import os
import sys
import time
import psutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

class SmartECGLauncher:
    """
    Intelligent ECG system launcher with automatic optimization
    """
    
    def __init__(self):
        self.system_info = self._detect_system()
        self.project_root = Path(__file__).parent.absolute()  # Always use absolute path
        
    def _detect_system(self) -> Dict:
        """
        Comprehensive system detection and capability analysis
        """
        # Memory detection
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # CPU detection
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 0
        
        # System classification
        if memory_gb <= 8:
            system_type = "Laptop/Mobile"
            performance = "Optimized"
            recommended_dataset = "rapid"
        elif memory_gb <= 16:
            system_type = "Standard PC"
            performance = "Good"  
            recommended_dataset = "training"
        else:
            system_type = "High-End PC"
            performance = "Excellent"
            recommended_dataset = "validation"
        
        return {
            'memory_gb': memory_gb,
            'cpu_count': cpu_count,
            'cpu_freq_mhz': cpu_freq,
            'system_type': system_type,
            'performance': performance,
            'recommended_dataset': recommended_dataset,
            'platform': platform.system(),
            'python_version': platform.python_version()
        }
    
    def show_system_banner(self):
        """
        Display smart system banner with detected capabilities
        """
        print("=" * 70)
        print("       SMART ECG SYSTEM LAUNCHER - PYTHON EDITION")
        print("=" * 70)
        print()
        
        info = self.system_info
        print(f"System Type:     {info['system_type']}")
        print(f"Performance:     {info['performance']} ({info['memory_gb']:.1f}GB RAM, {info['cpu_count']} cores)")
        print(f"Platform:        {info['platform']} | Python {info['python_version']}")
        print(f"Recommended:     {info['recommended_dataset']} dataset size")
        print()
        print("=" * 70)
        print()
    
    def quick_system_check(self) -> Dict[str, bool]:
        """
        Rapid system verification for all components
        """
        results = {}
        
        print("[CHECK] Running system verification...")
        
        # Check Python dependencies
        deps = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'plotly', 'sklearn', 'pickle']
        for dep in deps:
            try:
                __import__(dep)
                results[f'dep_{dep}'] = True
            except ImportError:
                results[f'dep_{dep}'] = False
        
        # Check data directories
        data_paths = {
            'cache': self.project_root / 'data' / 'cache',
            'models': self.project_root / 'data' / 'models', 
            'ptbxl': self.project_root / 'data' / 'raw' / 'ptbxl'
        }
        
        for name, path in data_paths.items():
            results[f'path_{name}'] = path.exists()
        
        # Check data loader
        try:
            sys.path.append(str(self.project_root))
            from app.utils.laptop_optimized_loader import test_system
            results['data_loader'] = test_system()
        except Exception:
            results['data_loader'] = False
        
        # Check main application
        try:
            import complete_user_friendly
            results['main_app'] = True
        except Exception:
            results['main_app'] = False
            
        return results
    
    def display_check_results(self, results: Dict[str, bool]):
        """
        Display system check results with recommendations
        """
        print("\n[RESULTS] System Check Summary:")
        print("-" * 50)
        
        # Group results
        deps = {k: v for k, v in results.items() if k.startswith('dep_')}
        paths = {k: v for k, v in results.items() if k.startswith('path_')}
        systems = {k: v for k, v in results.items() if not (k.startswith('dep_') or k.startswith('path_'))}
        
        # Dependencies
        print("Dependencies:")
        for dep, status in deps.items():
            dep_name = dep.replace('dep_', '')
            status_text = "[OK]" if status else "[MISSING]"
            print(f"  {status_text} {dep_name}")
        
        # Data paths
        print("\nData Directories:")
        for path, status in paths.items():
            path_name = path.replace('path_', '')
            status_text = "[OK]" if status else "[MISSING]"
            print(f"  {status_text} {path_name}")
        
        # Systems
        print("\nSystem Components:")
        for system, status in systems.items():
            status_text = "[OK]" if status else "[FAILED]"
            print(f"  {status_text} {system.replace('_', ' ')}")
        
        # Overall assessment
        total_checks = len(results)
        passed_checks = sum(results.values())
        success_rate = (passed_checks / total_checks) * 100
        
        print(f"\nOverall Status: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("[EXCELLENT] System ready for full operation!")
        elif success_rate >= 70:
            print("[GOOD] System functional with minor issues")
        else:
            print("[ATTENTION] System needs attention before use")
    
    def intelligent_data_loading_demo(self):
        """
        Demonstrate intelligent data loading based on system capabilities
        """
        print("\n[DEMO] Intelligent Data Loading")
        print("-" * 40)
        
        recommended = self.system_info['recommended_dataset']
        print(f"Recommended dataset for your {self.system_info['system_type']}: {recommended}")
        
        try:
            from app.utils.laptop_optimized_loader import quick_load
            
            print(f"\n[LOADING] Testing {recommended} dataset...")
            start_time = time.time()
            
            X, y, metadata = quick_load(recommended, verbose=True)
            
            load_time = time.time() - start_time
            
            print(f"\n[ANALYSIS] Load Performance:")
            print(f"  Actual load time: {load_time:.2f}s")
            print(f"  Performance rating: {'Excellent' if load_time < 5 else 'Good' if load_time < 15 else 'Acceptable'}")
            print(f"  Memory efficiency: {sys.getsizeof(X) / (1024*1024):.1f}MB in memory")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Data loading demo failed: {e}")
            return False
    
    def launch_streamlit_with_optimization(self, port: int = 8507):
        """
        Launch Streamlit with system-optimized settings
        """
        print(f"\n[LAUNCH] Starting optimized Streamlit server...")
        print(f"Port: {port}")
        print(f"Optimization: {self.system_info['performance']} mode")
        
        # Set environment variables based on system
        env = os.environ.copy()
        env['STREAMLIT_SERVER_HEADLESS'] = 'true'
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Memory optimization for lower-end systems
        if self.system_info['memory_gb'] <= 8:
            env['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '50'  # 50MB limit
            env['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '50'
            print("[OPTIMIZATION] Applied memory constraints for laptop")
        
        try:
            # Start Streamlit
            cmd = [
                sys.executable, '-m', 'streamlit', 'run', 
                'complete_user_friendly.py',
                '--server.port', str(port),
                '--server.headless', 'true'
            ]
            
            print(f"[COMMAND] {' '.join(cmd)}")
            print(f"[INFO] Starting server... (this may take 10-15 seconds)")
            
            # Start process
            process = subprocess.Popen(
                cmd, 
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_root
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"[SUCCESS] Server started successfully!")
                print(f"[URL] http://localhost:{port}")
                
                # Open browser
                import webbrowser
                time.sleep(2)
                webbrowser.open(f'http://localhost:{port}')
                
                print(f"[RUNNING] Server is running. Press Ctrl+C to stop.")
                
                # Keep alive
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print(f"\n[STOPPING] Shutting down server...")
                    process.terminate()
                    process.wait()
                    print(f"[STOPPED] Server stopped successfully")
            else:
                stdout, stderr = process.communicate()
                print(f"[ERROR] Server failed to start")
                if stderr:
                    print(f"Error: {stderr}")
                    
        except Exception as e:
            print(f"[ERROR] Failed to launch Streamlit: {e}")
    
    def interactive_menu(self):
        """
        Interactive menu system
        """
        while True:
            self.show_system_banner()
            
            print("SMART LAUNCHER OPTIONS:")
            print()
            print("  1. Quick System Check")
            print("  2. Intelligent Data Loading Demo") 
            print("  3. Launch Optimized Streamlit Server")
            print("  4. System Performance Analysis")
            print("  5. Data Management Tools")
            print("  6. Exit")
            print()
            
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                results = self.quick_system_check()
                self.display_check_results(results)
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                self.intelligent_data_loading_demo()
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                port = input("Port (default 8507): ").strip()
                port = int(port) if port else 8507
                self.launch_streamlit_with_optimization(port)
                
            elif choice == '4':
                self.system_performance_analysis()
                input("\nPress Enter to continue...")
                
            elif choice == '5':
                self.data_management_tools()
                input("\nPress Enter to continue...")
                
            elif choice == '6':
                print("\n[EXIT] Thanks for using Smart ECG Launcher!")
                break
                
            else:
                print("\n[ERROR] Invalid choice. Please select 1-6.")
                time.sleep(1)
    
    def system_performance_analysis(self):
        """
        Detailed system performance analysis
        """
        print("\n[ANALYSIS] System Performance Analysis")
        print("-" * 50)
        
        # Current system usage
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"Current Usage:")
        print(f"  CPU: {cpu_percent:.1f}%")
        print(f"  Memory: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
        print(f"  Available: {memory.available / (1024**3):.1f}GB free")
        
        # Recommendations
        print(f"\nRecommendations for {self.system_info['system_type']}:")
        
        if memory.percent > 80:
            print("  [WARN] High memory usage - consider closing other applications")
        
        if cpu_percent > 80:
            print("  [WARN] High CPU usage - system may be under load")
            
        if self.system_info['memory_gb'] <= 8:
            print("  [TIP] Use 'rapid' datasets (25 samples) for optimal performance")
            print("  [TIP] Close browser tabs and other applications during training")
        elif self.system_info['memory_gb'] <= 16:
            print("  [TIP] 'training' datasets (1000 samples) work well on your system")
            print("  [TIP] Full validation datasets should work without issues")
        else:
            print("  [TIP] All dataset sizes work optimally on your system")
            print("  [TIP] Consider contributing to model development")
    
    def data_management_tools(self):
        """
        Data management utilities
        """
        print("\n[TOOLS] Data Management Tools")
        print("-" * 40)
        
        try:
            from app.utils.laptop_optimized_loader import show_menu
            show_menu()
        except Exception as e:
            print(f"[ERROR] Could not load data management tools: {e}")

def main():
    """
    Main launcher entry point
    """
    launcher = SmartECGLauncher()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'check':
            launcher.show_system_banner()
            results = launcher.quick_system_check()
            launcher.display_check_results(results)
        elif command == 'demo':
            launcher.show_system_banner()
            launcher.intelligent_data_loading_demo()
        elif command == 'launch':
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8507
            launcher.show_system_banner()
            launcher.launch_streamlit_with_optimization(port)
        elif command == 'analyze':
            launcher.show_system_banner()
            launcher.system_performance_analysis()
        else:
            print(f"Unknown command: {command}")
            print("Available: check, demo, launch, analyze")
    else:
        # Interactive mode
        launcher.interactive_menu()

if __name__ == "__main__":
    main()