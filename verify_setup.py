#!/usr/bin/env python3
"""
Verify the setup for Urban Climate Resilience System
"""
import sys
import os
import importlib
import subprocess

def check_environment():
    """Check environment variables and settings"""
    print("🔍 Checking environment...")
    print(f"  Python: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"  Platform: {sys.platform}")
    
    # GitHub Actions specific
    if os.environ.get('GITHUB_ACTIONS'):
        print("  ✓ Running in GitHub Actions")
        print(f"  Runner OS: {os.environ.get('RUNNER_OS', 'Unknown')}")
    
    return True

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required = {
        'numpy': '1.24.0',
        'scipy': '1.10.0',
        'matplotlib': '3.6.0',
        'networkx': '3.0',
        'pandas': '1.5.0',
        'seaborn': '0.12.0',
        'sklearn': '1.2.0',
        'joblib': '1.2.0'
    }
    
    all_ok = True
    for module, min_version in required.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {module}: {version}")
        except ImportError:
            print(f"  ✗ {module}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_local_modules():
    """Check if local modules can be imported"""
    print("\n🔍 Checking local modules...")
    
    modules = [
        'models',
        'models.coupled_system',
        'models.optimal_control',
        'models.epidemiological_model',
        'models.network_model',
        'analysis',
        'analysis.stability_analysis',
        'analysis.sensitivity_analysis',
        'analysis.control_analysis',
        'utils',
        'utils.parameters',
        'utils.visualization',
        'utils.data_generator'
    ]
    
    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            all_ok = False
    
    return all_ok

def check_files():
    """Check if required files exist"""
    print("\n🔍 Checking required files...")
    
    files = [
        'main.py',
        'requirements.txt',
        'test_quick_run.py',
        '.github/workflows/climate_analysis.yml'
    ]
    
    all_ok = True
    for file in files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}: NOT FOUND")
            all_ok = False
    
    return all_ok

def run_quick_test():
    """Run a minimal test"""
    print("\n🧪 Running minimal test...")
    
    try:
        # Set minimal environment
        env = os.environ.copy()
        env['QUICK_MODE'] = 'true'
        env['SIMULATION_DAYS'] = '7'
        env['N_SAMPLES'] = '10'
        env['MPLBACKEND'] = 'Agg'
        
        # Run a very quick test
        result = subprocess.run(
            [sys.executable, '-c', """
import matplotlib
matplotlib.use('Agg')
from utils.parameters import ModelParameters
params = ModelParameters()
print(f'Test successful! Population: {params.N}')
"""],
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("  ✓ Minimal test passed")
            if result.stdout:
                print(f"    {result.stdout.strip()}")
            return True
        else:
            print("  ✗ Minimal test failed")
            if result.stderr:
                print(f"    Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Test error: {e}")
        return False

def main():
    """Main verification routine"""
    print("🚀 Urban Climate Resilience System - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies),
        ("Local Modules", check_local_modules),
        ("Required Files", check_files),
        ("Quick Test", run_quick_test)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All checks passed! The system is ready to run.")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())