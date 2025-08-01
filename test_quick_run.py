#!/usr/bin/env python3
"""
Test script for quick mode analysis with enhanced error handling
"""
import os
import sys
import subprocess
import glob
import traceback
import importlib
import json
from pathlib import Path

def verify_imports():
    """Verify all required modules can be imported"""
    print("🔍 Verifying imports...")
    required_modules = [
        'numpy', 'scipy', 'matplotlib', 'networkx', 'pandas',
        'seaborn', 'sklearn', 'joblib'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {module}: {version}")
        except ImportError as e:
            print(f"  ✗ {module}: MISSING - {e}")
            missing_modules.append(module)
    
    # Check local modules
    print("\n🔍 Verifying local modules...")
    local_modules = [
        'models.coupled_system',
        'models.optimal_control',
        'analysis.stability_analysis',
        'analysis.sensitivity_analysis',
        'utils.parameters',
        'utils.visualization',
        'utils.data_generator'
    ]
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            missing_modules.append(module)
    
    return len(missing_modules) == 0

def create_test_directories():
    """Create necessary directories"""
    dirs = ['results', 'logs', 'temp']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("📁 Created test directories")

def test_quick_mode():
    """Test the quick mode analysis"""
    print("🧪 Testing Quick Mode Analysis")
    print("=" * 50)
    
    # Verify imports first
    if not verify_imports():
        print("\n❌ Import verification failed!")
        return False
    
    # Create directories
    create_test_directories()
    
    # Set environment variables for quick mode
    env = os.environ.copy()
    env['QUICK_MODE'] = 'true'
    env['MAX_TIME'] = '300'  # 5 minutes
    env['N_SAMPLES'] = '50'
    env['SIMULATION_DAYS'] = '90'
    env['GITHUB_ACTIONS'] = 'true'
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONPATH'] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    
    print("\nEnvironment settings:")
    print(f"  QUICK_MODE: {env['QUICK_MODE']}")
    print(f"  MAX_TIME: {env['MAX_TIME']} seconds")
    print(f"  N_SAMPLES: {env['N_SAMPLES']}")
    print(f"  SIMULATION_DAYS: {env['SIMULATION_DAYS']}")
    print(f"  PYTHONPATH: {env['PYTHONPATH']}")
    
    try:
        # Test if main.py exists
        if not Path('main.py').exists():
            print("❌ main.py not found!")
            return False
        
        # Run the analysis
        print("\n🚀 Starting quick mode analysis...")
        cmd = [sys.executable, 'main.py', '--analysis-type', 'quick', '--quick-mode']
        
        # Log command
        print(f"Command: {' '.join(cmd)}")
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"  > {line.rstrip()}")
                output_lines.append(line)
        
        process.wait()
        return_code = process.returncode
        
        print(f"\n✅ Analysis completed with return code: {return_code}")
        
        # Check for generated files
        png_files = list(Path('results').glob('**/*.png'))
        json_files = list(Path('results').glob('**/*.json'))
        
        print(f"\n📊 Generated files:")
        print(f"  - PNG files: {len(png_files)}")
        for f in png_files[:5]:  # Show first 5
            print(f"    • {f}")
        if len(png_files) > 5:
            print(f"    • ... and {len(png_files) - 5} more")
        
        print(f"  - JSON files: {len(json_files)}")
        for f in json_files[:5]:  # Show first 5
            print(f"    • {f}")
        
        # Save test results
        test_results = {
            'return_code': return_code,
            'png_files': len(png_files),
            'json_files': len(json_files),
            'output_lines': len(output_lines)
        }
        
        with open('test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Check success criteria
        success = return_code == 0 and len(png_files) > 0
        
        if success:
            print("\n✅ Test PASSED!")
        else:
            print("\n❌ Test FAILED!")
            if return_code != 0:
                print(f"  - Non-zero return code: {return_code}")
            if len(png_files) == 0:
                print("  - No PNG files generated")
        
        return success
        
    except subprocess.TimeoutExpired:
        print("⏰ Analysis timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

def main():
    """Main entry point"""
    print("🏃 Urban Climate Resilience System - Test Runner")
    print("=" * 60)
    
    # Show Python info
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in directory: {len(list(Path('.').glob('*.py')))} Python files")
    
    # Run test
    success = test_quick_mode()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()