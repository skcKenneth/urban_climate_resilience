#!/usr/bin/env python3
"""
Test script for quick mode analysis
"""
import os
import sys
import subprocess
import glob

def test_quick_mode():
    """Test the quick mode analysis"""
    print("🧪 Testing Quick Mode Analysis")
    print("=" * 50)
    
    # Set environment variables for quick mode
    env = os.environ.copy()
    env['QUICK_MODE'] = 'true'
    env['MAX_TIME'] = '300'  # 5 minutes
    env['N_SAMPLES'] = '50'
    env['SIMULATION_DAYS'] = '90'
    env['GITHUB_ACTIONS'] = 'true'
    env['MPLBACKEND'] = 'Agg'
    
    print("Environment settings:")
    print(f"  QUICK_MODE: {env['QUICK_MODE']}")
    print(f"  MAX_TIME: {env['MAX_TIME']} seconds")
    print(f"  N_SAMPLES: {env['N_SAMPLES']}")
    print(f"  SIMULATION_DAYS: {env['SIMULATION_DAYS']}")
    
    try:
        # Run the analysis
        print("\n🚀 Starting quick mode analysis...")
        result = subprocess.run([
            sys.executable, 'main.py', '--analysis-type', 'quick', '--quick-mode'
        ], env=env, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        print("✅ Analysis completed!")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("\n📊 Output:")
            print(result.stdout[-1000:])  # Last 1000 chars
        
        if result.stderr:
            print("\n⚠️  Errors:")
            print(result.stderr[-500:])  # Last 500 chars
        
        # Check for generated files
        png_files = glob.glob("results/*.png")
        if png_files:
            print(f"\n📈 Generated {len(png_files)} PNG files:")
            for f in png_files:
                print(f"  - {f}")
        else:
            print("\n❌ No PNG files generated")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Analysis timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_quick_mode()
    sys.exit(0 if success else 1)