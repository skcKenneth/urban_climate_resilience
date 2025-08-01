#!/usr/bin/env python3
"""
Test script for full mode analysis
"""
import os
import sys
import subprocess
import glob

def test_full_mode():
    """Test the full mode analysis"""
    print("🧪 Testing Full Mode Analysis")
    print("=" * 50)
    
    # Set environment variables for full mode
    env = os.environ.copy()
    env['QUICK_MODE'] = 'false'
    env['MAX_TIME'] = '600'  # 10 minutes for testing
    env['N_SAMPLES'] = '100'  # Reduced for testing
    env['SIMULATION_DAYS'] = '180'  # Reduced for testing
    env['GITHUB_ACTIONS'] = 'true'
    env['MPLBACKEND'] = 'Agg'
    
    print("Environment settings:")
    print(f"  QUICK_MODE: {env['QUICK_MODE']}")
    print(f"  MAX_TIME: {env['MAX_TIME']} seconds")
    print(f"  N_SAMPLES: {env['N_SAMPLES']}")
    print(f"  SIMULATION_DAYS: {env['SIMULATION_DAYS']}")
    
    try:
        # Run the analysis
        print("\n🚀 Starting full mode analysis...")
        result = subprocess.run([
            sys.executable, 'main.py', '--analysis-type', 'full'
        ], env=env, capture_output=True, text=True, timeout=1200)  # 20 minute timeout
        
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
        json_files = glob.glob("results/*.json")
        csv_files = glob.glob("results/*.csv")
        
        print(f"\n📈 Generated files:")
        print(f"  - {len(png_files)} PNG files")
        print(f"  - {len(json_files)} JSON files")
        print(f"  - {len(csv_files)} CSV files")
        
        if png_files:
            print("\n📊 PNG files:")
            for f in sorted(png_files):
                print(f"  - {f}")
        
        # Expect at least 3 PNG files (one per scenario)
        expected_scenarios = ['baseline', 'heatwave', 'extreme']
        missing_scenarios = []
        for scenario in expected_scenarios:
            if not any(scenario in f for f in png_files):
                missing_scenarios.append(scenario)
        
        if missing_scenarios:
            print(f"\n⚠️  Missing visualizations for: {', '.join(missing_scenarios)}")
        
        return result.returncode == 0 and len(missing_scenarios) == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Analysis timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_full_mode()
    sys.exit(0 if success else 1)