"""
Complete automated execution with GitHub Actions support
"""
import os
import sys
from debug_runner import run_automated_analysis
from config_manager import SmartConfig

def main():
    print("="*60)
    print("AUTOMATED CLIMATE-SOCIAL RESILIENCE ANALYSIS")
    
    # Check if running in GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        print("🤖 Running in GitHub Actions environment")
        # Use conservative settings for CI
        settings = {
            'n_samples_sensitivity': 100,
            'n_samples_mc': 200,
            'simulation_time': 365,
            'quick_mode': False,
            'max_optimization_time': 1200  # 20 minutes
        }
    else:
        print("💻 Running locally")
        config = SmartConfig()
        settings = config.get_optimal_settings()
        print(f"System Memory: {config.system_memory:.1f} GB")
        print(f"CPU Cores: {config.cpu_count}")
    
    print("="*60)
    
    print(f"Selected Mode: {'Quick' if settings['quick_mode'] else 'Full'}")
    print(f"Max Analysis Time: {settings['max_optimization_time']/60:.1f} minutes")
    
    # Set environment variables for the analysis
    os.environ['QUICK_MODE'] = str(settings['quick_mode']).lower()
    os.environ['MAX_TIME'] = str(settings['max_optimization_time'])
    os.environ['N_SAMPLES'] = str(settings['n_samples_mc'])
    
    # Run automated analysis
    print("\nStarting automated analysis...")
    if not os.getenv('GITHUB_ACTIONS'):
        print("You can safely leave - results will be saved automatically!")
    print("Check debug_log.txt for progress updates.")
    
    result = run_automated_analysis()
    
    if result:
        print("\n🎉 SUCCESS! Analysis completed.")
        print("📊 Check the generated PNG files for results.")
        print("📝 Full log available in debug_log.txt")
        
        if os.getenv('GITHUB_ACTIONS'):
            print("🔄 Results will be committed to repository automatically.")
            
        sys.exit(0)  # Success exit code
    else:
        print("\n⚠️  Analysis encountered issues.")
        print("🔍 Check debug_log.txt and error_state.json for details.")
        
        if os.getenv('GITHUB_ACTIONS'):
            print("🚨 GitHub Issue will be created automatically.")
            sys.exit(1)  # Failure exit code

if __name__ == "__main__":
    main()
