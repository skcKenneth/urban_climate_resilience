"""
Complete automated execution with GitHub Actions support
"""
import os
import sys
import argparse
from debug_runner import run_automated_analysis
from config_manager import SmartConfig

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Climate-Social Resilience Analysis')
    parser.add_argument('--analysis-type', choices=['quick', 'full'], default='full',
                       help='Analysis type: quick (fast) or full (comprehensive)')
    parser.add_argument('--max-time', type=int, default=None,
                       help='Maximum analysis time in minutes')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of Monte Carlo samples')
    parser.add_argument('--simulation-days', type=int, default=None,
                       help='Number of simulation days')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AUTOMATED CLIMATE-SOCIAL RESILIENCE ANALYSIS")
    print(f"Analysis Type: {args.analysis_type.upper()}")
    
    # Check if running in GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        print("🤖 Running in GitHub Actions environment")
        
        # Optimized settings for GitHub Actions
        if args.analysis_type == 'quick':
            settings = {
                'n_samples_sensitivity': 50,    # Reduced from 100
                'n_samples_mc': 100,            # Reduced from 200
                'simulation_time': 180,          # Reduced from 365
                'quick_mode': True,
                'max_optimization_time': 600     # 10 minutes
            }
        else:  # full
            settings = {
                'n_samples_sensitivity': 100,
                'n_samples_mc': 200,
                'simulation_time': 365,
                'quick_mode': False,
                'max_optimization_time': 1200   # 20 minutes
            }
    else:
        print("💻 Running locally")
        config = SmartConfig()
        settings = config.get_optimal_settings()
        print(f"System Memory: {config.system_memory:.1f} GB")
        print(f"CPU Cores: {config.cpu_count}")
    
    # Override settings with command line arguments
    if args.max_time:
        settings['max_optimization_time'] = args.max_time * 60
    if args.n_samples:
        settings['n_samples_mc'] = args.n_samples
    if args.simulation_days:
        settings['simulation_time'] = args.simulation_days
    
    print("="*60)
    
    print(f"Selected Mode: {'Quick' if settings['quick_mode'] else 'Full'}")
    print(f"Max Analysis Time: {settings['max_optimization_time']/60:.1f} minutes")
    print(f"Monte Carlo Samples: {settings['n_samples_mc']}")
    print(f"Simulation Days: {settings['simulation_time']}")
    
    # Set environment variables for the analysis
    os.environ['QUICK_MODE'] = str(settings['quick_mode']).lower()
    os.environ['MAX_TIME'] = str(settings['max_optimization_time'])
    os.environ['N_SAMPLES'] = str(settings['n_samples_mc'])
    os.environ['SIMULATION_DAYS'] = str(settings['simulation_time'])
    os.environ['ANALYSIS_TYPE'] = args.analysis_type
    
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
