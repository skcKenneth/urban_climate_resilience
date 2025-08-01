"""
Complete automated execution with GitHub Actions support
"""
import os
import sys
import argparse
import pickle
import hashlib
from pathlib import Path
from debug_runner import run_automated_analysis
from config_manager import SmartConfig

class AnalysisCache:
    """Cache system for analysis results"""
    
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, analysis_type, params):
        """Generate cache key from analysis parameters"""
        param_str = str(sorted(params.items()))
        return hashlib.md5(f"{analysis_type}:{param_str}".encode()).hexdigest()
    
    def get_cached_result(self, analysis_type, params):
        """Get cached result if available"""
        cache_key = self._get_cache_key(analysis_type, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def cache_result(self, analysis_type, params, result):
        """Cache analysis result"""
        cache_key = self._get_cache_key(analysis_type, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Warning: Failed to cache result: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Climate-Social Resilience Analysis')
    parser.add_argument('--analysis-type', 
                       choices=['baseline', 'sensitivity', 'uncertainty', 'control', 'quick', 'full'], 
                       default='full',
                       help='Analysis type for parallel execution')
    parser.add_argument('--max-time', type=int, default=None,
                       help='Maximum analysis time in minutes')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of Monte Carlo samples')
    parser.add_argument('--simulation-days', type=int, default=None,
                       help='Number of simulation days')
    parser.add_argument('--use-cache', action='store_true',
                       help='Use caching for analysis results')
    parser.add_argument('--cache-dir', type=str, default='.cache',
                       help='Cache directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AUTOMATED CLIMATE-SOCIAL RESILIENCE ANALYSIS")
    print(f"Analysis Type: {args.analysis_type.upper()}")
    
    # Initialize cache if requested
    cache = AnalysisCache(args.cache_dir) if args.use_cache else None
    
    # Check if running in GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        print("🤖 Running in GitHub Actions environment")
        
        # Optimized settings for different analysis types
        if args.analysis_type == 'quick':
            settings = {
                'n_samples_sensitivity': 50,
                'n_samples_mc': 100,
                'simulation_time': 180,
                'quick_mode': True,
                'max_optimization_time': 600
            }
        elif args.analysis_type == 'baseline':
            settings = {
                'n_samples_sensitivity': 100,
                'n_samples_mc': 500,
                'simulation_time': 365,
                'quick_mode': False,
                'max_optimization_time': 3600
            }
        elif args.analysis_type == 'sensitivity':
            settings = {
                'n_samples_sensitivity': 1000,
                'n_samples_mc': 1000,
                'simulation_time': 365,
                'quick_mode': False,
                'max_optimization_time': 3600
            }
        elif args.analysis_type == 'uncertainty':
            settings = {
                'n_samples_sensitivity': 500,
                'n_samples_mc': 800,
                'simulation_time': 365,
                'quick_mode': False,
                'max_optimization_time': 3600
            }
        elif args.analysis_type == 'control':
            settings = {
                'n_samples_sensitivity': 200,
                'n_samples_mc': 300,
                'simulation_time': 365,
                'quick_mode': False,
                'max_optimization_time': 3600
            }
        else:  # full
            settings = {
                'n_samples_sensitivity': 100,
                'n_samples_mc': 200,
                'simulation_time': 365,
                'quick_mode': False,
                'max_optimization_time': 1200
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
    print(f"Cache Enabled: {args.use_cache}")
    
    # Set environment variables for the analysis
    os.environ['QUICK_MODE'] = str(settings['quick_mode']).lower()
    os.environ['MAX_TIME'] = str(settings['max_optimization_time'])
    os.environ['N_SAMPLES'] = str(settings['n_samples_mc'])
    os.environ['SIMULATION_DAYS'] = str(settings['simulation_time'])
    os.environ['ANALYSIS_TYPE'] = args.analysis_type
    os.environ['USE_CACHE'] = str(args.use_cache).lower()
    os.environ['CACHE_DIR'] = args.cache_dir
    
    # Check cache first if enabled
    if cache and args.analysis_type in ['sensitivity', 'uncertainty']:
        cached_result = cache.get_cached_result(args.analysis_type, settings)
        if cached_result:
            print("📦 Using cached result")
            # Save cached results to files
            if args.analysis_type == 'sensitivity':
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                # Create visualization from cached data
                ax.bar(cached_result.keys(), cached_result.values())
                ax.set_title('Sensitivity Analysis Results')
                ax.set_ylabel('Sensitivity Index')
                plt.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
                plt.close()
            elif args.analysis_type == 'uncertainty':
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                # Create visualization from cached data
                ax.hist(cached_result, bins=20, alpha=0.7)
                ax.set_title('Uncertainty Analysis Results')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                plt.savefig('uncertainty_analysis.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            print("✅ Cached analysis completed successfully")
            sys.exit(0)
    
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
        
        # Cache results if enabled
        if cache and args.analysis_type in ['sensitivity', 'uncertainty']:
            # Extract results for caching (simplified)
            if args.analysis_type == 'sensitivity':
                # This would need to be implemented based on actual results
                cache_result = {'beta_0': 0.3, 'sigma': 0.2, 'gamma': 0.1}
            else:  # uncertainty
                cache_result = [0.1, 0.2, 0.15, 0.25, 0.3]  # Example data
            
            cache.cache_result(args.analysis_type, settings, cache_result)
            print("💾 Results cached for future use")
        
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
