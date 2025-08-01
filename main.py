"""
Urban Climate-Social Network Resilience System - Main Entry Point
"""
import os
import sys
import argparse
import signal
import time
import logging
import traceback
from contextlib import contextmanager
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('analysis.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
logger.info(f"Matplotlib backend set to: {matplotlib.get_backend()}")

# Import with error handling
try:
    import numpy as np
    import matplotlib.pyplot as plt
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    sys.exit(1)

# Import local modules with error handling
try:
    from models.coupled_system import CoupledSystemModel
    from models.optimal_control import OptimalControlModel
    from analysis.stability_analysis import StabilityAnalysis
    from analysis.sensitivity_analysis import SensitivityAnalysis
    from utils.parameters import ModelParameters
    from utils.visualization import SystemVisualizer
    from utils.data_generator import DataGenerator
    logger.info("Successfully imported all local modules")
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Python path: {sys.path}")
    logger.error(traceback.format_exc())
    sys.exit(1)


@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up signal handler (Unix only)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows doesn't have SIGALRM
        yield


def run_analysis(analysis_type='full', quick_mode=False, parallel=False, output_dir='results'):
    """
    Run the climate resilience analysis
    
    Args:
        analysis_type: Type of analysis ('full', 'baseline', 'heatwave', 'extreme', 'quick')
        quick_mode: Whether to run in quick mode with reduced parameters
        parallel: Whether to use parallel processing
        output_dir: Directory for output files
    """
    logger.info("=" * 60)
    logger.info(f"Urban Climate-Social Network Resilience System")
    logger.info(f"Analysis Type: {analysis_type} | Quick Mode: {quick_mode} | Parallel: {parallel}")
    logger.info("=" * 60)
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir}")
        
        # Configure parameters based on mode
        if quick_mode or os.getenv('QUICK_MODE', 'false').lower() == 'true':
            days = int(os.getenv('SIMULATION_DAYS', '90'))
            samples = int(os.getenv('N_SAMPLES', '50'))
            population = 5000
            logger.info(f"Quick mode: {days} days, {samples} samples, {population} population")
        else:
            days = int(os.getenv('SIMULATION_DAYS', '365'))
            samples = int(os.getenv('N_SAMPLES', '500'))
            population = 10000
        
        # Configure scenarios
        baseline_params = ModelParameters()
        baseline_params.N = population
        
        heatwave_params = ModelParameters()
        heatwave_params.N = population
        heatwave_params.T_0 = 30.0  # Higher reference temperature
        heatwave_params.H_0 = 0.8   # Higher humidity
        
        extreme_params = ModelParameters()
        extreme_params.N = population
        extreme_params.T_0 = 35.0   # Extreme temperature
        extreme_params.H_0 = 0.9    # Extreme humidity
        extreme_params.T_critical = 40.0  # Higher critical threshold
        
        scenarios = {
            'baseline': baseline_params,
            'heatwave': heatwave_params,
            'extreme': extreme_params
        }
        
        # Determine which scenarios to run
        if analysis_type == 'quick':
            scenarios_to_run = ['baseline']
        elif analysis_type in scenarios:
            scenarios_to_run = [analysis_type]
        elif analysis_type == 'full':
            scenarios_to_run = list(scenarios.keys())
        else:
            logger.error(f"Unknown analysis type: {analysis_type}")
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        logger.info(f"Running scenarios: {scenarios_to_run}")
        
        # Initialize visualizer
        viz = SystemVisualizer(output_dir=output_dir)
        
        # Run analyses
        results = {}
        for scenario_name in scenarios_to_run:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {scenario_name} scenario...")
            logger.info(f"{'='*50}")
            
            try:
                params = scenarios[scenario_name]
                
                # Generate synthetic data
                logger.info("Generating synthetic data...")
                data_gen = DataGenerator(params)
                climate_data, network_data = data_gen.generate_synthetic_data(
                    days=days,
                    save_path=Path(output_dir) / f"{scenario_name}_data.npz"
                )
                
                # Initialize models
                logger.info("Initializing coupled system model...")
                model = CoupledSystemModel(params, climate_data, network_data)
                
                # Run simulation
                logger.info(f"Running simulation for {days} days...")
                t_span = (0, days)
                result = model.simulate(t_span, method='RK45')
                
                # Store results
                results[scenario_name] = {
                    'model': model,
                    'simulation': result,
                    'params': params,
                    'climate_data': climate_data,
                    'network_data': network_data
                }
                
                # Perform analyses
                if not quick_mode or analysis_type != 'quick':
                    # Stability analysis
                    logger.info("Performing stability analysis...")
                    stability = StabilityAnalysis(model)
                    stability_results = stability.analyze_equilibrium(result.y[:, -1])
                    results[scenario_name]['stability'] = stability_results
                    
                    # Sensitivity analysis
                    logger.info("Performing sensitivity analysis...")
                    sensitivity = SensitivityAnalysis(model)
                    sensitivity_results = sensitivity.morris_screening(n_samples=samples)
                    results[scenario_name]['sensitivity'] = sensitivity_results
                
                # Generate visualizations
                logger.info("Generating visualizations...")
                viz.plot_simulation_results(
                    result, model, 
                    title=f"{scenario_name.capitalize()} Scenario",
                    save_name=f"{scenario_name}_simulation"
                )
                
                if 'stability' in results[scenario_name]:
                    viz.plot_stability_analysis(
                        stability_results,
                        save_name=f"{scenario_name}_stability"
                    )
                
                if 'sensitivity' in results[scenario_name]:
                    viz.plot_sensitivity_results(
                        sensitivity_results,
                        save_name=f"{scenario_name}_sensitivity"
                    )
                
                logger.info(f"✅ {scenario_name} scenario completed successfully")
                
            except Exception as e:
                logger.error(f"Error in {scenario_name} scenario: {e}")
                logger.error(traceback.format_exc())
                results[scenario_name] = {'error': str(e)}
        
        # Generate comparison plots if multiple scenarios
        if len(results) > 1 and not quick_mode:
            logger.info("\nGenerating comparison plots...")
            try:
                viz.compare_scenarios(results)
            except Exception as e:
                logger.error(f"Error generating comparison plots: {e}")
        
        # Summary report
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        for scenario, result in results.items():
            if 'error' in result:
                logger.info(f"{scenario}: ❌ Failed - {result['error']}")
            else:
                logger.info(f"{scenario}: ✅ Completed")
        
        # Save summary
        summary_path = Path(output_dir) / "analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Urban Climate Resilience Analysis Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Analysis Type: {analysis_type}\n")
            f.write(f"Quick Mode: {quick_mode}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"Scenarios Run: {', '.join(scenarios_to_run)}\n")
            f.write("\nResults:\n")
            for scenario, result in results.items():
                if 'error' in result:
                    f.write(f"  {scenario}: Failed - {result['error']}\n")
                else:
                    f.write(f"  {scenario}: Completed successfully\n")
        
        logger.info(f"\nSummary saved to: {summary_path}")
        logger.info("Analysis complete!")
        
        return results
        
    except Exception as e:
        logger.error(f"Critical error in analysis: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Urban Climate-Social Network Resilience System Analysis'
    )
    parser.add_argument(
        '--analysis-type',
        choices=['full', 'baseline', 'heatwave', 'extreme', 'quick'],
        default='full',
        help='Type of analysis to run'
    )
    parser.add_argument(
        '--quick-mode',
        action='store_true',
        help='Run in quick mode with reduced parameters'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel processing where available'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory for output files (default: results)'
    )
    
    args = parser.parse_args()
    
    # Run the analysis
    try:
        run_analysis(
            analysis_type=args.analysis_type,
            quick_mode=args.quick_mode,
            parallel=args.parallel,
            output_dir=args.output_dir
        )
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
