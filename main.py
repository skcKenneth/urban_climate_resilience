"""
Urban Climate-Social Network Resilience System - Main Entry Point
"""
import os
import sys
import argparse
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments

import numpy as np
import matplotlib.pyplot as plt
from models.coupled_system import CoupledSystemModel
from models.optimal_control import OptimalControlModel
from analysis.stability_analysis import StabilityAnalysis
from analysis.sensitivity_analysis import SensitivityAnalysis
from utils.parameters import ModelParameters
from utils.visualization import SystemVisualizer
from utils.data_generator import DataGenerator


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
    print("=" * 60)
    print(f"Urban Climate-Social Network Resilience System")
    print(f"Analysis Type: {analysis_type} | Quick Mode: {quick_mode} | Parallel: {parallel}")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure parameters based on mode
    if quick_mode or os.getenv('QUICK_MODE', 'false').lower() == 'true':
        days = int(os.getenv('SIMULATION_DAYS', '90'))
        samples = int(os.getenv('N_SAMPLES', '50'))
        population = 5000
        print(f"Quick mode: {days} days, {samples} samples, {population} population")
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
    else:  # full analysis
        scenarios_to_run = list(scenarios.keys())
    
    # Initialize components
    generator = DataGenerator()
    visualizer = SystemVisualizer()
    
    # Store results
    all_results = {}
    
    # Run simulations for each scenario
    for scenario_name in scenarios_to_run:
        print(f"\nProcessing {scenario_name} scenario...")
        params = scenarios[scenario_name]
        
        try:
            with timeout_context(int(os.getenv('MAX_TIME', '3600'))):
                # Generate time-varying parameters
                t = np.linspace(0, days, days)
                temp_data, humidity_data = generator.generate_climate_scenario(
                    scenario_type=scenario_name,
                    days=days
                )
                
                # Create and run coupled model
                model = CoupledSystemModel(params)
                
                # Create interpolation functions for temperature and humidity
                T_func = lambda time: np.interp(time, t, temp_data)
                H_func = lambda time: np.interp(time, t, humidity_data)
                
                # Initial conditions: [S, E, I, R, k_avg, C]
                y0 = [params.N * 0.99, 0, params.N * 0.01, 0, params.k_0, 0.3]
                
                # Solve the coupled system
                time_points, state = model.solve_coupled_system(
                    t_span=(0, days),
                    y0=y0,
                    T_func=T_func,
                    H_func=H_func,
                    t_eval=t
                )
                
                # Store results
                S, E, I, R, k_avg, C = state
                all_results[scenario_name] = {
                    'time': time_points,
                    'epidemic': {'S': S, 'E': E, 'I': I, 'R': R},
                    'network': {'k_avg': k_avg, 'C': C},
                    'temperature': temp_data,
                    'humidity': humidity_data,
                    'params': params,
                    'state': state
                }
                
                # Generate visualizations
                print(f"  Generating visualizations for {scenario_name}...")
                fig = visualizer.plot_epidemic_dynamics(
                    time_points, 
                    state,
                    T_func,
                    title=f"Epidemic Dynamics - {scenario_name}"
                )
                fig.savefig(f"{output_dir}/epidemic_{scenario_name}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
        except TimeoutError:
            print(f"  Timeout reached for {scenario_name} scenario")
        except Exception as e:
            print(f"  Error in {scenario_name} scenario: {e}")
    
    # Run additional analyses if not in quick mode
    if not quick_mode and analysis_type in ['full', 'baseline']:
        try:
            # Stability analysis
            print("\nPerforming stability analysis...")
            stability = StabilityAnalysis(scenarios['baseline'])
            
            # Phase portrait using baseline results
            if 'baseline' in all_results:
                fig = visualizer.plot_phase_portrait(
                    all_results['baseline'],
                    variables=['I', 'k_avg']
                )
                fig.savefig(f"{output_dir}/phase_portrait.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # Bifurcation analysis
            bifurcation_results = stability.bifurcation_analysis(
                parameter_name='temperature_baseline',
                parameter_range=np.linspace(15, 40, 20 if quick_mode else 50)
            )
            fig = visualizer.plot_bifurcation_diagram(bifurcation_results)
            fig.savefig(f"{output_dir}/bifurcation_diagram.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Sensitivity analysis (reduced for quick mode)
            if parallel:
                print("\nPerforming parallel sensitivity analysis...")
                from parallel_analysis import ParallelAnalysis
                parallel_analyzer = ParallelAnalysis(scenarios['baseline'])
                sensitivity_results = parallel_analyzer.run_sensitivity_analysis(
                    n_samples=samples
                )
            else:
                print("\nPerforming sensitivity analysis...")
                sensitivity = SensitivityAnalysis(scenarios['baseline'])
                sensitivity_results = sensitivity.sobol_sensitivity_analysis(
                    n_samples=samples,
                    T_scenario='baseline'
                )
            
            if sensitivity_results:
                fig = visualizer.plot_sensitivity_analysis(sensitivity_results)
                fig.savefig(f"{output_dir}/sensitivity_analysis.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            
        except Exception as e:
            print(f"Error in additional analyses: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 60)
    
    return all_results


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
