"""
Main parallel execution script for Urban Climate-Social Network Resilience System
"""
import os
import signal
import time
from contextlib import contextmanager
import matplotlib
# Set non-interactive backend for headless environments
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from models.coupled_system import CoupledSystemModel
from models.optimal_control import OptimalControlModel
from analysis.stability_analysis import StabilityAnalysis
from analysis.sensitivity_analysis import SensitivityAnalysis
from utils.parameters import ModelParameters
from utils.visualization import SystemVisualizer
from utils.data_generator import DataGenerator
from parallel_analysis import ParallelAnalysis

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set up signal handler (Unix only)
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        yield
    except AttributeError:
        # Windows doesn't have SIGALRM, just yield without timeout
        yield
    finally:
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        except AttributeError:
            pass

def main():
    """Main parallel analysis pipeline"""
    print("=" * 60)
    print("Urban Climate-Social Network Resilience System - Parallel Analysis")
    print("=" * 60)
    
    # Get environment settings
    analysis_type = os.getenv('ANALYSIS_TYPE', 'full')
    max_time = int(os.getenv('MAX_TIME', 3600))  # Default 60 minutes
    simulation_days = int(os.getenv('SIMULATION_DAYS', 365))
    n_samples = int(os.getenv('N_SAMPLES', 500))
    
    print(f"Analysis Type: {analysis_type}")
    print(f"Max Time: {max_time/60:.1f} minutes")
    print(f"Simulation Days: {simulation_days}")
    print(f"Monte Carlo Samples: {n_samples}")
    
    # Initialize components
    params = ModelParameters()
    coupled_model = CoupledSystemModel(params)
    visualizer = SystemVisualizer()
    data_gen = DataGenerator(params)
    parallel_analyzer = ParallelAnalysis()
    
    # Run analysis based on type
    if analysis_type == 'baseline':
        run_baseline_analysis(params, coupled_model, visualizer, data_gen, max_time)
    elif analysis_type == 'sensitivity':
        run_sensitivity_analysis(params, parallel_analyzer, visualizer, max_time, n_samples)
    elif analysis_type == 'uncertainty':
        run_uncertainty_analysis(params, parallel_analyzer, visualizer, max_time, n_samples)
    elif analysis_type == 'control':
        run_control_analysis(params, coupled_model, visualizer, max_time)
    else:
        # Run all analyses
        run_baseline_analysis(params, coupled_model, visualizer, data_gen, max_time // 4)
        run_sensitivity_analysis(params, parallel_analyzer, visualizer, max_time // 4, n_samples)
        run_uncertainty_analysis(params, parallel_analyzer, visualizer, max_time // 4, n_samples)
        run_control_analysis(params, coupled_model, visualizer, max_time // 4)
    
    print("\n" + "=" * 60)
    print("Parallel analysis completed!")
    print("=" * 60)
    
    return True

def run_baseline_analysis(params, coupled_model, visualizer, data_gen, max_time):
    """Run baseline system analysis"""
    print("\n1. Running baseline system analysis...")
    
    # Generate climate scenarios
    climate_scenarios = data_gen.generate_climate_scenarios()
    
    # Limit scenarios for efficiency
    climate_scenarios = {k: v for k, v in list(climate_scenarios.items())[:3]}
    
    results = {}
    
    for scenario_name, climate_data in climate_scenarios.items():
        print(f"   Processing {scenario_name} scenario...")
        
        t_climate = climate_data['t']
        T_climate = climate_data['T']
        H_climate = climate_data['H']
        
        # Create interpolation functions
        T_func = lambda time: np.interp(time, t_climate, T_climate)
        H_func = lambda time: np.interp(time, t_climate, H_climate)
        
        # Initial conditions
        y0 = [params.N * 0.99, 0, params.N * 0.01, 0, params.k_0, 0.3]
        
        # Solve system with timeout
        try:
            with timeout_context(max_time // len(climate_scenarios)):
                sim_days = int(os.getenv('SIMULATION_DAYS', 365))
                t, y = coupled_model.solve_coupled_system([0, sim_days], y0, T_func, H_func)
                results[scenario_name] = {'t': t, 'y': y, 'T_func': T_func, 'H_func': H_func}
                
                # Calculate final metrics
                S, E, I, R, k_avg, C = y
                final_attack_rate = R[-1] / params.N
                peak_infections = np.max(I)
                min_network_degree = np.min(k_avg)
                
                print(f"      Final attack rate: {final_attack_rate:.3f}")
                print(f"      Peak infections: {peak_infections:.0f}")
                print(f"      Min network degree: {min_network_degree:.2f}")
                
        except Exception as e:
            print(f"      Error in {scenario_name}: {e}")
            continue
    
    # Visualize baseline results
    print("\n2. Creating baseline visualizations...")
    for scenario_name, result in results.items():
        if 'y' in result:
            try:
                fig = visualizer.plot_epidemic_dynamics(
                    result['t'], result['y'], 
                    f"Epidemic Dynamics - {scenario_name}",
                    save_path=f"epidemic_dynamics_{scenario_name}.png"
                )
                plt.close(fig)
                
                fig = visualizer.plot_network_evolution(
                    result['t'], result['y'],
                    f"Network Evolution - {scenario_name}",
                    save_path=f"network_evolution_{scenario_name}.png"
                )
                plt.close(fig)
                
            except Exception as e:
                print(f"      Visualization error for {scenario_name}: {e}")
                continue

def run_sensitivity_analysis(params, parallel_analyzer, visualizer, max_time, n_samples):
    """Run parallel sensitivity analysis"""
    print("\n3. Running parallel sensitivity analysis...")
    
    try:
        with timeout_context(max_time):
            sensitivity_results = parallel_analyzer.parallel_sensitivity_analysis(
                n_samples=n_samples, T_scenario='baseline'
            )
            
        if sensitivity_results:
            # Visualize sensitivity results
            fig = visualizer.plot_sensitivity_analysis(sensitivity_results)
            plt.savefig("sensitivity_analysis.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("   ✅ Sensitivity analysis completed")
        else:
            print("   ⚠️  No sensitivity results generated")
            
    except Exception as e:
        print(f"   ❌ Sensitivity analysis error: {e}")

def run_uncertainty_analysis(params, parallel_analyzer, visualizer, max_time, n_samples):
    """Run parallel uncertainty analysis"""
    print("\n4. Running parallel uncertainty analysis...")
    
    try:
        with timeout_context(max_time):
            mc_results = parallel_analyzer.parallel_monte_carlo(
                n_samples=n_samples, T_scenario='heatwave'
            )
            
        if mc_results:
            # Calculate uncertainty bounds
            uncertainty_bounds = {}
            metrics = list(mc_results[0].keys())
            
            for metric in metrics:
                values = [r[metric] for r in mc_results if r is not None]
                if values:
                    values = np.array(values)
                    uncertainty_bounds[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'percentiles': np.percentile(values, [5, 25, 50, 75, 95])
                    }
            
            # Visualize uncertainty results
            fig = visualizer.plot_uncertainty_analysis(uncertainty_bounds)
            plt.savefig("uncertainty_analysis.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("   ✅ Uncertainty analysis completed")
        else:
            print("   ⚠️  No Monte Carlo results generated")
            
    except Exception as e:
        print(f"   ❌ Uncertainty analysis error: {e}")

def run_control_analysis(params, coupled_model, visualizer, max_time):
    """Run optimal control analysis"""
    print("\n5. Running optimal control analysis...")
    
    try:
        with timeout_context(max_time):
            control_model = OptimalControlModel(params)
            control_results = control_model.optimize_control_strategy()
            
        if control_results:
            # Visualize control results
            fig = visualizer.plot_optimal_control(control_results)
            plt.savefig("optimal_control.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("   ✅ Control analysis completed")
        else:
            print("   ⚠️  No control results generated")
            
    except Exception as e:
        print(f"   ❌ Control analysis error: {e}")

if __name__ == "__main__":
    main()