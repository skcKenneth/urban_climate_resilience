"""
Main execution script for Urban Climate-Social Network Resilience System
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
    """Main analysis pipeline with optimized settings"""
    print("=" * 60)
    print("Urban Climate-Social Network Resilience System Analysis")
    print("=" * 60)
    
    # Get environment settings
    quick_mode = os.getenv('QUICK_MODE', 'false').lower() == 'true'
    max_time = int(os.getenv('MAX_TIME', 1200))  # Default 20 minutes
    simulation_days = int(os.getenv('SIMULATION_DAYS', 365))
    
    print(f"Quick Mode: {quick_mode}")
    print(f"Max Time: {max_time/60:.1f} minutes")
    print(f"Simulation Days: {simulation_days}")
    
    # Initialize components
    params = ModelParameters()
    coupled_model = CoupledSystemModel(params)
    visualizer = SystemVisualizer()
    data_gen = DataGenerator(params)
    
    # Generate climate scenarios
    print("\n1. Generating climate scenarios...")
    climate_scenarios = data_gen.generate_climate_scenarios()
    
    # Limit scenarios in quick mode
    if quick_mode:
        climate_scenarios = {k: v for k, v in list(climate_scenarios.items())[:2]}
        print(f"Quick mode: Using only {len(climate_scenarios)} scenarios")
    
    # Task 1-3: Basic system analysis
    print("\n2. Running basic system simulations...")
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
                t, y = coupled_model.solve_coupled_system([0, simulation_days], y0, T_func, H_func)
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
    
    # Visualize basic results
    print("\n3. Creating basic visualizations...")
    for scenario_name, result in results.items():
        if 'y' in result:
            try:
                fig = visualizer.plot_epidemic_dynamics(
                    result['t'], result['y'], 
                    f"Epidemic Dynamics - {scenario_name}",
                    save_path=f"epidemic_dynamics_{scenario_name}.png"
                )
                plt.close(fig)  # Close to free memory
                
                fig = visualizer.plot_network_evolution(
                    result['t'], result['y'],
                    f"Network Evolution - {scenario_name}",
                    save_path=f"network_evolution_{scenario_name}.png"
                )
                plt.close(fig)
                
            except Exception as e:
                print(f"      Visualization error for {scenario_name}: {e}")
                continue
    
    # Task 4: Stability analysis (skip in quick mode)
    if not quick_mode:
        print("\n4. Running stability analysis...")
        try:
            stability_analyzer = StabilityAnalysis(params)
            
            with timeout_context(max_time // 4):
                stability_results = stability_analyzer.analyze_system_stability()
                
            # Visualize stability results
            if stability_results:
                fig = visualizer.plot_stability_analysis(stability_results)
                plt.savefig("stability_analysis.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
        except Exception as e:
            print(f"   Stability analysis error: {e}")
    else:
        print("\n4. Skipping stability analysis (quick mode)")
    
    # Task 5: Sensitivity analysis
    print("\n5. Running sensitivity analysis...")
    try:
        sensitivity_analyzer = SensitivityAnalysis(params)
        
        with timeout_context(max_time // 4):
            sensitivity_results = sensitivity_analyzer.sobol_sensitivity_analysis()
            
        if sensitivity_results:
            # Visualize sensitivity results
            fig = visualizer.plot_sensitivity_analysis(sensitivity_results)
            plt.savefig("sensitivity_analysis.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            print("   No sensitivity results generated")
            
    except Exception as e:
        print(f"   Sensitivity analysis error: {e}")
    
    # Task 6: Monte Carlo uncertainty analysis
    print("\n6. Running Monte Carlo uncertainty analysis...")
    try:
        with timeout_context(max_time // 4):
            mc_results = sensitivity_analyzer.monte_carlo_uncertainty()
            
        if mc_results:
            uncertainty_bounds = sensitivity_analyzer.calculate_uncertainty_bounds(mc_results)
            
            # Visualize uncertainty results
            fig = visualizer.plot_uncertainty_analysis(uncertainty_bounds)
            plt.savefig("uncertainty_analysis.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            print("   No Monte Carlo results generated")
            
    except Exception as e:
        print(f"   Monte Carlo analysis error: {e}")
    
    # Task 7: Optimal control (skip in quick mode)
    if not quick_mode:
        print("\n7. Running optimal control analysis...")
        try:
            control_model = OptimalControlModel(params)
            
            with timeout_context(max_time // 4):
                control_results = control_model.optimize_control_strategy()
                
            if control_results:
                # Visualize control results
                fig = visualizer.plot_optimal_control(control_results)
                plt.savefig("optimal_control.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            else:
                print("   No control results generated")
                
        except Exception as e:
            print(f"   Optimal control error: {e}")
    else:
        print("\n7. Skipping optimal control (quick mode)")
    
    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()
