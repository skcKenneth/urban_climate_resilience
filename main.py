"""
Urban Climate-Social Network Resilience System - Main Entry Point
"""
import os
import sys
import argparse
from pathlib import Path

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

try:
    from models.coupled_system import CoupledSystemModel
    from models.optimal_control import OptimalControlModel
    from analysis.stability_analysis import StabilityAnalysis
    from analysis.sensitivity_analysis import SensitivityAnalysis
    from utils.parameters import ModelParameters
    from utils.visualization import SystemVisualizer
    from utils.data_generator import DataGenerator
    print("Successfully imported all modules")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print("Directory contents:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            if file.endswith('.py'):
                print(f"{subindent}{file}")
    raise


def run_analysis(analysis_type='baseline', quick_mode=False, output_dir='results'):
    """Run the climate resilience analysis"""
    print("="*60)
    print(f"Urban Climate-Social Network Resilience System")
    print(f"Analysis Type: {analysis_type} | Quick Mode: {quick_mode}")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure parameters based on mode
    if quick_mode:
        days = 90
        samples = 50
        population = 5000
    else:
        days = 365
        samples = 500
        population = 10000
    
    print(f"Configuration: {days} days, {samples} samples, {population} population")
    
    try:
        # Set up parameters
        params = ModelParameters()
        params.N = population
        params.T_sim = days  # Set simulation time
        
        if analysis_type == 'heatwave':
            params.T_0 = 30.0
            params.H_0 = 0.8
        elif analysis_type == 'extreme':
            params.T_0 = 35.0
            params.H_0 = 0.9
            params.T_critical = 40.0
        
        # Generate climate data
        print("Generating climate scenario...")
        data_gen = DataGenerator(params)
        
        # Generate climate scenario
        t, T, H = data_gen.generate_climate_scenario(analysis_type, days=days)
        
        # Create climate functions for the model
        T_func = lambda time: np.interp(time, t, T)
        H_func = lambda time: np.interp(time, t, H)
        
        # Initialize model
        print("Initializing coupled system model...")
        model = CoupledSystemModel(params, n_nodes=population)
        
        # Set initial conditions
        # [S, E, I, R, k_avg, C]
        I0 = params.N * 0.01  # 1% initially infected
        E0 = 0
        S0 = params.N - I0
        R0 = 0
        k_avg0 = params.k_0  # Initial average degree
        C0 = 0.3  # Initial clustering coefficient
        
        y0 = [S0, E0, I0, R0, k_avg0, C0]
        
        # Solve the coupled system
        print(f"Running simulation for {days} days...")
        t_span = (0, days)
        t_eval = np.linspace(0, days, days+1)
        
        # Use the coupled system's solve method
        result = model.solve_coupled_system(
            t_span=t_span,
            y0=y0,
            T_func=T_func,
            H_func=H_func,
            t_eval=t_eval
        )
        
        # Extract results
        time_points, states = result
        
        # Save data
        save_path = Path(output_dir) / f"{analysis_type}_data.npz"
        np.savez(save_path,
            time=time_points,
            states=states,
            temperature=T,
            humidity=H,
            parameters={
                'N': params.N,
                'T_0': params.T_0,
                'H_0': params.H_0,
                'days': days
            }
        )
        print(f"Data saved to {save_path}")
        
        # Visualize results
        print("Generating visualizations...")
        viz = SystemVisualizer()
        
        # Create epidemic dynamics plot
        fig = viz.plot_epidemic_dynamics(
            time_points,
            states,
            T_func,
            title=f"Epidemic Dynamics - {analysis_type.capitalize()} Scenario"
        )
        fig.savefig(Path(output_dir) / f"{analysis_type}_epidemic.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Additional analysis if not in quick mode
        if not quick_mode:
            # Stability analysis
            print("Performing stability analysis...")
            stability = StabilityAnalysis(params)
            
            # Get final state for equilibrium analysis
            final_state = states[:, -1]
            
            # Analyze stability at equilibrium
            try:
                stability_results = stability.analyze_equilibrium_stability(final_state)
                
                # Generate stability visualization if we have results
                if stability_results and 'eigenvalues' in stability_results:
                    fig = viz.plot_stability_analysis(stability_results)
                    fig.savefig(Path(output_dir) / f"{analysis_type}_stability.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                print(f"Stability analysis skipped: {e}")
            
            # Sensitivity analysis (with fewer samples for speed)
            print("Performing sensitivity analysis...")
            try:
                sensitivity = SensitivityAnalysis(params)
                # Use Morris screening for quick analysis
                sensitivity_results = sensitivity.morris_screening(
                    n_samples=min(samples, 100),
                    n_groups=5,
                    parameter_ranges={
                        'beta_0': (0.1, 0.5),
                        'alpha_T': (0.01, 0.05),
                        'sigma': (0.1, 0.3),
                        'gamma': (0.1, 0.3),
                        'kappa': (0.1, 0.5)
                    }
                )
                
                if sensitivity_results:
                    fig = viz.plot_sensitivity_analysis(sensitivity_results)
                    fig.savefig(Path(output_dir) / f"{analysis_type}_sensitivity.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                print(f"Sensitivity analysis skipped: {e}")
        
        # Generate summary
        summary_path = Path(output_dir) / f"{analysis_type}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Analysis Summary - {analysis_type.capitalize()}\n")
            f.write("="*50 + "\n")
            f.write(f"Days simulated: {days}\n")
            f.write(f"Population: {population}\n")
            f.write(f"Final susceptible: {states[0, -1]:.0f}\n")
            f.write(f"Final exposed: {states[1, -1]:.0f}\n")
            f.write(f"Final infected: {states[2, -1]:.0f}\n")
            f.write(f"Final recovered: {states[3, -1]:.0f}\n")
            f.write(f"Final network connectivity: {states[4, -1]:.2f}\n")
            f.write(f"Final clustering coefficient: {states[5, -1]:.3f}\n")
        
        print(f"\nAnalysis complete! Results saved to {output_dir}/")
        print("Generated files:")
        for file in Path(output_dir).glob("*"):
            print(f"  - {file.name}")
        
        return True
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Urban Climate Resilience Analysis')
    parser.add_argument('--analysis-type', type=str, default='baseline',
                        choices=['baseline', 'heatwave', 'extreme'],
                        help='Type of analysis to run')
    parser.add_argument('--quick-mode', action='store_true',
                        help='Run in quick mode with reduced parameters')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    success = run_analysis(
        analysis_type=args.analysis_type,
        quick_mode=args.quick_mode,
        output_dir=args.output_dir
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
