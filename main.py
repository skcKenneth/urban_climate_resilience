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
        
        if analysis_type == 'heatwave':
            params.T_0 = 30.0
            params.H_0 = 0.8
        elif analysis_type == 'extreme':
            params.T_0 = 35.0
            params.H_0 = 0.9
            params.T_critical = 40.0
        
        # Generate data
        print("Generating synthetic data...")
        data_gen = DataGenerator(params)
        climate_data, network_data = data_gen.generate_synthetic_data(
            days=days,
            save_path=Path(output_dir) / f"{analysis_type}_data.npz"
        )
        
        # Initialize model
        print("Initializing coupled system model...")
        model = CoupledSystemModel(params, climate_data, network_data)
        
        # Run simulation
        print(f"Running simulation for {days} days...")
        t_span = (0, days)
        result = model.simulate(t_span, method='RK45')
        
        # Visualize results
        print("Generating visualizations...")
        viz = SystemVisualizer(output_dir=output_dir)
        
        # Main simulation plot
        viz.plot_simulation_results(
            result, model,
            title=f"{analysis_type.capitalize()} Scenario",
            save_name=f"{analysis_type}_simulation"
        )
        
        # Additional analysis if not in quick mode
        if not quick_mode:
            # Stability analysis
            print("Performing stability analysis...")
            stability = StabilityAnalysis(model)
            eq_state = result.y[:, -1]
            stability_results = stability.analyze_equilibrium(eq_state)
            viz.plot_stability_analysis(
                stability_results,
                save_name=f"{analysis_type}_stability"
            )
            
            # Sensitivity analysis (with fewer samples for speed)
            print("Performing sensitivity analysis...")
            sensitivity = SensitivityAnalysis(model)
            sensitivity_results = sensitivity.morris_screening(n_samples=min(samples, 100))
            viz.plot_sensitivity_results(
                sensitivity_results,
                save_name=f"{analysis_type}_sensitivity"
            )
        
        # Generate summary
        summary_path = Path(output_dir) / f"{analysis_type}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Analysis Summary - {analysis_type.capitalize()}\n")
            f.write("="*50 + "\n")
            f.write(f"Days simulated: {days}\n")
            f.write(f"Population: {population}\n")
            f.write(f"Final infected: {result.y[2, -1]:.0f}\n")
            f.write(f"Final recovered: {result.y[3, -1]:.0f}\n")
            f.write(f"Final network connectivity: {result.y[4, -1]:.2f}\n")
        
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
