#!/usr/bin/env python3
"""
Full analysis script for research paper
Generates all figures and results for the climate resilience study
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Set backend before importing matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import all required modules
from models.coupled_system import CoupledSystemModel
from models.optimal_control import OptimalControlModel
from analysis.stability_analysis import StabilityAnalysis
from analysis.sensitivity_analysis import SensitivityAnalysis
from utils.parameters import ModelParameters
from utils.visualization import SystemVisualizer
from utils.data_generator import DataGenerator


def run_full_paper_analysis(output_dir='results'):
    """Run complete analysis for research paper"""
    print("="*60)
    print("FULL CLIMATE RESILIENCE ANALYSIS FOR PAPER")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Analysis parameters for paper
    days = int(os.getenv('SIMULATION_DAYS', '365'))
    n_samples = int(os.getenv('N_SAMPLES', '500'))
    
    # Define scenarios
    scenarios = {
        'baseline': {
            'N': 10000,
            'T_0': 25.0,
            'H_0': 0.6,
            'description': 'Normal climate conditions'
        },
        'heatwave': {
            'N': 10000,
            'T_0': 30.0,
            'H_0': 0.8,
            'description': 'Heatwave conditions'
        },
        'extreme': {
            'N': 10000,
            'T_0': 35.0,
            'H_0': 0.9,
            'T_critical': 40.0,
            'description': 'Extreme climate conditions'
        }
    }
    
    results = {}
    viz = SystemVisualizer(output_dir=output_dir)
    
    # Run each scenario
    for scenario_name, config in scenarios.items():
        print(f"\n{'='*50}")
        print(f"Running {scenario_name.upper()} scenario")
        print(f"Description: {config['description']}")
        print(f"{'='*50}")
        
        # Set up parameters
        params = ModelParameters()
        params.N = config['N']
        params.T_0 = config.get('T_0', params.T_0)
        params.H_0 = config.get('H_0', params.H_0)
        if 'T_critical' in config:
            params.T_critical = config['T_critical']
        
        # Generate data
        print(f"Generating {days}-day climate data...")
        data_gen = DataGenerator(params)
        climate_data, network_data = data_gen.generate_synthetic_data(
            days=days,
            save_path=Path(output_dir) / f"{scenario_name}_data.npz"
        )
        
        # Initialize and run model
        print("Running coupled system simulation...")
        model = CoupledSystemModel(params, climate_data, network_data)
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
        
        # Generate main visualization
        print("Generating simulation plots...")
        viz.plot_simulation_results(
            result, model,
            title=f"{scenario_name.capitalize()} Scenario - {days} Days",
            save_name=f"{scenario_name}_simulation"
        )
        
        # Epidemic dynamics plot
        viz.plot_epidemic_dynamics(
            result.t, result.y,
            lambda t: np.interp(t, np.arange(days), climate_data['temperature']),
            title=f"Epidemic Dynamics - {scenario_name.capitalize()}",
            save_name=f"{scenario_name}_epidemic"
        )
    
    # Stability analysis for baseline
    print("\n" + "="*50)
    print("STABILITY ANALYSIS")
    print("="*50)
    
    baseline_model = results['baseline']['model']
    stability = StabilityAnalysis(baseline_model)
    
    # Equilibrium analysis
    print("Analyzing equilibrium points...")
    eq_state = results['baseline']['simulation'].y[:, -1]
    stability_results = stability.analyze_equilibrium(eq_state)
    viz.plot_stability_analysis(stability_results, save_name="stability_analysis")
    
    # Bifurcation analysis
    print("Performing bifurcation analysis...")
    T_range = np.linspace(20, 40, 50)
    bifurcation_results = stability.bifurcation_analysis_temperature(T_range)
    viz.plot_bifurcation_diagram(bifurcation_results, save_name="bifurcation_diagram")
    
    # Sensitivity analysis
    print("\n" + "="*50)
    print("SENSITIVITY ANALYSIS")
    print("="*50)
    
    sensitivity = SensitivityAnalysis(baseline_model)
    
    # Morris screening
    print(f"Running Morris screening with {n_samples} samples...")
    morris_results = sensitivity.morris_screening(n_samples=n_samples)
    viz.plot_sensitivity_results(morris_results, save_name="morris_sensitivity")
    
    # Sobol analysis (reduced samples for time)
    print("Running Sobol sensitivity analysis...")
    sobol_results = sensitivity.sobol_sensitivity_analysis(
        n_samples=min(n_samples//2, 100),
        T_scenario='baseline'
    )
    viz.plot_sensitivity_heatmap(sobol_results, save_name="sobol_sensitivity")
    
    # Comparison plots
    print("\n" + "="*50)
    print("GENERATING COMPARISON PLOTS")
    print("="*50)
    
    # Compare scenarios
    viz.compare_scenarios(results, save_name="scenario_comparison")
    
    # Phase portraits
    viz.plot_phase_portrait(
        {'baseline': results['baseline']['simulation']},
        variables=['I', 'k_avg'],
        save_name="phase_portrait_baseline"
    )
    
    # Optimal control analysis
    print("\n" + "="*50)
    print("OPTIMAL CONTROL ANALYSIS")
    print("="*50)
    
    control_model = OptimalControlModel(
        results['baseline']['params'],
        results['baseline']['climate_data'],
        results['baseline']['network_data']
    )
    
    # Simple control optimization
    print("Computing optimal control strategies...")
    initial_state = np.array([params.N*0.99, 0, params.N*0.01, 0, params.k_0, 0.3])
    t_control = np.linspace(0, 90, 91)  # 90 days control horizon
    
    control_results = control_model.solve_optimal_control(
        initial_state, t_control,
        control_bounds={'u_medical': (0, 0.5), 'u_social': (0, 0.3)}
    )
    
    viz.plot_control_strategies(control_results, save_name="optimal_control")
    
    # Generate summary statistics
    print("\n" + "="*50)
    print("GENERATING SUMMARY STATISTICS")
    print("="*50)
    
    with open(Path(output_dir) / "analysis_summary.txt", 'w') as f:
        f.write("URBAN CLIMATE RESILIENCE ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        for scenario, result in results.items():
            sim = result['simulation']
            f.write(f"{scenario.upper()} SCENARIO:\n")
            f.write(f"  Final infected: {sim.y[2, -1]:.0f}\n")
            f.write(f"  Total recovered: {sim.y[3, -1]:.0f}\n")
            f.write(f"  Final network connectivity: {sim.y[4, -1]:.2f}\n")
            f.write(f"  Final clustering: {sim.y[5, -1]:.3f}\n\n")
        
        f.write("STABILITY ANALYSIS:\n")
        f.write(f"  Eigenvalues (real parts): {stability_results['eigenvalues'].real}\n")
        f.write(f"  System stable: {stability_results['stable']}\n\n")
        
        f.write("SENSITIVITY ANALYSIS:\n")
        f.write("  Most sensitive parameters:\n")
        for param, sens in morris_results['mean'].items():
            if abs(sens) > 0.1:
                f.write(f"    {param}: {sens:.3f}\n")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"All results saved to: {output_dir}/")
    print("="*60)
    
    # List all generated files
    print("\nGenerated files:")
    for file in sorted(Path(output_dir).glob("*")):
        print(f"  - {file.name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run full analysis for paper')
    parser.add_argument('--analysis-type', default='full',
                        choices=['full', 'baseline', 'heatwave', 'extreme'])
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    
    try:
        if args.analysis_type == 'full':
            return 0 if run_full_paper_analysis(args.output_dir) else 1
        else:
            # For individual scenarios, still run full analysis but highlight the specific one
            print(f"Running full analysis with focus on {args.analysis_type}")
            return 0 if run_full_paper_analysis(args.output_dir) else 1
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())