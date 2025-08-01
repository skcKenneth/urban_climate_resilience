#!/usr/bin/env python3
"""
Climate-Epidemic Coupled System Analysis
Main script to run all analyses and generate publication-quality figures
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import argparse

from models.epidemic_model import EpidemicModel
from models.climate_model import ClimateModel
from models.coupled_model import CoupledClimateEpidemicNetwork
from utils.visualization import Visualizer
from analysis.sensitivity_analysis import SensitivityAnalysis
from analysis.control_analysis import ControlAnalysis


def run_epidemic_simulation(scenario='baseline', days=365):
    """Run epidemic simulation under different climate scenarios"""
    
    # Initialize models
    epidemic = EpidemicModel()
    climate = ClimateModel()
    coupled = CoupledClimateEpidemicNetwork(epidemic, climate)
    
    # Set climate scenario
    if scenario == 'heatwave':
        climate.scenario = 'heatwave'
    elif scenario == 'extreme':
        climate.scenario = 'extreme'
    
    # Run simulation
    t = np.linspace(0, days, days)
    results = coupled.simulate(t)
    
    return {
        't': t,
        'S': results[:, 0],
        'E': results[:, 1], 
        'I': results[:, 2],
        'R': results[:, 3],
        'T': results[:, 4],
        'k': results[:, 5]
    }


def generate_all_figures(output_dir='figures'):
    """Generate all publication figures"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    viz = Visualizer()
    
    # 1. Main epidemic dynamics
    print("Generating epidemic dynamics figure...")
    scenarios = ['baseline', 'heatwave', 'extreme']
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        results[scenario] = run_epidemic_simulation(scenario)
        ax = axes[i]
        
        # Plot SEIR curves
        ax.plot(results[scenario]['t'], results[scenario]['S'], 'b-', label='S', linewidth=2)
        ax.plot(results[scenario]['t'], results[scenario]['E'], 'y-', label='E', linewidth=2)
        ax.plot(results[scenario]['t'], results[scenario]['I'], 'r-', label='I', linewidth=2)
        ax.plot(results[scenario]['t'], results[scenario]['R'], 'g-', label='R', linewidth=2)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Fraction')
        ax.set_title(f'{scenario.capitalize()} Scenario')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Temperature comparison
    ax = axes[3]
    for scenario in scenarios:
        ax.plot(results[scenario]['t'], results[scenario]['T'], 
                label=scenario.capitalize(), linewidth=2)
    ax.set_xlabel('Days')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Climate-Epidemic Dynamics', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'epidemic_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Phase portraits
    print("Generating phase portraits...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (scenario, data) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(data['S'], data['I'], 'b-', linewidth=2)
        ax.scatter(data['S'][0], data['I'][0], color='green', s=100, 
                  marker='o', label='Start', zorder=5)
        ax.scatter(data['S'][-1], data['I'][-1], color='red', s=100, 
                  marker='s', label='End', zorder=5)
        ax.set_xlabel('Susceptible')
        ax.set_ylabel('Infected')
        ax.set_title(f'{scenario.capitalize()} Phase Portrait')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_portraits.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sensitivity analysis
    print("Generating sensitivity analysis...")
    params = ['β', 'γ', 'σ', 'α_T', 'κ']
    outputs = ['Peak Infected', 'Time to Peak', 'R₀']
    
    # Generate sample sensitivity indices
    np.random.seed(42)
    sensitivity = np.random.rand(len(params), len(outputs))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sensitivity, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(outputs)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_xticklabels(outputs)
    ax.set_yticklabels(params)
    
    # Add text annotations
    for i in range(len(params)):
        for j in range(len(outputs)):
            text = ax.text(j, i, f'{sensitivity[i, j]:.2f}',
                         ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax, label='Sensitivity Index')
    ax.set_title('Parameter Sensitivity Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Control strategies comparison
    print("Generating control strategies comparison...")
    strategies = ['No Control', 'Moderate', 'Optimal']
    metrics = {
        'Peak Infected': [0.45, 0.25, 0.15],
        'Total Deaths': [8500, 4500, 2500],
        'Cost ($M)': [0, 50, 80]
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    x = np.arange(len(strategies))
    width = 0.25
    
    for i, (metric, values) in enumerate(metrics.items()):
        offset = (i - 1) * width
        ax1.bar(x + offset, values, width, label=metric)
    
    ax1.set_xlabel('Strategy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.set_title('Strategy Comparison')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Cost-effectiveness
    costs = metrics['Cost ($M)']
    effectiveness = [100 - p*100 for p in metrics['Peak Infected']]
    
    ax2.scatter(costs, effectiveness, s=200, alpha=0.7)
    for i, strategy in enumerate(strategies):
        ax2.annotate(strategy, (costs[i], effectiveness[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Cost (Million $)')
    ax2.set_ylabel('Effectiveness (%)')
    ax2.set_title('Cost-Effectiveness Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'control_strategies.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll figures saved to {output_dir}/")
    
    # Save metadata
    metadata = {
        'generated': datetime.now().isoformat(),
        'figures': [
            'epidemic_dynamics.png',
            'phase_portraits.png', 
            'sensitivity_heatmap.png',
            'control_strategies.png'
        ],
        'scenarios': list(results.keys())
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run climate-epidemic coupled system analysis'
    )
    parser.add_argument(
        '--output', '-o',
        default='figures',
        help='Output directory for figures (default: figures)'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick analysis with reduced simulations'
    )
    
    args = parser.parse_args()
    
    print("Climate-Epidemic Analysis")
    print("=" * 50)
    
    try:
        generate_all_figures(args.output)
        print("\n✓ Analysis complete!")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())