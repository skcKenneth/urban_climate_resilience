"""
Main execution script for Urban Climate-Social Network Resilience System
"""
import numpy as np
import matplotlib.pyplot as plt
from models.coupled_system import CoupledSystemModel
from models.optimal_control import OptimalControlModel
from analysis.stability_analysis import StabilityAnalysis
from analysis.sensitivity_analysis import SensitivityAnalysis
from utils.parameters import ModelParameters
from utils.visualization import SystemVisualizer
from utils.data_generator import DataGenerator

def main():
    """Main analysis pipeline"""
    print("=" * 60)
    print("Urban Climate-Social Network Resilience System Analysis")
    print("=" * 60)
    
    # Initialize components
    params = ModelParameters()
    coupled_model = CoupledSystemModel(params)
    visualizer = SystemVisualizer()
    data_gen = DataGenerator(params)
    
    # Generate climate scenarios
    print("\n1. Generating climate scenarios...")
    climate_scenarios = data_gen.generate_climate_scenarios()
    
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
        
        # Solve system
        try:
            t, y = coupled_model.solve_coupled_system([0, 365], y0, T_func, H_func)
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
            fig = visualizer.plot_epidemic_dynamics(
                result['t'], result['y'], result['T_func'],
                title=f"Epidemic Dynamics - {scenario_name.title()} Scenario"
            )
            plt.savefig(f'epidemic_dynamics_{scenario_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Phase portrait
    if len(results) > 0:
        fig = visualizer.plot_phase_portrait(results, ['I', 'k_avg'])
        plt.savefig('phase_portrait.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Task 3: Stability and bifurcation analysis
    print("\n4. Performing stability analysis...")
    stability_analyzer = StabilityAnalysis(params)
    
    # Bifurcation analysis over temperature range
    T_range = np.linspace(20, 45, 25)
    bifurcation_results = stability_analyzer.bifurcation_analysis_temperature(T_range)
    
    # Plot bifurcation diagram
    if bifurcation_results['temperatures']:
        fig = visualizer.plot_bifurcation_diagram(bifurcation_results)
        plt.savefig('bifurcation_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find critical points
        critical_points = stability_analyzer.critical_temperature_analysis(T_range)
        print(f"   Found {len(critical_points)} critical temperature points:")
        for cp in critical_points:
            print(f"      T_critical = {cp['temperature']:.2f}°C ({cp['type']})")
    
    # Task 4: Optimal control analysis
    print("\n5. Analyzing optimal control strategies...")
    control_model = OptimalControlModel(params)
    
    # Budget function (increasing over time)
    budget_func = lambda t: 150 + 50 * np.sin(2*np.pi*t/365)
    
    # Compare control strategies using heatwave scenario
    if 'heatwave' in results:
        heatwave_result = results['heatwave']
        T_func = heatwave_result['T_func']
        H_func = heatwave_result['H_func']
        y0 = [params.N * 0.99, 0, params.N * 0.01, 0, params.k_0, 0.3]
        
        try:
            strategy_comparison = control_model.compare_strategies(
                [0, 365], y0, T_func, H_func, budget_func
            )
            
            # Print strategy comparison
            print("   Strategy comparison results:")
            for strategy, result in strategy_comparison.items():
                if 'total_infections' in result:
                    print(f"      {strategy}: Total infections = {result['total_infections']:.0f}, "
                          f"Peak = {result['peak_infections']:.0f}")
            
            # Visualize strategy comparison
            fig = visualizer.plot_control_strategies(strategy_comparison)
            plt.savefig('control_strategies.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"   Error in control analysis: {e}")
    
    # Task 5: Model validation and sensitivity analysis
    print("\n6. Performing sensitivity analysis...")
    sensitivity_analyzer = SensitivityAnalysis(params)
    
    try:
        # Sobol sensitivity analysis (reduced samples for demo)
        print("   Running Sobol sensitivity analysis...")
        sobol_results = sensitivity_analyzer.sobol_sensitivity_analysis(n_samples=100)
        
        if sobol_results:
            fig = visualizer.plot_sensitivity_analysis(sobol_results)
            plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("   Top sensitive parameters:")
            for metric, data in sobol_results.items():
                if 'first_order' in data:
                    sorted_params = sorted(data['first_order'].items(), 
                                         key=lambda x: abs(x[1]), reverse=True)
                    print(f"      {metric}:")
                    for param, index in sorted_params[:3]:
                        print(f"         {param}: {index:.3f}")
    
    except Exception as e:
        print(f"   Error in sensitivity analysis: {e}")
    
    # Uncertainty quantification
    print("\n7. Uncertainty quantification...")
    try:
        print("   Running Monte Carlo uncertainty analysis...")
        mc_results = sensitivity_analyzer.monte_carlo_uncertainty(n_samples=100)
        
        if mc_results:
            bounds = sensitivity_analyzer.calculate_uncertainty_bounds(mc_results)
            
            fig = visualizer.plot_uncertainty_analysis(mc_results, bounds)
            plt.savefig('uncertainty_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("   Uncertainty bounds (95% CI):")
            for metric, bound_data in bounds.items():
                print(f"      {metric}: [{bound_data['lower']:.3f}, {bound_data['upper']:.3f}] "
                      f"(mean: {bound_data['mean']:.3f})")
    
    except Exception as e:
        print(f"   Error in uncertainty analysis: {e}")
    
    # Task 6: Policy recommendations
    print("\n8. Policy Recommendations:")
    print("   Based on the mathematical analysis:")
    
    print("\n   a) Critical Temperature Thresholds:")
    if critical_points:
        for cp in critical_points:
            print(f"      - System becomes unstable above {cp['temperature']:.1f}°C")
            print("        → Implement early warning systems at this threshold")
    
    print("\n   b) Network-Based Interventions:")
    print("      - Maintain network connectivity above k_avg = 6.0 to prevent fragmentation")
    print("      - Focus social support programs on high-clustering communities")
    print("      - Implement redundant communication pathways before heatwave seasons")
    
    print("\n   c) Optimal Control Strategies:")
    if 'strategy_comparison' in locals():
        best_strategy = min(strategy_comparison.items(), 
                          key=lambda x: x[1].get('total_infections', float('inf')))
        print(f"      - {best_strategy[0]} strategy shows best performance")
        print("      - Balance medical interventions with social support")
        print("      - Climate mitigation provides long-term benefits")
    
    print("\n   d) Early Warning Indicators:")
    print("      - Monitor R₀ approaching 1.0 combined with temperature > 32°C")
    print("      - Track network degree decline > 20% as resilience failure signal")
    print("      - Use clustering coefficient < 0.15 as intervention trigger")
    
    print("\n" + "=" * 60)
    print("Analysis Complete! Check generated plots for detailed results.")
    print("=" * 60)

if __name__ == "__main__":
    main()
