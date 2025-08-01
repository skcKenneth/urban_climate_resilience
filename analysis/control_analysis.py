"""
Optimal control analysis with comprehensive visualizations
"""
import numpy as np
import json
import pickle
from pathlib import Path
from models.optimal_control import OptimalControlModel
from models.coupled_system import CoupledSystemModel
from utils.parameters import ModelParameters
from utils.data_generator import DataGenerator
from utils.visualization import SystemVisualizer
import logging

logger = logging.getLogger(__name__)

class ControlAnalysis:
    """Comprehensive control analysis with visualizations"""
    
    def __init__(self, params=None, results_dir='results'):
        self.params = params if params else ModelParameters()
        self.control_model = OptimalControlModel(params)
        self.coupled_model = CoupledSystemModel(params)
        self.visualizer = SystemVisualizer()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def run_control_comparison(self, t_span=(0, 365), quick_mode=False):
        """Run comprehensive control strategy comparison"""
        logger.info("Starting control strategy comparison analysis")
        
        # Generate initial conditions and functions
        climate_gen = DataGenerator()
        T_func = climate_gen.temperature_function()
        H_func = climate_gen.humidity_function()
        
        # Budget function
        budget_func = lambda t: 1000 * (1 + 0.5 * np.sin(2 * np.pi * t / 365))
        
        # Initial conditions
        N = self.params.N
        I0 = 100
        E0 = 50
        S0 = N - I0 - E0
        R0 = 0
        k0 = self.params.k_0
        C0 = self.params.C_0
        y0 = [S0, E0, I0, R0, k0, C0]
        
        # Run comparison
        results = self.control_model.compare_strategies(
            t_span, y0, T_func, H_func, budget_func, quick_mode=quick_mode
        )
        
        # Save results
        results_file = self.results_dir / 'control_comparison_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Generate visualizations
        logger.info("Generating control analysis visualizations")
        
        # 1. Strategy comparison dashboard
        fig = self.visualizer.plot_strategy_comparison_dashboard(
            results, 
            save_path=str(self.results_dir / 'strategy_dashboard.png')
        )
        
        # 2. Control trajectories
        fig = self.visualizer.plot_control_trajectories(
            results,
            t_span,
            save_path=str(self.results_dir / 'control_trajectories.png')
        )
        
        # 3. 3D phase space
        fig = self.visualizer.plot_3d_phase_space(
            results,
            variables=['S', 'I', 'k_avg'],
            save_path=str(self.results_dir / 'phase_space_3d.png')
        )
        
        # 4. Phase portraits for different variable pairs
        variable_pairs = [['I', 'k_avg'], ['S', 'R'], ['I', 'C']]
        for i, vars in enumerate(variable_pairs):
            fig = self.visualizer.plot_phase_portrait(
                results,
                variables=vars,
                save_path=str(self.results_dir / f'phase_portrait_{i+1}.png')
            )
        
        # Save summary statistics
        summary = self._compute_summary_statistics(results)
        summary_file = self.results_dir / 'control_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Control analysis complete. Results saved to {self.results_dir}")
        return results, summary
    
    def run_optimal_control_sensitivity(self, parameter_ranges=None):
        """Analyze sensitivity of optimal control to parameter variations"""
        logger.info("Starting optimal control sensitivity analysis")
        
        if parameter_ranges is None:
            parameter_ranges = {
                'beta_0': (0.1, 0.5),
                'gamma': (0.05, 0.2),
                'k_0': (10, 30),
                'costs': [(0.5, 2.0), (0.5, 2.0), (0.5, 2.0)],
                'weights': [(0.2, 0.8), (0.2, 0.8), (0.1, 0.5)]
            }
        
        results = {}
        base_params = ModelParameters()
        
        for param_name, param_range in parameter_ranges.items():
            logger.info(f"Analyzing sensitivity to {param_name}")
            
            if isinstance(param_range[0], tuple):  # For vector parameters
                # Test extreme values for each component
                param_results = []
                for i, (low, high) in enumerate(param_range):
                    for value in [low, high]:
                        test_params = ModelParameters()
                        # Copy base parameters
                        for attr in dir(base_params):
                            if not attr.startswith('_'):
                                setattr(test_params, attr, getattr(base_params, attr))
                        
                        # Modify specific parameter
                        current_values = list(getattr(test_params, param_name))
                        current_values[i] = value
                        setattr(test_params, param_name, current_values)
                        
                        # Run control analysis
                        control_model = OptimalControlModel(test_params)
                        try:
                            opt_control, opt_value, _ = control_model.solve_optimal_control(
                                t_span=(0, 100),  # Shorter span for sensitivity
                                y0=[test_params.N-150, 50, 100, 0, test_params.k_0, test_params.C_0],
                                T_func=lambda t: 20 + 10*np.sin(2*np.pi*t/365),
                                H_func=lambda t: 0.6 + 0.2*np.sin(2*np.pi*t/365),
                                budget_func=lambda t: 1000,
                                complexity='low'
                            )
                            param_results.append({
                                'component': i,
                                'value': value,
                                'objective': opt_value
                            })
                        except Exception as e:
                            logger.warning(f"Failed for {param_name}[{i}]={value}: {e}")
                
                results[param_name] = param_results
            
            else:  # For scalar parameters
                param_results = []
                for value in np.linspace(param_range[0], param_range[1], 5):
                    test_params = ModelParameters()
                    # Copy base parameters
                    for attr in dir(base_params):
                        if not attr.startswith('_'):
                            setattr(test_params, attr, getattr(base_params, attr))
                    
                    setattr(test_params, param_name, value)
                    
                    # Run control analysis
                    control_model = OptimalControlModel(test_params)
                    try:
                        opt_control, opt_value, _ = control_model.solve_optimal_control(
                            t_span=(0, 100),
                            y0=[test_params.N-150, 50, 100, 0, test_params.k_0, test_params.C_0],
                            T_func=lambda t: 20 + 10*np.sin(2*np.pi*t/365),
                            H_func=lambda t: 0.6 + 0.2*np.sin(2*np.pi*t/365),
                            budget_func=lambda t: 1000,
                            complexity='low'
                        )
                        param_results.append({
                            'value': value,
                            'objective': opt_value
                        })
                    except Exception as e:
                        logger.warning(f"Failed for {param_name}={value}: {e}")
                
                results[param_name] = param_results
        
        # Save results
        results_file = self.results_dir / 'control_sensitivity_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _compute_summary_statistics(self, results):
        """Compute summary statistics for control strategies"""
        summary = {}
        
        for strategy, data in results.items():
            if 'error' not in data:
                summary[strategy] = {
                    'total_infections': float(data.get('total_infections', 0)),
                    'peak_infections': float(data.get('peak_infections', 0)),
                    'final_network_degree': float(data.get('final_network_degree', 0)),
                    'avg_clustering': float(data.get('avg_clustering', 0)),
                    'avg_resilience': float(np.mean(data.get('resilience_scores', [0]))),
                    'min_resilience': float(np.min(data.get('resilience_scores', [0]))),
                    'performance_score': self._calculate_performance_score(data)
                }
                
                # Add control effort metrics for optimal strategy
                if strategy == 'optimal' and 'control_values' in data:
                    control_values = np.array(data['control_values'])
                    summary[strategy]['avg_medical_control'] = float(np.mean(control_values[:, 0]))
                    summary[strategy]['avg_social_control'] = float(np.mean(control_values[:, 1]))
                    summary[strategy]['avg_climate_control'] = float(np.mean(control_values[:, 2]))
                    summary[strategy]['total_control_effort'] = float(np.sum(control_values))
        
        # Rank strategies
        ranked_strategies = sorted(
            summary.items(),
            key=lambda x: x[1].get('performance_score', 0),
            reverse=True
        )
        
        for rank, (strategy, _) in enumerate(ranked_strategies):
            summary[strategy]['rank'] = rank + 1
        
        return summary
    
    def _calculate_performance_score(self, data):
        """Calculate overall performance score for a strategy"""
        # Normalize metrics (lower is better for infections, higher for network/resilience)
        infection_score = 1 / (1 + data.get('total_infections', 1e6))
        network_score = data.get('final_network_degree', 0) / self.params.k_0
        resilience_score = np.mean(data.get('resilience_scores', [0]))
        
        # Weighted average
        weights = [0.4, 0.3, 0.3]  # Infection, network, resilience
        score = (weights[0] * infection_score + 
                weights[1] * network_score + 
                weights[2] * resilience_score)
        
        return float(score)
    
    def generate_publication_figures(self, results=None):
        """Generate all figures needed for publication"""
        logger.info("Generating publication-ready figures")
        
        # Load results if not provided
        if results is None:
            results_file = self.results_dir / 'control_comparison_results.pkl'
            if results_file.exists():
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
            else:
                logger.error("No results found. Run control comparison first.")
                return
        
        # Create publication directory
        pub_dir = Path('figures/publication')
        pub_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate complete figure set
        figures = self.visualizer.create_publication_figure_set(
            {'control_comparison': results},
            save_dir=str(pub_dir)
        )
        
        # Generate LaTeX code
        self.visualizer.generate_latex_figure_code(
            list(figures.keys()),
            save_path=str(pub_dir / 'latex_figures.tex')
        )
        
        logger.info(f"Publication figures saved to {pub_dir}")
        return figures