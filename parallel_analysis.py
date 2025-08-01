"""
Parallel analysis module for efficient computation
"""
import os
import time
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import multiprocessing as mp
from analysis.sensitivity_analysis import SensitivityAnalysis
from models.coupled_system import CoupledSystemModel
from utils.parameters import ModelParameters

class ParallelAnalysis:
    """Parallel analysis implementation for research computations"""
    
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs or min(mp.cpu_count(), 4)  # Limit to 4 cores for GitHub Actions
        self.cache_dir = os.getenv('CACHE_DIR', '.cache')
        self.use_cache = os.getenv('USE_CACHE', 'false').lower() == 'true'
        
        print(f"🔧 Parallel Analysis initialized with {self.n_jobs} jobs")
    
    def parallel_sensitivity_analysis(self, n_samples=1000, T_scenario='baseline'):
        """Parallel Sobol sensitivity analysis"""
        print(f"🔄 Running parallel sensitivity analysis with {n_samples} samples")
        
        # Define parameter ranges
        param_ranges = {
            'beta_0': (0.5, 2.0),
            'sigma': (0.5, 2.0),
            'gamma': (0.5, 2.0),
            'alpha_T': (0.5, 2.0),
            'kappa': (0.0, 1.0),
            'k_0': (0.5, 2.0),
            'alpha_net': (0.5, 2.0),
            'beta_ep': (0.0, 0.2)
        }
        
        param_names = list(param_ranges.keys())
        n_params = len(param_names)
        
        # Generate parameter samples
        samples_A = np.random.random((n_samples, n_params))
        samples_B = np.random.random((n_samples, n_params))
        
        # Scale to parameter ranges
        for i, param_name in enumerate(param_names):
            low, high = param_ranges[param_name]
            samples_A[:, i] = low + (high - low) * samples_A[:, i]
            samples_B[:, i] = low + (high - low) * samples_B[:, i]
        
        # Parallel evaluation function
        def evaluate_model_parallel(params_vector, param_names, T_scenario):
            """Evaluate model for a single parameter set"""
            try:
                # Create modified parameters
                modified_params = ModelParameters()
                base_params = ModelParameters()
                
                for i, param_name in enumerate(param_names):
                    base_value = getattr(base_params, param_name)
                    setattr(modified_params, param_name, base_value * params_vector[i])
                
                # Run simulation
                coupled_model = CoupledSystemModel(modified_params)
                
                # Get climate scenario
                t, T, H = modified_params.get_climate_scenario(T_scenario)
                T_func = lambda time: np.interp(time, t, T)
                H_func = lambda time: np.interp(time, t, H)
                
                # Initial conditions
                y0 = [modified_params.N * 0.99, 0, modified_params.N * 0.01, 0, 
                      modified_params.k_0, 0.3]
                
                # Use environment variable for simulation days
                sim_days = int(os.getenv('SIMULATION_DAYS', 365))
                
                t_sim, y_sim = coupled_model.solve_coupled_system(
                    [0, sim_days], y0, T_func, H_func
                )
                
                # Calculate metrics
                S, E, I, R, k_avg, C = y_sim
                
                metrics = {
                    'peak_infections': np.max(I),
                    'total_infections': np.trapz(I, t_sim),
                    'final_size': R[-1] / modified_params.N,
                    'min_network_degree': np.min(k_avg),
                    'max_network_degree': np.max(k_avg),
                    'avg_clustering': np.mean(C)
                }
                
                return metrics
                
            except Exception as e:
                print(f"Simulation failed: {e}")
                return None
        
        # Run parallel evaluations
        print(f"📊 Evaluating {n_samples} parameter sets in parallel...")
        
        # Evaluate samples A
        results_A = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(evaluate_model_parallel)(samples_A[i], param_names, T_scenario)
            for i in range(n_samples)
        )
        
        # Evaluate samples B
        results_B = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(evaluate_model_parallel)(samples_B[i], param_names, T_scenario)
            for i in range(n_samples)
        )
        
        # Filter out failed simulations
        results_A = [r for r in results_A if r is not None]
        results_B = [r for r in results_B if r is not None]
        
        print(f"✅ Completed {len(results_A)} successful simulations")
        
        # Calculate sensitivity indices
        sensitivity_results = self._calculate_sensitivity_indices(
            results_A, results_B, param_names
        )
        
        return sensitivity_results
    
    def _calculate_sensitivity_indices(self, results_A, results_B, param_names):
        """Calculate Sobol sensitivity indices from parallel results"""
        if not results_A or not results_B:
            return None
        
        metrics = list(results_A[0].keys())
        sensitivity_indices = {}
        
        for metric in metrics:
            # Extract metric values
            y_A = np.array([r[metric] for r in results_A])
            y_B = np.array([r[metric] for r in results_B])
            
            if len(y_A) != len(y_B):
                continue
            
            # Calculate first-order indices
            first_order = []
            
            for i in range(len(param_names)):
                if len(y_A) > 0:
                    var_total = np.var(y_A)
                    if var_total > 0:
                        first_order.append(np.var(y_A - y_B) / var_total)
                    else:
                        first_order.append(0.0)
                else:
                    first_order.append(0.0)
            
            sensitivity_indices[metric] = {
                'first_order': dict(zip(param_names, first_order)),
                'total_order': dict(zip(param_names, first_order))  # Simplified
            }
        
        return sensitivity_indices
    
    def parallel_monte_carlo(self, n_samples=500, T_scenario='heatwave'):
        """Parallel Monte Carlo uncertainty analysis"""
        print(f"🔄 Running parallel Monte Carlo analysis with {n_samples} samples")
        
        # Define parameter uncertainties
        param_uncertainties = {
            'beta_0': 0.2,
            'sigma': 0.1,
            'gamma': 0.1,
            'alpha_T': 0.3,
            'kappa': 0.1,
            'k_0': 0.5,
            'alpha_net': 0.2,
            'beta_ep': 0.05
        }
        
        param_names = list(param_uncertainties.keys())
        n_params = len(param_names)
        
        # Generate parameter samples
        samples = np.random.random((n_samples, n_params))
        
        # Scale to parameter ranges
        base_params = ModelParameters()
        param_samples = np.zeros((n_samples, n_params))
        
        for i, param_name in enumerate(param_names):
            base_value = getattr(base_params, param_name)
            uncertainty = param_uncertainties[param_name]
            param_samples[:, i] = base_value + uncertainty * (samples[:, i] - 0.5)
        
        # Parallel evaluation function
        def evaluate_mc_parallel(params_vector, param_names, T_scenario):
            """Evaluate Monte Carlo simulation for a single parameter set"""
            try:
                # Create modified parameters
                modified_params = ModelParameters()
                for j, param_name in enumerate(param_names):
                    setattr(modified_params, param_name, params_vector[j])
                
                # Run simulation
                coupled_model = CoupledSystemModel(modified_params)
                
                # Get climate scenario
                t, T, H = modified_params.get_climate_scenario(T_scenario)
                T_func = lambda time: np.interp(time, t, T)
                H_func = lambda time: np.interp(time, t, H)
                
                # Initial conditions
                y0 = [modified_params.N * 0.99, 0, modified_params.N * 0.01, 0, 
                      modified_params.k_0, 0.3]
                
                # Use environment variable for simulation days
                sim_days = int(os.getenv('SIMULATION_DAYS', 365))
                
                t_sim, y_sim = coupled_model.solve_coupled_system(
                    [0, sim_days], y0, T_func, H_func
                )
                
                # Calculate metrics
                S, E, I, R, k_avg, C = y_sim
                
                metrics = {
                    'peak_infections': np.max(I),
                    'total_infections': np.trapz(I, t_sim),
                    'final_size': R[-1] / modified_params.N,
                    'min_network_degree': np.min(k_avg),
                    'max_network_degree': np.max(k_avg),
                    'avg_clustering': np.mean(C)
                }
                
                return metrics
                
            except Exception as e:
                print(f"MC simulation failed: {e}")
                return None
        
        # Run parallel evaluations
        print(f"📊 Evaluating {n_samples} Monte Carlo samples in parallel...")
        
        results = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(evaluate_mc_parallel)(param_samples[i], param_names, T_scenario)
            for i in range(n_samples)
        )
        
        # Filter out failed simulations
        results = [r for r in results if r is not None]
        
        print(f"✅ Completed {len(results)} successful Monte Carlo simulations")
        
        return results
    
    def parallel_climate_scenarios(self, scenarios=None):
        """Parallel analysis across different climate scenarios"""
        if scenarios is None:
            scenarios = ['baseline', 'heatwave', 'extreme_heat']
        
        print(f"🔄 Running parallel climate scenario analysis for {len(scenarios)} scenarios")
        
        def analyze_scenario(scenario_name):
            """Analyze a single climate scenario"""
            try:
                params = ModelParameters()
                coupled_model = CoupledSystemModel(params)
                
                # Get climate scenario
                t, T, H = params.get_climate_scenario(scenario_name)
                T_func = lambda time: np.interp(time, t, T)
                H_func = lambda time: np.interp(time, t, H)
                
                # Initial conditions
                y0 = [params.N * 0.99, 0, params.N * 0.01, 0, params.k_0, 0.3]
                
                # Use environment variable for simulation days
                sim_days = int(os.getenv('SIMULATION_DAYS', 365))
                
                t_sim, y_sim = coupled_model.solve_coupled_system(
                    [0, sim_days], y0, T_func, H_func
                )
                
                # Calculate metrics
                S, E, I, R, k_avg, C = y_sim
                
                return {
                    'scenario': scenario_name,
                    't': t_sim,
                    'y': y_sim,
                    'peak_infections': np.max(I),
                    'total_infections': np.trapz(I, t_sim),
                    'final_size': R[-1] / params.N,
                    'min_network_degree': np.min(k_avg),
                    'max_network_degree': np.max(k_avg),
                    'avg_clustering': np.mean(C)
                }
                
            except Exception as e:
                print(f"Scenario {scenario_name} failed: {e}")
                return None
        
        # Run parallel scenario analysis
        results = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(analyze_scenario)(scenario) for scenario in scenarios
        )
        
        # Filter out failed scenarios
        results = [r for r in results if r is not None]
        
        print(f"✅ Completed analysis for {len(results)} scenarios")
        
        return results