"""
Sensitivity analysis and model validation
"""
import numpy as np
from scipy.stats import sobol_seq
import matplotlib.pyplot as plt
from models.coupled_system import CoupledSystemModel
from utils.parameters import ModelParameters

class SensitivityAnalysis:
    """Model validation and sensitivity analysis tools"""
    
    def __init__(self, params=None):
        self.params = params if params else ModelParameters()
        
    def sobol_sensitivity_analysis(self, n_samples=1000, T_scenario='baseline'):
        """Perform Sobol sensitivity analysis"""
        
        # Define parameter ranges (as multipliers of base values)
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
        
        n_params = len(param_ranges)
        
        # Generate Sobol sequence
        sobol_points = sobol_seq.i4_sobol_generate(2 * n_params, n_samples)
        
        # Scale to parameter ranges
        param_names = list(param_ranges.keys())
        param_samples_A = np.zeros((n_samples, n_params))
        param_samples_B = np.zeros((n_samples, n_params))
        
        for i, param_name in enumerate(param_names):
            low, high = param_ranges[param_name]
            param_samples_A[:, i] = low + (high - low) * sobol_points[:, i]
            param_samples_B[:, i] = low + (high - low) * sobol_points[:, n_params + i]
        
        # Simulation function
        def evaluate_model(params_vector):
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
            
            try:
                t_sim, y_sim = coupled_model.solve_coupled_system(
                    [0, 365], y0, T_func, H_func
                )
                
                # Calculate metrics
                S, E, I, R, k_avg, C = y_sim
                
                metrics = {
                    'peak_infections': np.max(I),
                    'total_infections': np.trapz(I, t_sim),
                    'final_size': R[-1] / modified_params.N,
                    'min_network_degree': np.min(k_avg),
                    'avg_clustering': np.mean(C)
                }
                
                return metrics
                
            except Exception as e:
                # Return default values if simulation fails
                return {
                    'peak_infections': np.nan,
                    'total_infections': np.nan,
                    'final_size': np.nan,
                    'min_network_degree': np.nan,
                    'avg_clustering': np.nan
                }
        
        # Evaluate model for all parameter combinations
        results_A = []
        results_B = []
        results_AB = []
        
        print("Running Sobol sensitivity analysis...")
        
        # Sample A
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"Sample A: {i}/{n_samples}")
            results_A.append(evaluate_model(param_samples_A[i]))
        
        # Sample B
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"Sample B: {i}/{n_samples}")
            results_B.append(evaluate_model(param_samples_B[i]))
        
        # Sample AB (substitute each parameter)
        for j in range(n_params):
            results_AB_j = []
            for i in range(n_samples):
                if i % 100 == 0:
                    print(f"Sample AB_{j}: {i}/{n_samples}")
                
                params_AB = param_samples_A[i].copy()
                params_AB[j] = param_samples_B[i, j]
                results_AB_j.append(evaluate_model(params_AB))
            results_AB.append(results_AB_j)
        
        # Calculate Sobol indices
        sobol_indices = {}
        
        metrics = list(results_A[0].keys())
        
        for metric in metrics:
            # Extract values
            Y_A = np.array([r[metric] for r in results_A])
            Y_B = np.array([r[metric] for r in results_B])
            
            # Remove NaN values
            valid_idx = ~(np.isnan(Y_A) | np.isnan(Y_B))
            if np.sum(valid_idx) < n_samples * 0.5:
                continue
                
            Y_A = Y_A[valid_idx]
            Y_B = Y_B[valid_idx]
            
            # Calculate total variance
            Y_total = np.concatenate([Y_A, Y_B])
            V_total = np.var(Y_total)
            
            if V_total == 0:
                continue
            
            # First-order indices
            S1 = np.zeros(n_params)
            for j in range(n_params):
                Y_AB_j = np.array([r[metric] for r in results_AB[j]])
                Y_AB_j = Y_AB_j[valid_idx]
                
                V_j = np.mean(Y_B * (Y_AB_j - Y_A))
                S1[j] = V_j / V_total
            
            sobol_indices[metric] = {
                'first_order': dict(zip(param_names, S1)),
                'total_variance': V_total
            }
        
        return sobol_indices
    
    def monte_carlo_uncertainty(self, n_samples=500, T_scenario='heatwave'):
        """Monte Carlo uncertainty quantification"""
        
        # Parameter uncertainty distributions (as coefficient of variation)
        param_uncertainty = {
            'beta_0': 0.3,
            'sigma': 0.2,
            'gamma': 0.2,
            'alpha_T': 0.5,
            'kappa': 0.4,
            'k_0': 0.2,
            'alpha_net': 0.3,
            'beta_ep': 0.5
        }
        
        base_params = ModelParameters()
        results = []
        
        print("Running Monte Carlo uncertainty analysis...")
        
        for i in range(n_samples):
            if i % 50 == 0:
                print(f"Sample {i}/{n_samples}")
            
            # Sample parameters
            modified_params = ModelParameters()
            
            for param_name, cv in param_uncertainty.items():
                base_value = getattr(base_params, param_name)
                # Log-normal distribution
                sigma_ln = np.sqrt(np.log(1 + cv**2))
                mu_ln = np.log(base_value) - 0.5 * sigma_ln**2
                new_value = np.random.lognormal(mu_ln, sigma_ln)
                setattr(modified_params, param_name, new_value)
            
            # Run simulation
            coupled_model = CoupledSystemModel(modified_params)
            
            # Get climate scenario
            t, T, H = modified_params.get_climate_scenario(T_scenario)
            T_func = lambda time: np.interp(time, t, T)
            H_func = lambda time: np.interp(time, t, H)
            
            # Initial conditions with uncertainty
            I0_uncertainty = np.random.uniform(0.005, 0.02)  # 0.5% to 2% initially infected
            y0 = [modified_params.N * (1 - I0_uncertainty), 0, 
                  modified_params.N * I0_uncertainty, 0, 
                  modified_params.k_0 * np.random.uniform(0.8, 1.2), 
                  np.random.uniform(0.2, 0.4)]
            
            try:
                t_sim, y_sim = coupled_model.solve_coupled_system(
                    [0, 365], y0, T_func, H_func
                )
                
                # Calculate outputs
                S, E, I, R, k_avg, C = y_sim
                
                # Calculate resilience metrics over time
                resilience_metrics = []
                for j in range(len(t_sim)):
                    resilience = coupled_model.calculate_system_resilience(
                        t_sim[j], y_sim[:, j], T_func
                    )
                    resilience_metrics.append(resilience['overall_resilience'])
                
                result = {
                    'peak_infections': np.max(I),
                    'total_infections': np.trapz(I, t_sim),
                    'attack_rate': R[-1] / modified_params.N,
                    'min_resilience': np.min(resilience_metrics),
                    'avg_resilience': np.mean(resilience_metrics),
                    'network_degradation': (modified_params.k_0 - np.min(k_avg)) / modified_params.k_0,
                    'time_to_peak': t_sim[np.argmax(I)]
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Simulation failed: {e}")
                continue
        
        return results
    
    def calculate_uncertainty_bounds(self, mc_results, confidence_level=0.95):
        """Calculate confidence intervals from Monte Carlo results"""
        
        if not mc_results:
            return {}
        
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        metrics = mc_results[0].keys()
        bounds = {}
        
        for metric in metrics:
            values = [r[metric] for r in mc_results if not np.isnan(r[metric])]
            
            if values:
                bounds[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'lower': np.percentile(values, lower_percentile),
                    'upper': np.percentile(values, upper_percentile),
                    'median': np.median(values)
                }
        
        return bounds
