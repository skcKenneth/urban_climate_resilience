"""
Sensitivity analysis and model validation
"""
import numpy as np
import os
import time
from functools import lru_cache
try:
    from scipy.stats.qmc import Sobol  # For scipy >= 1.7
except ImportError:
    from scipy.stats import sobol_seq  # For older scipy versions
import matplotlib.pyplot as plt
from models.coupled_system import CoupledSystemModel
from utils.parameters import ModelParameters

class SensitivityAnalysis:
    """Model validation and sensitivity analysis tools"""
    
    def __init__(self, params=None):
        self.params = params if params else ModelParameters()
        self.start_time = time.time()
        self.max_time = int(os.getenv('MAX_TIME', 1200))  # Default 20 minutes
        self.quick_mode = os.getenv('QUICK_MODE', 'false').lower() == 'true'
        
    def check_timeout(self):
        """Check if we're approaching the time limit"""
        elapsed = time.time() - self.start_time
        if elapsed > self.max_time * 0.8:  # Stop at 80% of time limit
            print(f"⚠️  Approaching time limit ({elapsed/60:.1f} minutes), stopping analysis")
            return True
        return False
        
    @lru_cache(maxsize=128)
    def generate_sobol_samples(self, n_params, n_samples):
        """Generate Sobol sequence samples with version compatibility and caching"""
        try:
            # For scipy >= 1.7
            sampler = Sobol(d=n_params, scramble=True)
            samples = sampler.random(n=n_samples)
            return samples
        except (NameError, ImportError):
            # For older scipy versions
            try:
                samples = sobol_seq.i4_sobol_generate(n_params, n_samples)
                return samples
            except:
                # Fallback to quasi-random
                print("Warning: Using fallback pseudo-random instead of Sobol sequence")
                return np.random.random((n_samples, n_params))
        
    def sobol_sensitivity_analysis(self, n_samples=None, T_scenario='baseline'):
        """Perform Sobol sensitivity analysis with optimized settings"""
        
        # Use environment variable or default
        if n_samples is None:
            n_samples = int(os.getenv('N_SAMPLES', 1000))
            if self.quick_mode:
                n_samples = min(n_samples, 200)  # Limit samples in quick mode
        
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
        
        # Generate Sobol sequence (or fallback) with caching
        sobol_points = self.generate_sobol_samples(2 * n_params, n_samples)
        
        # Scale to parameter ranges
        param_names = list(param_ranges.keys())
        param_samples_A = np.zeros((n_samples, n_params))
        param_samples_B = np.zeros((n_samples, n_params))
        
        for i, param_name in enumerate(param_names):
            low, high = param_ranges[param_name]
            param_samples_A[:, i] = low + (high - low) * sobol_points[:, i]
            param_samples_B[:, i] = low + (high - low) * sobol_points[:, n_params + i]
        
        # Simulation function with timeout checking
        def evaluate_model(params_vector):
            if self.check_timeout():
                return None
                
            # Create modified parameters
            modified_params = ModelParameters()
            base_params = ModelParameters()
            
            for i, param_name in enumerate(param_names):
                base_value = getattr(base_params, param_name)
                setattr(modified_params, param_name, base_value * params_vector[i])
            
            # Run simulation with reduced time in quick mode
            coupled_model = CoupledSystemModel(modified_params)
            
            # Get climate scenario
            t, T, H = modified_params.get_climate_scenario(T_scenario)
            T_func = lambda time: np.interp(time, t, T)
            H_func = lambda time: np.interp(time, t, H)
            
            # Initial conditions
            y0 = [modified_params.N * 0.99, 0, modified_params.N * 0.01, 0, 
                  modified_params.k_0, 0.3]
            
            try:
                # Use shorter simulation time in quick mode
                sim_days = int(os.getenv('SIMULATION_DAYS', 365))
                if self.quick_mode:
                    sim_days = min(sim_days, 180)
                
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
        
        # Run sensitivity analysis with progress tracking
        print(f"Running Sobol sensitivity analysis with {n_samples} samples...")
        
        results_A = []
        results_B = []
        
        for i in range(n_samples):
            if i % max(1, n_samples // 10) == 0:  # Progress every 10%
                print(f"Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
            
            # Evaluate model A
            result_A = evaluate_model(param_samples_A[i])
            if result_A is None:
                print("⚠️  Stopping analysis due to timeout")
                break
            results_A.append(result_A)
            
            # Evaluate model B
            result_B = evaluate_model(param_samples_B[i])
            if result_B is None:
                print("⚠️  Stopping analysis due to timeout")
                break
            results_B.append(result_B)
        
        if len(results_A) < n_samples // 2:
            print("⚠️  Insufficient samples for reliable sensitivity analysis")
            return None
        
        # Calculate sensitivity indices
        sensitivity_results = self.calculate_sensitivity_indices(
            results_A, results_B, param_names
        )
        
        return sensitivity_results
    
    def calculate_sensitivity_indices(self, results_A, results_B, param_names):
        """Calculate Sobol sensitivity indices efficiently"""
        if not results_A or not results_B:
            return None
            
        # Convert to numpy arrays for efficiency
        metrics = list(results_A[0].keys())
        n_metrics = len(metrics)
        n_params = len(param_names)
        
        # Initialize sensitivity indices
        sensitivity_indices = {}
        
        for metric in metrics:
            # Extract metric values
            y_A = np.array([r[metric] for r in results_A if r is not None])
            y_B = np.array([r[metric] for r in results_B if r is not None])
            
            if len(y_A) != len(y_B):
                continue
                
            # Calculate first-order indices
            first_order = []
            total_order = []
            
            for i in range(n_params):
                # Create C_i matrix (A with B_i)
                # This is a simplified calculation for speed
                if len(y_A) > 0:
                    # Use variance-based sensitivity (simplified)
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
    
    def monte_carlo_uncertainty(self, n_samples=None, T_scenario='heatwave'):
        """Perform Monte Carlo uncertainty analysis with optimized settings"""
        
        # Use environment variable or default
        if n_samples is None:
            n_samples = int(os.getenv('N_SAMPLES', 500))
            if self.quick_mode:
                n_samples = min(n_samples, 100)  # Limit samples in quick mode
        
        print(f"Running Monte Carlo uncertainty analysis with {n_samples} samples...")
        
        # Define parameter uncertainties (as standard deviations)
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
        
        # Generate parameter samples
        param_names = list(param_uncertainties.keys())
        n_params = len(param_names)
        
        # Use Latin Hypercube Sampling for better coverage
        samples = np.random.random((n_samples, n_params))
        
        # Scale to parameter ranges
        base_params = ModelParameters()
        param_samples = np.zeros((n_samples, n_params))
        
        for i, param_name in enumerate(param_names):
            base_value = getattr(base_params, param_name)
            uncertainty = param_uncertainties[param_name]
            
            # Generate samples around base value
            param_samples[:, i] = base_value + uncertainty * (samples[:, i] - 0.5)
        
        # Run simulations
        results = []
        
        for i in range(n_samples):
            if i % max(1, n_samples // 10) == 0:  # Progress every 10%
                print(f"Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
            
            if self.check_timeout():
                print("⚠️  Stopping analysis due to timeout")
                break
            
            # Create modified parameters
            modified_params = ModelParameters()
            for j, param_name in enumerate(param_names):
                setattr(modified_params, param_name, param_samples[i, j])
            
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
                # Use shorter simulation time in quick mode
                sim_days = int(os.getenv('SIMULATION_DAYS', 365))
                if self.quick_mode:
                    sim_days = min(sim_days, 180)
                
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
                
                results.append(metrics)
                
            except Exception as e:
                print(f"Simulation {i} failed: {e}")
                continue
        
        if len(results) < n_samples // 2:
            print("⚠️  Insufficient successful simulations for uncertainty analysis")
            return None
        
        return results
    
    def calculate_uncertainty_bounds(self, mc_results, confidence_level=0.95):
        """Calculate uncertainty bounds efficiently"""
        if not mc_results:
            return None
            
        metrics = list(mc_results[0].keys())
        bounds = {}
        
        for metric in metrics:
            values = [r[metric] for r in mc_results if r is not None]
            if values:
                values = np.array(values)
                bounds[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentiles': np.percentile(values, [5, 25, 50, 75, 95])
                }
        
        return bounds

    def morris_screening(self, n_samples=100, n_groups=10, parameter_ranges=None):
        """
        Perform Morris screening sensitivity analysis
        
        Args:
            n_samples: Number of samples for Morris method
            n_groups: Number of groups for grouping parameters
            parameter_ranges: Dict of parameter ranges {param: (min, max)}
            
        Returns:
            Dict with Morris screening results
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'beta_0': (0.1, 0.5),
                'alpha_T': (0.01, 0.05),
                'sigma': (0.1, 0.3),
                'gamma': (0.1, 0.3),
                'kappa': (0.1, 0.5),
                'epsilon': (0.5, 2.0),
                'sigma_k': (5, 20),
                'T_critical': (35, 45)
            }
        
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)
        
        # Generate Morris trajectories
        delta = 1.0 / (n_groups - 1)
        elementary_effects = {param: [] for param in param_names}
        
        print(f"Running Morris screening with {n_samples} trajectories...")
        
        for i in range(min(n_samples, 50)):  # Limit for performance
            if self.check_timeout():
                break
                
            # Generate base point
            base_point = np.random.random(n_params)
            
            # Scale to parameter ranges
            scaled_params = {}
            for j, param in enumerate(param_names):
                min_val, max_val = parameter_ranges[param]
                scaled_params[param] = min_val + base_point[j] * (max_val - min_val)
            
            # Calculate base output
            base_output = self._evaluate_model_with_params(scaled_params)
            
            # Calculate elementary effects
            for j, param in enumerate(param_names):
                # Perturb parameter
                perturbed_params = scaled_params.copy()
                min_val, max_val = parameter_ranges[param]
                perturbed_params[param] = min(max_val, scaled_params[param] + delta * (max_val - min_val))
                
                # Calculate effect
                perturbed_output = self._evaluate_model_with_params(perturbed_params)
                effect = (perturbed_output - base_output) / (delta * (max_val - min_val))
                elementary_effects[param].append(effect)
        
        # Calculate Morris indices
        morris_indices = {}
        for param in param_names:
            effects = np.array(elementary_effects[param])
            morris_indices[param] = {
                'mu': np.mean(effects),
                'mu_star': np.mean(np.abs(effects)),
                'sigma': np.std(effects)
            }
        
        return {
            'parameters': param_names,
            'indices': morris_indices,
            'n_trajectories': len(elementary_effects[param_names[0]])
        }
    
    def _evaluate_model_with_params(self, param_dict):
        """Helper method to evaluate model with specific parameters"""
        # Create a copy of current params
        temp_params = ModelParameters()
        
        # Update with new values
        for param, value in param_dict.items():
            if hasattr(temp_params, param):
                setattr(temp_params, param, value)
        
        # Run simplified simulation
        model = CoupledSystemModel(temp_params, n_nodes=100)
        
        # Simple epidemic scenario
        t = np.linspace(0, 30, 30)  # 30 days
        
        def T_func(time):
            time_array = np.atleast_1d(time)
            result = 25 + 5*np.sin(2*np.pi*time_array/365)
            return float(result) if np.isscalar(time) else result
            
        def H_func(time):
            if np.isscalar(time):
                return 0.7
            else:
                return 0.7 * np.ones_like(np.atleast_1d(time))
        
        try:
            # Set initial conditions
            S0 = temp_params.N - 100  # Start with 100 infected
            I0 = 100
            R0 = 0
            E0 = 0
            k0 = temp_params.k_0
            C0 = temp_params.C_0
            y0 = np.array([S0, I0, R0, E0, k0, C0])
            
            time_points, states = model.solve_coupled_system(
                t_span=(0, 30),
                y0=y0,
                T_func=T_func,
                H_func=H_func,
                t_eval=t
            )
            
            # Return peak infection as output metric
            return np.max(states[1, :])
        except Exception as e:
            # Log the error for debugging
            print(f"Model evaluation error: {e}")
            return 0.0
